import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

REASSEMBLY_TIME_WINDOW = 5.0   # seconds

FEATURE_COLS = ['frame.len', 'ip.frag_offset', 'ip.flags.mf',
                'tcp.options.mss_val', 'tcp.seq']


def _parse_hex_id(val):
    try:
        if isinstance(val, str):
            val = val.strip()
            if val.lower().startswith('0x'):
                return int(val, 16)
        return float(val)
    except Exception:
        return -1


def build_hetero_graph(csv_path):
    print(f"[*] Loading traffic from {csv_path} ...")
    df = pd.read_csv(csv_path)

    if 'ip.flags.mf' in df.columns:
        df['ip.flags.mf'] = (df['ip.flags.mf'].astype(str).str.upper()
                             .map({'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0}).fillna(0))
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    x_raw = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float)
    data = HeteroData()
    data['packet'].x = x_raw
    num_nodes = len(df)

    if 'frame.time_epoch' in df.columns:
        df['_t'] = pd.to_numeric(df['frame.time_epoch'], errors='coerce').fillna(0.0)
    else:
        df['_t'] = np.arange(num_nodes, dtype=float)

    print("[*] Building flow-based (src, dst) temporal edges ...")
    has_ip = df['ip.src'].astype(str).ne('-1') & df['ip.dst'].astype(str).ne('-1')
    tmp = df[has_ip].copy()
    tmp['_idx'] = tmp.index
    tmp = tmp.sort_values(['ip.src', 'ip.dst', '_t'], kind='stable')
    s_ip = tmp['ip.src'].values
    d_ip = tmp['ip.dst'].values
    idxs = tmp['_idx'].values
    if len(idxs) > 1:
        same_flow = (s_ip[1:] == s_ip[:-1]) & (d_ip[1:] == d_ip[:-1])
        a = idxs[:-1][same_flow]
        b = idxs[1:][same_flow]
        src_t = np.concatenate([a, b])
        dst_t = np.concatenate([b, a])
        edge_t = torch.tensor(np.array([src_t, dst_t]), dtype=torch.long)
    else:
        edge_t = torch.empty((2, 0), dtype=torch.long)
    data['packet', 'temporal', 'packet'].edge_index = edge_t

    print("[*] Building Reassembly edges ...")
    df['ip.id'] = df['ip.id'].apply(_parse_hex_id).fillna(-1)
    is_frag = (df['ip.flags.mf'] == 1) | (df['ip.frag_offset'] > 0)
    valid_id = (df['ip.id'] != -1) & (df['ip.id'] != 0)
    frag = df[is_frag & valid_id].copy()
    frag['node_idx'] = frag.index

    src_r, dst_r = [], []
    for _, group in frag.groupby(['ip.src', 'ip.dst', 'ip.id']):
        g = group.sort_values('_t')
        cluster_id = (g['_t'].diff().fillna(0) > REASSEMBLY_TIME_WINDOW).cumsum()
        for _, sub in g.groupby(cluster_id):
            if len(sub) < 2:
                continue
            ordered = sub.sort_values('ip.frag_offset')['node_idx'].values
            for u, v in zip(ordered[:-1], ordered[1:]):
                src_r += [u, v]
                dst_r += [v, u]

    if src_r:
        edge_r = torch.tensor(np.array([src_r, dst_r]), dtype=torch.long)
    else:
        edge_r = torch.empty((2, 0), dtype=torch.long)
    data['packet', 'reassembly', 'packet'].edge_index = edge_r

    df.drop(columns=['_t'], inplace=True, errors='ignore')
    print(f"[+] Graph: {num_nodes} nodes | {edge_t.size(1)} temporal (flow) edges | "
          f"{edge_r.size(1)} reassembly edges.")
    return data, df