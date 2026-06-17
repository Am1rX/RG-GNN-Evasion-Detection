import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, to_hetero
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (precision_recall_curve, precision_recall_fscore_support,
                             accuracy_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, confusion_matrix)
from graph_builder import build_hetero_graph

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PCAP_CLEAN = "data/202601311400.pcap"           
PCAP_OUT   = "data/thgnn_evaluation_dataset.pcap"
GT_CSV     = "data/ground_truth.csv"
DATASET_CSV = "data/dataset.csv"

SEEDS = [42, 123, 2024, 2025, 9999]
HIDDEN = 128
EPOCHS = 100
SPLIT_FRACS = (0.60, 0.10, 0.30)           


# ============================ Architectures ============================
class ProposedGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels, aggr='sum')
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels, aggr='sum')
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        h1 = self.dropout(F.gelu(self.norm1(self.conv1(x, edge_index))))
        h2 = F.gelu(self.norm2(self.conv2(h1, edge_index)))
        return self.classifier(h1 + h2)


class HomogeneousGCN(torch.nn.Module):
    def __init__(self, in_c, hid, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid)
        self.conv2 = GCNConv(hid, hid)
        self.cls = torch.nn.Linear(hid, out_c)

    def forward(self, x, ei):
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        return self.cls(x)


# ============================ Data & Splitting ============================
def load_labeled_graph(dataset_csv=DATASET_CSV, gt_csv=GT_CSV):
    data, df = build_hetero_graph(dataset_csv)
    gt = pd.read_csv(gt_csv)
    df['frame.number'] = pd.to_numeric(df['frame.number'], errors='coerce')
    gt['frame.number'] = pd.to_numeric(gt['frame.number'], errors='coerce')
    y = pd.merge(df, gt, on='frame.number', how='left')['is_evasion'].fillna(0).values.astype(int)
    data['packet'].y = torch.tensor(y, dtype=torch.long)
    return data, df, y


def make_split(df, y, fracs=SPLIT_FRACS):
    n = len(y)
    order = np.argsort(pd.to_numeric(df['frame.number'], errors='coerce').values, kind='stable')
    tr_end = int(fracs[0] * n)
    va_end = int((fracs[0] + fracs[1]) * n)
    train_idx, val_idx, test_idx = order[:tr_end], order[tr_end:va_end], order[va_end:]
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        pos = int(y[idx].sum())
        print(f"    {name:<6}: {len(idx):>9} nodes | {pos:>6} attacks")
        if pos == 0:
            raise RuntimeError(f"Split '{name}' has 0 attack samples; adjust ratios or injection settings.")
    return train_idx, val_idx, test_idx


def fit_scale(data, train_idx):
    raw = data['packet'].x.cpu().numpy()
    sc = RobustScaler().fit(raw[train_idx])
    x_clean = torch.tensor(sc.transform(raw), dtype=torch.float)
    return x_clean


def add_noise(x_clean, target_mask, std, seed):
    g = torch.Generator().manual_seed(seed)
    x = x_clean.clone()
    if std > 0:
        idx = torch.where(target_mask)[0]
        x[idx] = x[idx] + torch.randn(x[idx].shape, generator=g) * std
    return x


def set_masks(data, train_idx, val_idx, test_idx):
    n = data['packet'].num_nodes
    for k, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        m = torch.zeros(n, dtype=torch.bool); m[idx] = True
        data['packet'][f'{k}_mask'] = m


def class_weights_from(data):
    yl = data['packet'].y[data['packet'].train_mask]
    ratio = (yl == 0).sum().item() / ((yl == 1).sum().item() + 1e-5)
    return torch.tensor([1.0, ratio], dtype=torch.float).to(DEVICE), ratio


def tune_threshold(y_val, p_val):
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = (2 * prec * rec) / (prec + rec + 1e-10)
    ix = int(np.argmax(f1))
    return float(thr[ix]) if ix < len(thr) else 0.5


def full_report(y_true, probs, thr, t_ms):
    preds = (probs >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        'Accuracy': accuracy_score(y_true, preds),
        'Precision': p[0], 'Recall': r[0], 'F1-Score': f1[0],
        'ROC-AUC': roc_auc_score(y_true, probs),
        'PR-AUC': average_precision_score(y_true, probs),
        'MCC': matthews_corrcoef(y_true, preds),
        'FPR(%)': 100 * fp / (fp + tn) if (fp + tn) else 0.0,
        'FNR(%)': 100 * fn / (fn + tp) if (fn + tp) else 0.0,
        'Time(ms)': t_ms,
    }

def train_thgnn(data, w, seed, epochs=EPOCHS, return_model=False):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = to_hetero(ProposedGNN(HIDDEN, 2), data.metadata(), aggr='sum').to(DEVICE)
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)          # lazy init
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    tr = data['packet'].train_mask
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)['packet']
        loss = crit(out[tr], data['packet'].y[tr])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        t = time.time()
        out = model(data.x_dict, data.edge_index_dict)['packet']
        dt = (time.time() - t) * 1000
        prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    return (prob, dt, model) if return_model else (prob, dt)


def train_gcn(data, w, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    ei = torch.cat([data['packet', 'temporal', 'packet'].edge_index,
                    data['packet', 'reassembly', 'packet'].edge_index], dim=1).to(DEVICE)
    x, y = data['packet'].x, data['packet'].y
    tr = data['packet'].train_mask
    model = HomogeneousGCN(x.size(1), HIDDEN, 2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = crit(model(x, ei)[tr], y[tr])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        t = time.time(); out = model(x, ei); dt = (time.time() - t) * 1000
        prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    return prob, dt


def train_xgb(data, ratio, seed, feature_cols=None):
    import xgboost as xgb
    x = data['packet'].x.cpu().numpy()
    if feature_cols is not None:
        x = x[:, feature_cols]
    y = data['packet'].y.cpu().numpy()
    tr = data['packet'].train_mask.cpu().numpy()
    m = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=ratio, random_state=seed,
                          eval_metric='logloss', n_jobs=-1)
    m.fit(x[tr], y[tr])
    t = time.time(); prob = m.predict_proba(x)[:, 1]; dt = (time.time() - t) * 1000
    return prob, dt


def summarize(rows):
    d = pd.DataFrame(rows)
    for c in d.columns:
        print(f"    {c:<10} {d[c].mean():.4f} ± {d[c].std():.4f}")
    return d


def reassembly_only(data):
    import copy
    d = copy.deepcopy(data)
    d['packet', 'temporal', 'packet'].edge_index = torch.empty((2, 0), dtype=torch.long).to(DEVICE)
    return d