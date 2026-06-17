import torch
import th_utils as U


def run(robustness=False, noise_std=0.4):
    tag = "Robustness (Noise on train+test)" if robustness else "Standard Conditions"
    print(f"\n{'='*64}\n Proposed Model RG-GNN — {tag}\n{'='*64}")
    print(f"[*] Device: {U.DEVICE}")

    data, df, y = U.load_labeled_graph()
    print("[*] Temporal Split (Train/Val/Test):")
    tr, va, te = U.make_split(df, y)
    U.set_masks(data, tr, va, te)
    x_clean = U.fit_scale(data, tr)
    data = data.to(U.DEVICE); x_clean = x_clean.to(U.DEVICE)
    w, _ = U.class_weights_from(data)

    data_prop = U.reassembly_only(data)    # Proposed model: only reassembly edges
    noise_mask = torch.ones(data['packet'].num_nodes, dtype=torch.bool)

    va_np = data['packet'].val_mask.cpu().numpy()
    te_np = data['packet'].test_mask.cpu().numpy()
    y_val = data['packet'].y[data['packet'].val_mask].cpu().numpy()
    y_test = data['packet'].y[data['packet'].test_mask].cpu().numpy()

    rows = []
    for run_i, seed in enumerate(U.SEEDS):
        print(f"\n--- RUN {run_i+1}/{len(U.SEEDS)} | seed={seed} ---")
        x = U.add_noise(x_clean.cpu(), noise_mask, noise_std if robustness else 0.0, seed).to(U.DEVICE)
        data_prop['packet'].x = x
        prob, dt, model = U.train_thgnn(data_prop, w, seed, return_model=True)

        if run_i == 0:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Trainable Parameters: {params} | Approx Size: {params*4/1024**2:.2f} MB")

        thr = U.tune_threshold(y_val, prob[va_np])
        m = U.full_report(y_test, prob[te_np], thr, dt)
        rows.append(m)
        print(f"    F1={m['F1-Score']:.4f} | FNR={m['FNR(%)']:.2f}% | "
              f"AUC={m['ROC-AUC']:.4f} | thr={thr:.3f}")

    print(f"\n{'#'*64}\n Final Results over {len(U.SEEDS)} Runs (Mean ± Std)\n{'#'*64}")
    U.summarize(rows)
    torch.save(model.state_dict(), 'proposed_model_final.pth')
    print("[+] Saved: proposed_model_final.pth")


if __name__ == "__main__":
    run(robustness=False)   # Main results
    run(robustness=True)    # Robustness results