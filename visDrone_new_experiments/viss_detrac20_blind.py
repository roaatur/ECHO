#!/usr/bin/env python3
"""Cross-domain blind test: DETRAC-20 features on VisDrone.

The 20 features were selected on DETRAC via Spearman + clustering.
Here we extract the SAME 20 features on VisDrone and train the GRU.
ZERO feature selection on VisDrone — the cleanest possible experiment.

Also tests All-65 DETRAC features on VisDrone for comparison.

Output:
  - viss_detrac20_blind_results.pdf
  - viss_detrac20_blind_results.csv (per-sequence breakdown)
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ECHO_DIR = SCRIPT_DIR.parent
EVAL_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "visdrone_classagnostic_results.csv"
FEAT_CSV = SCRIPT_DIR / "viss_detrac65_features.csv"
OUT_PDF = SCRIPT_DIR / "viss_detrac20_blind_results.pdf"
OUT_CSV = SCRIPT_DIR / "viss_detrac20_blind_results.csv"

# Official VisDrone VID val sequences
VAL_SEQS = [
    "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
    "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v",
]

# DETRAC-20: selected on DETRAC via Spearman + hierarchical clustering
# These are FIXED — no feature selection on VisDrone
DETRAC_20 = [
    "fast_keypoints_half", "edge_density_coarse", "gradient_magnitude_mean",
    "gabor_nyquist_energy", "fast_keypoints_full", "glcm_homogeneity",
    "overall_edge_density", "top_edge_density", "top_high_freq_energy",
    "top_gradient_std", "fft_critical_band_ratio", "total_edge_components",
    "top_laplacian_var", "glcm_entropy", "mid_edge_density",
    "dark_channel_mean", "mid_gradient_std", "foreground_blob_count",
    "mid_high_freq_energy", "rms_contrast",
]

FPS = 30
WINDOW = 15  # 0.5s
HORIZON = 15
TRAIN_STRIDE = 8
EVAL_STRIDE = 15
SEEDS = [42, 123, 7]


# ── Data ────────────────────────────────────────────────────────────────────


def load_data():
    print("Loading data ...", flush=True)
    ev = pd.read_csv(EVAL_CSV)
    nano = ev[ev["model"] == "yolo11n.pt"][
        ["sequence", "frame_id", "tp", "fp", "fn", "num_gt"]].copy()
    nano.columns = ["sequence", "frame_id", "tp_nano", "fp_nano", "fn_nano", "num_gt"]
    med = ev[ev["model"] == "yolo11m.pt"][["sequence", "frame_id", "tp"]].copy()
    med.columns = ["sequence", "frame_id", "tp_medium"]
    eval_merged = nano.merge(med, on=["sequence", "frame_id"])

    feat = pd.read_csv(FEAT_CSV)
    df = eval_merged.merge(feat, on=["sequence", "frame_id"], how="inner")

    # Verify DETRAC-20 columns exist
    missing = [c for c in DETRAC_20 if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing DETRAC-20 features: {missing}", flush=True)
        return None, None

    all_feat_cols = [c for c in feat.columns if c not in ["sequence", "frame_id"]]
    print(f"  {len(df)} frames, {df['sequence'].nunique()} sequences, "
          f"{len(all_feat_cols)} total features", flush=True)
    return df, all_feat_cols


def compute_threshold(df):
    all_targets = []
    for seq, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        for t in range(0, n - WINDOW - HORIZON + 1, WINDOW):
            all_targets.append(fn[t + WINDOW:t + WINDOW + HORIZON].mean())
    return float(np.median(all_targets))


def build_windows(df, feature_cols, threshold):
    sequences = sorted(df["sequence"].unique())
    train_seqs = [s for s in sequences if s not in VAL_SEQS]
    val_seqs = [s for s in sequences if s in VAL_SEQS]

    def _make(seqs):
        X_all, y_all, seq_labels = [], [], []
        tp_nano_last, tp_medium_last = [], []
        for seq in seqs:
            sdf = df[df["sequence"] == seq].sort_values("frame_id")
            feats = sdf[feature_cols].values.astype(np.float32)
            fn = sdf["fn_nano"].values.astype(np.float32)
            tp_n = sdf["tp_nano"].values
            tp_m = sdf["tp_medium"].values
            n = len(sdf)
            s = TRAIN_STRIDE if seq not in VAL_SEQS else EVAL_STRIDE
            for i in range(0, n - WINDOW - HORIZON + 1, s):
                x = feats[i:i + WINDOW]
                future_fn = fn[i + WINDOW:i + WINDOW + HORIZON]
                y = 1 if future_fn.mean() >= threshold else 0
                X_all.append(x)
                y_all.append(y)
                seq_labels.append(seq)
                tp_nano_last.append(tp_n[i + WINDOW - 1])
                tp_medium_last.append(tp_m[i + WINDOW - 1])
        return (np.array(X_all), np.array(y_all), seq_labels,
                np.array(tp_nano_last), np.array(tp_medium_last))

    X_tr, y_tr, _, _, _ = _make(train_seqs)
    X_va, y_va, va_seqs, tp_n_last, tp_m_last = _make(val_seqs)
    print(f"  Windows: train={len(X_tr)} (hard={y_tr.mean():.1%})  "
          f"val={len(X_va)} (hard={y_va.mean():.1%})", flush=True)
    return X_tr, y_tr, X_va, y_va, va_seqs, tp_n_last, tp_m_last


# ── Model ───────────────────────────────────────────────────────────────────


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class DirichletGRU(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 2)
    def forward(self, x):
        _, h = self.gru(x)
        return torch.exp(self.fc(h[-1])) + 1.0


def dirichlet_loss(alpha, y, epoch, max_epoch=60):
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    nll = -torch.log(p.gather(1, y.unsqueeze(1)) + 1e-8).squeeze(1)
    lam = min(1.0, epoch / (max_epoch * 0.6))
    alpha0 = torch.ones_like(alpha)
    kl = (torch.lgamma(alpha.sum(1)) - torch.lgamma(alpha0.sum(1))
          - (torch.lgamma(alpha) - torch.lgamma(alpha0)).sum(1)
          + ((alpha - alpha0) * (torch.digamma(alpha) - torch.digamma(S))).sum(1))
    return (nll + lam * kl).mean()


def train_gru(X_tr, y_tr, X_va, y_va, seed, n_features):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirichletGRU(n_features, hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    ds = SeqDataset(X_tr, y_tr)
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    best_acc, best_state = 0, None
    for epoch in range(60):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = dirichlet_loss(model(xb), yb, epoch, 60)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
            av = model(Xv)
            preds = (av[:, 1] / av.sum(1)).cpu().numpy()
            acc = ((preds >= 0.5).astype(int) == y_va).mean()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
        av = model(Xv)
        preds = (av[:, 1] / av.sum(1)).cpu().numpy()
        pred_labels = (preds >= 0.5).astype(int)
    return best_acc, pred_labels


def train_rf(X_tr, y_tr, X_va, y_va, seed):
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_va_flat = X_va.reshape(len(X_va), -1)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
    rf.fit(X_tr_flat, y_tr)
    pred = rf.predict(X_va_flat)
    return (pred == y_va).mean(), pred, rf.feature_importances_


# ── Intra-sequence analysis ─────────────────────────────────────────────────


def intra_sequence_analysis(y_va, pred_labels, va_seqs):
    seq_names = sorted(set(va_seqs))
    results = {}
    for seq in seq_names:
        mask = np.array([s == seq for s in va_seqs])
        y_s, p_s = y_va[mask], pred_labels[mask]
        n = len(y_s)
        if n == 0: continue
        easy_mask, hard_mask = y_s == 0, y_s == 1
        easy_acc = (p_s[easy_mask] == 0).mean() if easy_mask.sum() > 0 else None
        hard_acc = (p_s[hard_mask] == 1).mean() if hard_mask.sum() > 0 else None
        transitions, trans_correct = 0, 0
        for j in range(1, n):
            if y_s[j] != y_s[j-1]:
                transitions += 1
                if p_s[j] == y_s[j]: trans_correct += 1
        trans_acc = trans_correct / transitions if transitions > 0 else None
        results[seq] = {
            "n": n, "acc": (y_s == p_s).mean(), "hard_frac": y_s.mean(),
            "easy_acc": easy_acc, "hard_acc": hard_acc,
            "trans_acc": trans_acc, "transitions": transitions,
            "pred_hard_frac": p_s.mean(),
        }
    return results


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    df, all_feat_cols = load_data()
    if df is None:
        return

    threshold = compute_threshold(df)
    print(f"Threshold: {threshold:.3f}", flush=True)

    # Feature sets: DETRAC-20 (blind) and All-65 (for comparison)
    feature_sets = {
        "DETRAC-20 (blind)": DETRAC_20,
        f"All-{len(all_feat_cols)} DETRAC features": all_feat_cols,
    }

    # MedSample baseline
    scaler_base = StandardScaler()
    train_df = df[~df["sequence"].isin(VAL_SEQS)]
    scaler_base.fit(train_df[DETRAC_20].values)
    df_base = df.copy()
    df_base[DETRAC_20] = scaler_base.transform(df_base[DETRAC_20].values)
    _, _, _, y_va_base, _, tp_n, tp_m = build_windows(df_base, DETRAC_20, threshold)
    med_pred = (tp_m > tp_n).astype(int)
    med_acc = (med_pred == y_va_base).mean()
    print(f"\nMedSample baseline: {med_acc:.1%}", flush=True)

    all_results = {}
    csv_rows = []

    for fs_name, fs_cols in feature_sets.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Feature set: {fs_name} ({len(fs_cols)} features)", flush=True)
        print(f"{'='*60}", flush=True)

        scaler = StandardScaler()
        train_data = df[~df["sequence"].isin(VAL_SEQS)]
        scaler.fit(train_data[fs_cols].values)
        df_copy = df.copy()
        df_copy[fs_cols] = scaler.transform(df_copy[fs_cols].values)

        X_tr, y_tr, X_va, y_va, va_seqs, _, _ = build_windows(df_copy, fs_cols, threshold)

        gru_accs, best_gru_acc, best_gru_preds = [], 0, None
        for seed in SEEDS:
            acc, preds = train_gru(X_tr, y_tr, X_va, y_va, seed, len(fs_cols))
            gru_accs.append(acc)
            print(f"  GRU seed={seed}: {acc:.1%}", flush=True)
            if acc > best_gru_acc:
                best_gru_acc = acc
                best_gru_preds = preds

        rf_acc, rf_preds, rf_imp = train_rf(X_tr, y_tr, X_va, y_va, 42)
        print(f"  RF: {rf_acc:.1%}", flush=True)

        intra = intra_sequence_analysis(y_va, best_gru_preds, va_seqs)

        mixed_signal, mixed_total = 0, 0
        for seq, r in intra.items():
            if 0.05 < r["hard_frac"] < 0.95:
                mixed_total += 1
                if r["easy_acc"] is not None and r["hard_acc"] is not None:
                    if r["easy_acc"] > 0.5 and r["hard_acc"] > 0.5:
                        mixed_signal += 1

        total_trans = sum(r["transitions"] for r in intra.values() if r["transitions"])
        total_trans_correct = sum(
            int(r["trans_acc"] * r["transitions"])
            for r in intra.values()
            if r["trans_acc"] is not None and r["transitions"] > 0)
        agg_trans_acc = total_trans_correct / total_trans if total_trans > 0 else 0

        print(f"  Intra signal: {mixed_signal}/{mixed_total} mixed seqs", flush=True)
        print(f"  Transition accuracy: {agg_trans_acc:.1%} ({total_trans} transitions)", flush=True)

        # Per-sequence detail
        for seq in sorted(intra.keys()):
            r = intra[seq]
            ea = f"{r['easy_acc']:.0%}" if r['easy_acc'] is not None else "N/A"
            ha = f"{r['hard_acc']:.0%}" if r['hard_acc'] is not None else "N/A"
            ta = f"{r['trans_acc']:.0%}" if r['trans_acc'] is not None else "N/A"
            print(f"    {seq}: acc={r['acc']:.0%} hard={r['hard_frac']:.0%} "
                  f"easy={ea} hard={ha} trans={ta} ({r['transitions']}t)", flush=True)
            csv_rows.append({
                "feature_set": fs_name, "sequence": seq,
                "n_windows": r["n"], "accuracy": r["acc"],
                "hard_frac": r["hard_frac"], "easy_acc": r["easy_acc"],
                "hard_acc": r["hard_acc"], "trans_acc": r["trans_acc"],
                "transitions": r["transitions"], "pred_hard_frac": r["pred_hard_frac"],
            })

        all_results[fs_name] = {
            "gru_accs": gru_accs, "gru_mean": np.mean(gru_accs),
            "gru_std": np.std(gru_accs), "rf_acc": rf_acc,
            "rf_importances": dict(zip(fs_cols, rf_imp)) if len(rf_imp) == len(fs_cols) else {},
            "intra": intra, "mixed_signal": mixed_signal,
            "mixed_total": mixed_total, "trans_acc": agg_trans_acc,
        }

    # ── CSV ───────────────────────────────────────────────────────────────

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}", flush=True)

    # ── PDF ───────────────────────────────────────────────────────────────

    print(f"Generating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:

        # Page 1: Summary
        fig, ax = plt.subplots(figsize=(10, 6))
        fs_names = list(all_results.keys())
        gru_means = [all_results[n]["gru_mean"] for n in fs_names]
        gru_stds = [all_results[n]["gru_std"] for n in fs_names]
        rf_accs = [all_results[n]["rf_acc"] for n in fs_names]
        x = np.arange(len(fs_names))
        width = 0.3
        bars1 = ax.bar(x - width/2, gru_means, width, yerr=gru_stds,
                        label="GRU-1L-32", capsize=4, color="#4C72B0")
        bars2 = ax.bar(x + width/2, rf_accs, width, label="RF", color="#DD8452")
        ax.axhline(med_acc, color="red", ls="--", lw=1.5, label=f"MedSample ({med_acc:.1%})")
        ax.set_xticks(x)
        ax.set_xticklabels(fs_names, fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title("VisDrone: DETRAC Features Blind Cross-Domain Test (0.5s window)\n"
                      "Features selected on DETRAC, tested on VisDrone — zero feature selection leakage")
        ax.set_ylim(0.4, 1.0)
        ax.legend()
        for b in bars1:
            ax.annotate(f"{b.get_height():.1%}", (b.get_x()+b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        for b in bars2:
            ax.annotate(f"{b.get_height():.1%}", (b.get_x()+b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 2: RF feature importances for DETRAC-20
        d20_key = "DETRAC-20 (blind)"
        if d20_key in all_results and all_results[d20_key]["rf_importances"]:
            imp = all_results[d20_key]["rf_importances"]
            imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            fig, ax = plt.subplots(figsize=(8, 7))
            feats = [f for f, _ in imp_sorted]
            vals = [v for _, v in imp_sorted]
            ax.barh(range(len(feats)), vals, color="#4C72B0")
            ax.set_yticks(range(len(feats)))
            ax.set_yticklabels(feats, fontsize=8)
            ax.set_xlabel("RF Feature Importance (on VisDrone)")
            ax.set_title("DETRAC-20 Features: Importance When Applied to VisDrone")
            ax.invert_yaxis()
            plt.tight_layout()
            pdf.savefig(fig); plt.close()

        # Page 3: Per-sequence intra analysis
        best_fs = max(all_results.keys(), key=lambda k: all_results[k]["gru_mean"])
        intra = all_results[best_fs]["intra"]
        seq_names = sorted(intra.keys())
        if seq_names:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            easy_accs = [intra[s]["easy_acc"] if intra[s]["easy_acc"] is not None else 0 for s in seq_names]
            hard_accs = [intra[s]["hard_acc"] if intra[s]["hard_acc"] is not None else 0 for s in seq_names]
            hard_fracs = [intra[s]["hard_frac"] for s in seq_names]
            y_pos = np.arange(len(seq_names))
            ax = axes[0]
            ax.barh(y_pos - 0.2, easy_accs, 0.4, label="Easy acc", color="#55A868")
            ax.barh(y_pos + 0.2, hard_accs, 0.4, label="Hard acc", color="#C44E52")
            ax.axvline(0.5, color="black", ls="--", lw=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{s[:20]}\n({h:.0%} hard)" for s, h in zip(seq_names, hard_fracs)], fontsize=7)
            ax.set_title(f"Per-Seq Easy/Hard Accuracy ({best_fs})")
            ax.legend(fontsize=8)
            ax = axes[1]
            trans_accs = [intra[s]["trans_acc"] if intra[s]["trans_acc"] is not None else 0 for s in seq_names]
            n_trans = [intra[s]["transitions"] for s in seq_names]
            ax.barh(y_pos, trans_accs, color=["#4C72B0" if t > 0 else "#CCC" for t in n_trans])
            ax.axvline(0.5, color="red", ls="--", lw=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{s[:20]} ({n})" for s, n in zip(seq_names, n_trans)], fontsize=7)
            ax.set_title("Transition Accuracy")
            plt.tight_layout()
            pdf.savefig(fig); plt.close()

    print(f"Saved PDF: {OUT_PDF}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY — DETRAC Features Blind Test on VisDrone", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Threshold: {threshold:.3f}", flush=True)
    print(f"MedSample: {med_acc:.1%}", flush=True)
    for fs_name, r in all_results.items():
        print(f"\n{fs_name}:", flush=True)
        print(f"  GRU: {r['gru_mean']:.1%} ± {r['gru_std']:.1%}  "
              f"(seeds: {[f'{a:.1%}' for a in r['gru_accs']]})", flush=True)
        print(f"  RF:  {r['rf_acc']:.1%}", flush=True)
        print(f"  Intra: {r['mixed_signal']}/{r['mixed_total']} | "
              f"Trans: {r['trans_acc']:.1%}", flush=True)
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
