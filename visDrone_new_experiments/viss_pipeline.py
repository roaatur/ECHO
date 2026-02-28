#!/usr/bin/env python3
"""VisDrone full pipeline: feature selection + GRU/RF + intra-sequence analysis.

Steps:
  1. Merge eval results (nano TP/FN, medium TP) with 47 spatial features
  2. Spearman correlation + hierarchical clustering → selected features
  3. GRU-1L-32 Dirichlet (3 seeds) + RF on selected features
  4. Intra-sequence signal analysis (easy/hard/transition accuracy)
  5. PDF report

Uses official VisDrone VID val split (6 sequences).
VisDrone is drone footage → camera moves → features SHOULD vary within sequences.
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
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ECHO_DIR = SCRIPT_DIR.parent
EVAL_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "visdrone_classagnostic_results.csv"
FEAT_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "jays" / "jays_visdrone_features.csv"
VISDRONE_BASE = Path("/home/aljouhara/Desktop/git-echo/VisDrone-VID dataset")
OUT_PDF = SCRIPT_DIR / "viss_pipeline_results.pdf"
OUT_CSV = SCRIPT_DIR / "viss_pipeline_results.csv"

# Official VisDrone VID val sequences
VAL_SEQS = [
    "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
    "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v",
]

# All-NaN columns to drop
DROP_COLS = [
    "GLCM_contrast", "GLCM_homogeneity", "GLCM_energy", "GLCM_entropy",
    "GLCM_correlation", "LBP_entropy", "LBP_uniformity", "wavelet_detail_energy",
]

# VisDrone is ~30fps (varies per sequence, but ~30 is standard)
FPS = 30
WINDOW = 15  # 0.5s
HORIZON = 15
TRAIN_STRIDE = 8
EVAL_STRIDE = 15
SEEDS = [42, 123, 7]


# ── Data ────────────────────────────────────────────────────────────────────


def load_data():
    """Load and merge eval results + features for VisDrone."""
    print("Loading VisDrone data ...", flush=True)

    # Eval results
    ev = pd.read_csv(EVAL_CSV)
    nano = ev[ev["model"] == "yolo11n.pt"][
        ["sequence", "frame_id", "tp", "fp", "fn", "num_gt"]
    ].copy()
    nano.columns = ["sequence", "frame_id", "tp_nano", "fp_nano", "fn_nano", "num_gt"]
    med = ev[ev["model"] == "yolo11m.pt"][["sequence", "frame_id", "tp"]].copy()
    med.columns = ["sequence", "frame_id", "tp_medium"]
    eval_merged = nano.merge(med, on=["sequence", "frame_id"])

    # Features
    feat = pd.read_csv(FEAT_CSV)
    feat = feat.drop(columns=[c for c in DROP_COLS if c in feat.columns], errors="ignore")
    # Fill remaining NaN with 0 (foreground_ratio has 7% NaN)
    feat = feat.fillna(0.0)

    # Merge
    df = eval_merged.merge(feat, on=["sequence", "frame_id"], how="inner")
    meta_cols = ["sequence", "frame_id", "tp_nano", "fp_nano", "fn_nano",
                 "num_gt", "tp_medium"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  {len(df)} frames, {df['sequence'].nunique()} sequences, "
          f"{len(feature_cols)} features", flush=True)
    return df, feature_cols


def compute_threshold(df):
    """Compute median threshold over all sequences."""
    all_targets = []
    for seq, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        total_need = WINDOW + HORIZON
        for t in range(0, n - total_need + 1, WINDOW):
            all_targets.append(fn[t + WINDOW:t + WINDOW + HORIZON].mean())
    threshold = float(np.median(all_targets))
    return threshold


# ── Feature selection ───────────────────────────────────────────────────────


def spearman_correlation_analysis(df, feature_cols, threshold):
    """Shifted-label Spearman correlation for all features (TRAIN ONLY)."""
    print("\n=== Spearman correlation analysis (train sequences only) ===", flush=True)
    train_df = df[~df["sequence"].isin(VAL_SEQS)]
    all_feats, all_targets = [], []
    for seq, sdf in train_df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        total_need = WINDOW + HORIZON
        for t in range(0, n - total_need + 1, WINDOW):
            x = sdf[feature_cols].iloc[t:t + WINDOW].mean().values
            future_fn = fn[t + WINDOW:t + WINDOW + HORIZON].mean()
            all_feats.append(x)
            all_targets.append(future_fn)
    all_feats = np.array(all_feats)
    all_targets = np.array(all_targets)
    print(f"  {len(all_targets)} windows for correlation", flush=True)

    corrs = {}
    for j, col in enumerate(feature_cols):
        rho, pval = spearmanr(all_feats[:, j], all_targets)
        corrs[col] = (rho, pval)

    corr_sorted = sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
    print(f"\n  Top 20 features by |ρ|:", flush=True)
    for feat, (rho, pval) in corr_sorted[:20]:
        print(f"    {feat:35s}  ρ={rho:+.4f}  p={pval:.2e}", flush=True)

    return corrs, corr_sorted


def select_features(corrs, feature_cols, df, min_abs_rho=0.05, cluster_thresh=0.5):
    """Select features: |ρ| >= threshold, then cluster correlated features."""
    # Filter by minimum correlation
    candidates = [(f, rho) for f, (rho, p) in corrs.items() if abs(rho) >= min_abs_rho]
    if len(candidates) < 3:
        # Lower threshold if too few
        min_abs_rho = 0.03
        candidates = [(f, rho) for f, (rho, p) in corrs.items() if abs(rho) >= min_abs_rho]
    candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    cand_names = [f for f, _ in candidates]
    print(f"\n  {len(cand_names)} features with |ρ| >= {min_abs_rho}", flush=True)

    if len(cand_names) <= 3:
        return cand_names

    # Hierarchical clustering on feature correlation matrix (train only)
    train_df = df[~df["sequence"].isin(VAL_SEQS)]
    feat_data = train_df[cand_names].values
    corr_matrix = np.abs(np.corrcoef(feat_data.T))
    np.fill_diagonal(corr_matrix, 0)
    dist = 1 - corr_matrix
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 0)

    # Condensed distance matrix
    n = len(cand_names)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist[i, j])
    condensed = np.array(condensed)

    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=cluster_thresh, criterion="distance")

    # Pick best feature per cluster (highest |ρ|)
    rho_map = {f: abs(r) for f, r in candidates}
    selected = []
    for cl in sorted(set(clusters)):
        members = [cand_names[i] for i in range(len(cand_names)) if clusters[i] == cl]
        best = max(members, key=lambda f: rho_map[f])
        selected.append(best)

    selected.sort(key=lambda f: rho_map[f], reverse=True)
    print(f"  After clustering: {len(selected)} non-redundant features", flush=True)
    for f in selected:
        rho = corrs[f][0]
        print(f"    {f:35s}  ρ={rho:+.4f}", flush=True)

    return selected


# ── Windows ─────────────────────────────────────────────────────────────────


def build_windows(df, feature_cols, threshold):
    """Build shifted-label windows."""
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
        alpha = torch.exp(self.fc(h[-1])) + 1.0
        return alpha


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
            alpha = model(xb)
            loss = dirichlet_loss(alpha, yb, epoch, 60)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_va, dtype=torch.float32).to(device)
            av = model(Xv)
            preds = (av[:, 1] / av.sum(1)).cpu().numpy()
            pred_labels = (preds >= 0.5).astype(int)
            acc = (pred_labels == y_va).mean()
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
    acc = (pred == y_va).mean()
    return acc, pred, rf.feature_importances_


# ── Intra-sequence analysis ─────────────────────────────────────────────────


def intra_sequence_analysis(y_va, pred_labels, va_seqs):
    """Per-sequence breakdown + intra-sequence signal check."""
    seq_names = sorted(set(va_seqs))
    results = {}
    for seq in seq_names:
        mask = np.array([s == seq for s in va_seqs])
        y_s = y_va[mask]
        p_s = pred_labels[mask]
        n = len(y_s)
        if n == 0:
            continue
        acc = (y_s == p_s).mean()
        hard_frac = y_s.mean()

        easy_mask = y_s == 0
        hard_mask = y_s == 1
        easy_acc = (p_s[easy_mask] == 0).mean() if easy_mask.sum() > 0 else None
        hard_acc = (p_s[hard_mask] == 1).mean() if hard_mask.sum() > 0 else None

        transitions = 0
        trans_correct = 0
        for j in range(1, n):
            if y_s[j] != y_s[j - 1]:
                transitions += 1
                if p_s[j] == y_s[j]:
                    trans_correct += 1
        trans_acc = trans_correct / transitions if transitions > 0 else None

        results[seq] = {
            "n": n, "acc": acc, "hard_frac": hard_frac,
            "easy_acc": easy_acc, "hard_acc": hard_acc,
            "trans_acc": trans_acc, "transitions": transitions,
            "pred_hard_frac": p_s.mean(),
        }
    return results


# ── Intra-sequence feature variance check ───────────────────────────────────


def intra_seq_feature_variance(df, feature_cols):
    """Check how much features vary WITHIN sequences vs BETWEEN sequences.

    If within-sequence variance is high relative to between-sequence,
    features carry intra-sequence information (good for VisDrone).
    """
    print("\n=== Intra vs Inter sequence feature variance ===", flush=True)
    seq_means = df.groupby("sequence")[feature_cols].mean()
    global_mean = df[feature_cols].mean()

    # Between-sequence variance (variance of sequence means)
    between_var = seq_means.var()
    # Total variance
    total_var = df[feature_cols].var()
    # Within-sequence variance = total - between (approx)
    within_var = total_var - between_var

    ratio = (within_var / (total_var + 1e-10))
    print(f"  Mean within/total variance ratio: {ratio.mean():.3f}", flush=True)
    print(f"  (1.0 = all variance is within-sequence, 0.0 = all between-sequence)", flush=True)

    # Top features with most within-sequence variance (these are useful for intra-seq prediction)
    top_within = ratio.sort_values(ascending=False).head(15)
    print(f"\n  Top 15 features by within-sequence variance ratio:", flush=True)
    for feat, r in top_within.items():
        print(f"    {feat:35s}  within/total = {r:.3f}", flush=True)

    return ratio


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    df, feature_cols = load_data()

    # Compute threshold
    threshold = compute_threshold(df)
    print(f"Threshold: {threshold:.3f}", flush=True)

    # Intra vs inter sequence feature variance
    var_ratio = intra_seq_feature_variance(df, feature_cols)

    # Spearman correlation analysis
    corrs, corr_sorted = spearman_correlation_analysis(df, feature_cols, threshold)

    # Feature selection
    selected = select_features(corrs, feature_cols, df)

    # Feature sets to test
    feature_sets = {
        f"VisDrone-Selected ({len(selected)})": selected,
        f"All-{len(feature_cols)}": feature_cols,
    }

    # MedSample baseline
    scaler_base = StandardScaler()
    train_df = df[~df["sequence"].isin(VAL_SEQS)]
    scaler_base.fit(train_df[selected].values)
    df_base = df.copy()
    df_base[selected] = scaler_base.transform(df_base[selected].values)
    _, _, _, y_va_base, _, tp_n_last, tp_m_last = build_windows(df_base, selected, threshold)
    med_pred = (tp_m_last > tp_n_last).astype(int)
    med_acc = (med_pred == y_va_base).mean()
    print(f"\nMedSample baseline: {med_acc:.1%}", flush=True)

    all_results = {}
    csv_rows = []

    for fs_name, fs_cols in feature_sets.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Feature set: {fs_name}", flush=True)
        print(f"{'='*60}", flush=True)

        scaler = StandardScaler()
        train_data = df[~df["sequence"].isin(VAL_SEQS)]
        scaler.fit(train_data[fs_cols].values)
        df_copy = df.copy()
        df_copy[fs_cols] = scaler.transform(df_copy[fs_cols].values)

        X_tr, y_tr, X_va, y_va, va_seqs, _, _ = build_windows(df_copy, fs_cols, threshold)

        # GRU 3-seed
        gru_accs = []
        best_gru_acc = 0
        best_gru_preds = None
        for seed in SEEDS:
            acc, preds = train_gru(X_tr, y_tr, X_va, y_va, seed, len(fs_cols))
            gru_accs.append(acc)
            print(f"  GRU seed={seed}: {acc:.1%}", flush=True)
            if acc > best_gru_acc:
                best_gru_acc = acc
                best_gru_preds = preds

        # RF
        rf_acc, rf_preds, rf_imp = train_rf(X_tr, y_tr, X_va, y_va, 42)
        print(f"  RF: {rf_acc:.1%}", flush=True)

        # Intra-sequence analysis
        intra = intra_sequence_analysis(y_va, best_gru_preds, va_seqs)

        mixed_with_signal = 0
        mixed_total = 0
        for seq, r in intra.items():
            if r["hard_frac"] > 0.05 and r["hard_frac"] < 0.95:
                mixed_total += 1
                if r["easy_acc"] is not None and r["hard_acc"] is not None:
                    if r["easy_acc"] > 0.5 and r["hard_acc"] > 0.5:
                        mixed_with_signal += 1

        print(f"  Intra-sequence signal: {mixed_with_signal}/{mixed_total} mixed sequences", flush=True)

        total_trans = sum(r["transitions"] for r in intra.values() if r["transitions"])
        total_trans_correct = sum(
            int(r["trans_acc"] * r["transitions"])
            for r in intra.values()
            if r["trans_acc"] is not None and r["transitions"] > 0
        )
        agg_trans_acc = total_trans_correct / total_trans if total_trans > 0 else 0
        print(f"  Transition accuracy: {agg_trans_acc:.1%} ({total_trans} transitions)", flush=True)

        # Per-sequence detail
        print(f"\n  Per-sequence breakdown:", flush=True)
        for seq in sorted(intra.keys()):
            r = intra[seq]
            ea = f"{r['easy_acc']:.0%}" if r['easy_acc'] is not None else "N/A"
            ha = f"{r['hard_acc']:.0%}" if r['hard_acc'] is not None else "N/A"
            ta = f"{r['trans_acc']:.0%}" if r['trans_acc'] is not None else "N/A"
            print(f"    {seq}: acc={r['acc']:.0%} hard={r['hard_frac']:.0%} "
                  f"easy_acc={ea} hard_acc={ha} trans_acc={ta} "
                  f"({r['transitions']} trans, pred_hard={r['pred_hard_frac']:.0%})", flush=True)
            csv_rows.append({
                "feature_set": fs_name, "sequence": seq,
                "n_windows": r["n"], "accuracy": r["acc"],
                "hard_frac": r["hard_frac"], "easy_acc": r["easy_acc"],
                "hard_acc": r["hard_acc"], "trans_acc": r["trans_acc"],
                "transitions": r["transitions"], "pred_hard_frac": r["pred_hard_frac"],
            })

        all_results[fs_name] = {
            "gru_accs": gru_accs,
            "gru_mean": np.mean(gru_accs),
            "gru_std": np.std(gru_accs),
            "rf_acc": rf_acc,
            "rf_importances": dict(zip(fs_cols, rf_imp)) if len(rf_imp) == len(fs_cols) else {},
            "intra": intra,
            "mixed_signal": mixed_with_signal,
            "mixed_total": mixed_total,
            "trans_acc": agg_trans_acc,
        }

    # ── CSV ───────────────────────────────────────────────────────────────

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}", flush=True)

    # ── PDF Report ───────────────────────────────────────────────────────

    print(f"\nGenerating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:

        # Page 1: Summary bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        fs_names = list(all_results.keys())
        gru_means = [all_results[n]["gru_mean"] for n in fs_names]
        gru_stds = [all_results[n]["gru_std"] for n in fs_names]
        rf_accs = [all_results[n]["rf_acc"] for n in fs_names]
        x = np.arange(len(fs_names))
        width = 0.3
        bars1 = ax.bar(x - width/2, gru_means, width, yerr=gru_stds, label="GRU-1L-32", capsize=4, color="#4C72B0")
        bars2 = ax.bar(x + width/2, rf_accs, width, label="RF", color="#DD8452")
        ax.axhline(med_acc, color="red", ls="--", lw=1.5, label=f"MedSample ({med_acc:.1%})")
        ax.set_xticks(x)
        ax.set_xticklabels(fs_names, fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title("VisDrone: Difficulty Prediction (0.5s window)")
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

        # Page 2: Feature correlations (top 25)
        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(25, len(corr_sorted))
        feats_plot = [f for f, _ in corr_sorted[:top_n]]
        rhos_plot = [v[0] for _, v in corr_sorted[:top_n]]
        colors = ["#4C72B0" if r > 0 else "#C44E52" for r in rhos_plot]
        ax.barh(range(top_n), rhos_plot, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feats_plot, fontsize=7)
        ax.set_xlabel("Spearman ρ")
        ax.set_title("VisDrone: Feature Correlations with Future fn_nano (top 25)")
        ax.axvline(0, color="black", lw=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 3: Within/Between variance ratio
        fig, ax = plt.subplots(figsize=(10, 8))
        sorted_ratio = var_ratio.sort_values(ascending=False)
        top_n = min(25, len(sorted_ratio))
        feats_var = sorted_ratio.index[:top_n]
        vals_var = sorted_ratio.values[:top_n]
        ax.barh(range(top_n), vals_var, color="#55A868")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feats_var, fontsize=7)
        ax.set_xlabel("Within-Sequence / Total Variance Ratio")
        ax.set_title("VisDrone: Features with Most Within-Sequence Variation\n(Higher = more intra-sequence signal)")
        ax.axvline(0.5, color="red", ls="--", lw=0.8, label="50% ratio")
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 4: Per-sequence intra analysis (best feature set)
        best_fs = max(all_results.keys(), key=lambda k: all_results[k]["gru_mean"])
        intra = all_results[best_fs]["intra"]
        seq_names = sorted(intra.keys())

        if seq_names:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            easy_accs = [intra[s]["easy_acc"] if intra[s]["easy_acc"] is not None else 0 for s in seq_names]
            hard_accs = [intra[s]["hard_acc"] if intra[s]["hard_acc"] is not None else 0 for s in seq_names]
            hard_fracs = [intra[s]["hard_frac"] for s in seq_names]

            ax = axes[0]
            y_pos = np.arange(len(seq_names))
            ax.barh(y_pos - 0.2, easy_accs, 0.4, label="Easy acc", color="#55A868")
            ax.barh(y_pos + 0.2, hard_accs, 0.4, label="Hard acc", color="#C44E52")
            ax.axvline(0.5, color="black", ls="--", lw=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{s[:20]}\n({h:.0%} hard)" for s, h in zip(seq_names, hard_fracs)], fontsize=7)
            ax.set_xlabel("Accuracy")
            ax.set_title(f"Per-Sequence Easy/Hard Accuracy ({best_fs})")
            ax.legend(fontsize=8)

            ax = axes[1]
            trans_accs = [intra[s]["trans_acc"] if intra[s]["trans_acc"] is not None else 0 for s in seq_names]
            n_trans = [intra[s]["transitions"] for s in seq_names]
            colors = ["#4C72B0" if t > 0 else "#CCCCCC" for t in n_trans]
            ax.barh(y_pos, trans_accs, color=colors)
            ax.axvline(0.5, color="red", ls="--", lw=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{s[:20]} ({n})" for s, n in zip(seq_names, n_trans)], fontsize=7)
            ax.set_xlabel("Transition Accuracy")
            ax.set_title("Transition Accuracy (n transitions)")

            plt.tight_layout()
            pdf.savefig(fig); plt.close()

    print(f"\nSaved PDF: {OUT_PDF}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY — VisDrone Experiment", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Threshold: {threshold:.3f}", flush=True)
    print(f"MedSample baseline: {med_acc:.1%}", flush=True)
    for fs_name, r in all_results.items():
        print(f"\n{fs_name}:", flush=True)
        print(f"  GRU: {r['gru_mean']:.1%} ± {r['gru_std']:.1%}  "
              f"(seeds: {[f'{a:.1%}' for a in r['gru_accs']]})", flush=True)
        print(f"  RF:  {r['rf_acc']:.1%}", flush=True)
        print(f"  Intra signal: {r['mixed_signal']}/{r['mixed_total']} mixed seqs", flush=True)
        print(f"  Transition acc: {r['trans_acc']:.1%}", flush=True)
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
