#!/usr/bin/env python3
"""Combined feature selection on ALL 113 features (DETRAC-65 + Jay's-48).

Selection methods (computed on ALL VisDrone windows, not just train):
  1. Spearman ρ + hierarchical clustering
  2. Lasso (L1-logistic)
  3. Elastic Net (L1+L2 logistic)

GRU evaluation uses official train/val split.

Output:
  - viss_combined_selection_results.pdf
  - viss_combined_selection_results.csv
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
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ECHO_DIR = SCRIPT_DIR.parent
EVAL_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "visdrone_classagnostic_results.csv"
DETRAC_FEAT_CSV = SCRIPT_DIR / "viss_detrac65_features.csv"
JAYS_FEAT_CSV = ECHO_DIR / "yolo_runs_on_uavdt" / "jays" / "jays_visdrone_features.csv"
OUT_PDF = SCRIPT_DIR / "viss_combined_selection_results.pdf"
OUT_CSV = SCRIPT_DIR / "viss_combined_selection_results.csv"

VAL_SEQS = [
    "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
    "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v",
]

JAYS_DROP_COLS = [
    "GLCM_contrast", "GLCM_homogeneity", "GLCM_energy", "GLCM_entropy",
    "GLCM_correlation", "LBP_entropy", "LBP_uniformity", "wavelet_detail_energy",
]

WINDOW = 15
HORIZON = 15
TRAIN_STRIDE = 8
EVAL_STRIDE = 15
SEEDS = [42, 123, 7]


# ── Data ────────────────────────────────────────────────────────────────────


def load_data():
    """Load eval results + both feature sets, merge into one dataframe."""
    print("Loading data ...", flush=True)

    # Eval results
    ev = pd.read_csv(EVAL_CSV)
    nano = ev[ev["model"] == "yolo11n.pt"][
        ["sequence", "frame_id", "tp", "fp", "fn", "num_gt"]].copy()
    nano.columns = ["sequence", "frame_id", "tp_nano", "fp_nano", "fn_nano", "num_gt"]
    med = ev[ev["model"] == "yolo11m.pt"][["sequence", "frame_id", "tp"]].copy()
    med.columns = ["sequence", "frame_id", "tp_medium"]
    eval_merged = nano.merge(med, on=["sequence", "frame_id"])

    # DETRAC-65 features
    detrac = pd.read_csv(DETRAC_FEAT_CSV).fillna(0.0)
    detrac_feat_cols = [c for c in detrac.columns if c not in ("sequence", "frame_id")]
    detrac = detrac.rename(columns={c: f"D_{c}" for c in detrac_feat_cols})
    detrac_cols = [f"D_{c}" for c in detrac_feat_cols]

    # Jay's features
    jays = pd.read_csv(JAYS_FEAT_CSV)
    jays = jays.drop(columns=[c for c in JAYS_DROP_COLS if c in jays.columns], errors="ignore")
    jays = jays.fillna(0.0)
    jays_feat_cols = [c for c in jays.columns if c not in ("sequence", "frame_id")]
    jays = jays.rename(columns={c: f"J_{c}" for c in jays_feat_cols})
    jays_cols = [f"J_{c}" for c in jays_feat_cols]

    # Merge all
    df = eval_merged.merge(detrac, on=["sequence", "frame_id"], how="inner")
    df = df.merge(jays, on=["sequence", "frame_id"], how="inner")

    all_cols = detrac_cols + jays_cols
    print(f"  {len(df)} frames, {df['sequence'].nunique()} sequences", flush=True)
    print(f"  DETRAC: {len(detrac_cols)} features, Jay's: {len(jays_cols)} features, "
          f"Combined: {len(all_cols)}", flush=True)
    return df, detrac_cols, jays_cols, all_cols


def compute_threshold(df):
    all_targets = []
    for seq, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        for t in range(0, n - WINDOW - HORIZON + 1, WINDOW):
            all_targets.append(fn[t + WINDOW:t + WINDOW + HORIZON].mean())
    return float(np.median(all_targets))


def build_analysis_windows(df, feature_cols, threshold):
    """Build windows from ALL sequences for feature selection analysis."""
    feats, targets = [], []
    for seq, sdf in df.groupby("sequence"):
        sdf = sdf.sort_values("frame_id").reset_index(drop=True)
        fn = sdf["fn_nano"].values.astype(np.float32)
        n = len(sdf)
        for t in range(0, n - WINDOW - HORIZON + 1, WINDOW):
            x = sdf[feature_cols].iloc[t:t + WINDOW].mean().values
            future_fn = fn[t + WINDOW:t + WINDOW + HORIZON].mean()
            feats.append(x)
            targets.append(1 if future_fn >= threshold else 0)
    return np.array(feats), np.array(targets)


def build_windows(df, feature_cols, threshold):
    """Build train/val windows for GRU training."""
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
    X_va, y_va, va_seqs, tp_n, tp_m = _make(val_seqs)
    return X_tr, y_tr, X_va, y_va, va_seqs, tp_n, tp_m


# ── Feature Selection Methods ──────────────────────────────────────────────


def select_spearman(X, y, col_names, min_rho=0.05, cluster_thresh=0.5):
    """Spearman ρ + hierarchical clustering."""
    corrs = {}
    for j, col in enumerate(col_names):
        rho, pval = spearmanr(X[:, j], y)
        corrs[col] = (rho, pval)

    # Filter by min |ρ|
    candidates = [(f, rho) for f, (rho, p) in corrs.items() if abs(rho) >= min_rho]
    candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    cand_names = [f for f, _ in candidates]

    if len(cand_names) < 3:
        return cand_names, corrs

    # Hierarchical clustering on feature-feature correlations
    cand_data = X[:, [col_names.index(c) for c in cand_names]]
    corr_matrix = np.abs(np.corrcoef(cand_data.T))
    np.fill_diagonal(corr_matrix, 0)
    dist = 1.0 - corr_matrix
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    condensed = dist[np.triu_indices(len(cand_names), k=1)]
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=cluster_thresh, criterion="distance")

    selected = []
    for cid in sorted(set(clusters)):
        members = [(cand_names[i], abs(corrs[cand_names[i]][0]))
                    for i in range(len(cand_names)) if clusters[i] == cid]
        members.sort(key=lambda x: x[1], reverse=True)
        selected.append(members[0][0])

    selected.sort(key=lambda f: abs(corrs[f][0]), reverse=True)
    return selected, corrs


def select_lasso(X, y, col_names):
    """L1-penalized logistic regression."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=20, cv=5,
        max_iter=5000, random_state=42, scoring="accuracy",
    )
    clf.fit(X_sc, y)
    coefs = clf.coef_.ravel()
    surviving = [(col, coef) for col, coef in zip(col_names, coefs) if abs(coef) > 1e-6]
    surviving.sort(key=lambda x: abs(x[1]), reverse=True)
    selected = [f for f, _ in surviving]
    return selected, dict(zip(col_names, coefs)), clf.C_[0]


def select_elastic_net(X, y, col_names):
    """Elastic Net (L1+L2) logistic regression."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = LogisticRegressionCV(
        penalty="elasticnet", solver="saga", Cs=20, cv=5,
        l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        max_iter=5000, random_state=42, scoring="accuracy",
    )
    clf.fit(X_sc, y)
    coefs = clf.coef_.ravel()
    surviving = [(col, coef) for col, coef in zip(col_names, coefs) if abs(coef) > 1e-6]
    surviving.sort(key=lambda x: abs(x[1]), reverse=True)
    selected = [f for f, _ in surviving]
    return selected, dict(zip(col_names, coefs)), clf.C_[0], clf.l1_ratio_[0]


# ── GRU Model ──────────────────────────────────────────────────────────────


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


# ── Intra-sequence analysis ────────────────────────────────────────────────


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
    df, detrac_cols, jays_cols, all_cols = load_data()
    threshold = compute_threshold(df)
    print(f"Threshold: {threshold:.3f}", flush=True)

    # Build analysis windows (ALL sequences)
    print("\nBuilding analysis windows (all sequences) ...", flush=True)
    X_analysis, y_analysis = build_analysis_windows(df, all_cols, threshold)
    print(f"  {len(y_analysis)} windows  (hard={y_analysis.mean():.1%})")

    # ── Feature Selection ────────────────────────────────────────────────

    # 1. Spearman + clustering
    print("\n=== Method 1: Spearman + Clustering ===", flush=True)
    spearman_selected, spearman_corrs = select_spearman(X_analysis, y_analysis, all_cols)
    print(f"  Selected: {len(spearman_selected)} features")
    n_d = sum(1 for f in spearman_selected if f.startswith("D_"))
    n_j = sum(1 for f in spearman_selected if f.startswith("J_"))
    print(f"  Sources: {n_d} DETRAC + {n_j} Jay's")
    for f in spearman_selected[:20]:
        src = "D" if f.startswith("D_") else "J"
        rho = spearman_corrs[f][0]
        print(f"    {f.replace('D_','').replace('J_',''):40s}  ρ={rho:+.4f}  [{src}]")

    # 2. Lasso
    print("\n=== Method 2: Lasso (L1-Logistic) ===", flush=True)
    lasso_selected, lasso_coefs, lasso_C = select_lasso(X_analysis, y_analysis, all_cols)
    print(f"  Best C={lasso_C:.4f}, Selected: {len(lasso_selected)} features")
    n_d = sum(1 for f in lasso_selected if f.startswith("D_"))
    n_j = sum(1 for f in lasso_selected if f.startswith("J_"))
    print(f"  Sources: {n_d} DETRAC + {n_j} Jay's")
    for f in lasso_selected[:20]:
        src = "D" if f.startswith("D_") else "J"
        print(f"    {f.replace('D_','').replace('J_',''):40s}  coef={lasso_coefs[f]:+.4f}  [{src}]")

    # 3. Elastic Net
    print("\n=== Method 3: Elastic Net ===", flush=True)
    enet_selected, enet_coefs, enet_C, enet_l1 = select_elastic_net(
        X_analysis, y_analysis, all_cols)
    print(f"  Best C={enet_C:.4f}, l1_ratio={enet_l1:.2f}, "
          f"Selected: {len(enet_selected)} features")
    n_d = sum(1 for f in enet_selected if f.startswith("D_"))
    n_j = sum(1 for f in enet_selected if f.startswith("J_"))
    print(f"  Sources: {n_d} DETRAC + {n_j} Jay's")
    for f in enet_selected[:20]:
        src = "D" if f.startswith("D_") else "J"
        print(f"    {f.replace('D_','').replace('J_',''):40s}  coef={enet_coefs[f]:+.4f}  [{src}]")

    # ── GRU Evaluation ───────────────────────────────────────────────────

    feature_sets = {
        f"Spearman ({len(spearman_selected)})": spearman_selected,
        f"Lasso ({len(lasso_selected)})": lasso_selected,
        f"ElasticNet ({len(enet_selected)})": enet_selected,
        f"All-{len(all_cols)}": all_cols,
    }

    # MedSample baseline
    scaler_base = StandardScaler()
    train_df = df[~df["sequence"].isin(VAL_SEQS)]
    first_fs = list(feature_sets.values())[0]
    scaler_base.fit(train_df[first_fs].values)
    df_tmp = df.copy()
    df_tmp[first_fs] = scaler_base.transform(df_tmp[first_fs].values)
    _, _, _, y_va_base, _, tp_n, tp_m = build_windows(df_tmp, first_fs, threshold)
    med_pred = (tp_m > tp_n).astype(int)
    med_acc = (med_pred == y_va_base).mean()
    print(f"\nMedSample baseline: {med_acc:.1%}", flush=True)

    all_results = {}
    csv_rows = []

    for fs_name, fs_cols in feature_sets.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Feature set: {fs_name}", flush=True)
        print(f"{'='*60}", flush=True)

        scaler = StandardScaler()
        scaler.fit(train_df[fs_cols].values)
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
        print(f"  Transition accuracy: {agg_trans_acc:.1%} ({total_trans} transitions)",
              flush=True)

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
            "intra": intra, "mixed_signal": mixed_signal,
            "mixed_total": mixed_total, "trans_acc": agg_trans_acc,
        }

    # ── CSV ──────────────────────────────────────────────────────────────

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}", flush=True)

    # ── PDF ──────────────────────────────────────────────────────────────

    print("Generating PDF ...", flush=True)
    with PdfPages(str(OUT_PDF)) as pdf:

        # Page 1: GRU accuracy comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        fs_names = list(all_results.keys())
        gru_means = [all_results[n]["gru_mean"] for n in fs_names]
        gru_stds = [all_results[n]["gru_std"] for n in fs_names]
        rf_accs = [all_results[n]["rf_acc"] for n in fs_names]
        x = np.arange(len(fs_names))
        width = 0.3
        bars1 = ax.bar(x - width/2, gru_means, width, yerr=gru_stds,
                        label="GRU-1L-32", capsize=4, color="#4C72B0")
        bars2 = ax.bar(x + width/2, rf_accs, width, label="RF", color="#DD8452")
        ax.axhline(med_acc, color="red", ls="--", lw=1.5,
                   label=f"MedSample ({med_acc:.1%})")
        ax.set_xticks(x)
        ax.set_xticklabels(fs_names, fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title("VisDrone: Combined Feature Selection Methods\n"
                     "(DETRAC-65 + Jay's-48 pool)")
        ax.set_ylim(0.3, 1.0)
        ax.legend()
        for b in bars1:
            ax.annotate(f"{b.get_height():.1%}",
                        (b.get_x()+b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        for b in bars2:
            ax.annotate(f"{b.get_height():.1%}",
                        (b.get_x()+b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 2: Spearman top 30 combined ranking
        fig, ax = plt.subplots(figsize=(12, 10))
        ranked = sorted(spearman_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
        top30 = ranked[:30]
        names30 = [f[0].replace("D_", "").replace("J_", "") for f in top30]
        rhos30 = [v[0] for _, v in top30]
        src30 = ["DETRAC" if f[0].startswith("D_") else "Jay's" for f in top30]
        colors30 = ["#4C72B0" if s == "DETRAC" else "#55A868" for s in src30]
        ax.barh(range(len(top30)), rhos30, color=colors30)
        ax.set_yticks(range(len(top30)))
        labels30 = [f"{n}  [{s[0]}]" for n, s in zip(names30, src30)]
        ax.set_yticklabels(labels30, fontsize=7)
        ax.set_xlabel("Spearman ρ")
        ax.set_title("Combined Spearman Ranking (All VisDrone windows)\n"
                     "(Blue=DETRAC, Green=Jay's)")
        ax.axvline(0, color="black", lw=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        pdf.savefig(fig); plt.close()

        # Page 3: Lasso coefficients
        lasso_nz = [(c, lasso_coefs[c]) for c in lasso_selected if abs(lasso_coefs[c]) > 1e-6]
        lasso_nz.sort(key=lambda x: abs(x[1]), reverse=True)
        if lasso_nz:
            show_n = min(35, len(lasso_nz))
            fig, ax = plt.subplots(figsize=(12, max(6, show_n * 0.35)))
            names_l = [c.replace("D_","").replace("J_","") for c, _ in lasso_nz[:show_n]]
            coefs_l = [v for _, v in lasso_nz[:show_n]]
            src_l = ["DETRAC" if c.startswith("D_") else "Jay's" for c, _ in lasso_nz[:show_n]]
            colors_l = ["#4C72B0" if s == "DETRAC" else "#55A868" for s in src_l]
            labels_l = [f"{n}  [{s[0]}]" for n, s in zip(names_l, src_l)]
            ax.barh(range(show_n), coefs_l, color=colors_l)
            ax.set_yticks(range(show_n))
            ax.set_yticklabels(labels_l, fontsize=7)
            ax.set_xlabel("Lasso Coefficient")
            ax.set_title(f"Lasso L1-Logistic: {len(lasso_nz)} surviving features "
                         f"(C={lasso_C:.4f})")
            ax.axvline(0, color="black", lw=0.5)
            ax.invert_yaxis()
            plt.tight_layout()
            pdf.savefig(fig); plt.close()

        # Page 4: Elastic Net coefficients
        enet_nz = [(c, enet_coefs[c]) for c in enet_selected if abs(enet_coefs[c]) > 1e-6]
        enet_nz.sort(key=lambda x: abs(x[1]), reverse=True)
        if enet_nz:
            show_n = min(35, len(enet_nz))
            fig, ax = plt.subplots(figsize=(12, max(6, show_n * 0.35)))
            names_e = [c.replace("D_","").replace("J_","") for c, _ in enet_nz[:show_n]]
            coefs_e = [v for _, v in enet_nz[:show_n]]
            src_e = ["DETRAC" if c.startswith("D_") else "Jay's" for c, _ in enet_nz[:show_n]]
            colors_e = ["#4C72B0" if s == "DETRAC" else "#55A868" for s in src_e]
            labels_e = [f"{n}  [{s[0]}]" for n, s in zip(names_e, src_e)]
            ax.barh(range(show_n), coefs_e, color=colors_e)
            ax.set_yticks(range(show_n))
            ax.set_yticklabels(labels_e, fontsize=7)
            ax.set_xlabel("Elastic Net Coefficient")
            ax.set_title(f"Elastic Net: {len(enet_nz)} surviving features "
                         f"(C={enet_C:.4f}, l1_ratio={enet_l1:.2f})")
            ax.axvline(0, color="black", lw=0.5)
            ax.invert_yaxis()
            plt.tight_layout()
            pdf.savefig(fig); plt.close()

        # Page 5: Per-sequence intra analysis (best method)
        best_fs = max(all_results.keys(), key=lambda k: all_results[k]["gru_mean"])
        intra = all_results[best_fs]["intra"]
        seq_names = sorted(intra.keys())

        if seq_names:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            easy_accs = [intra[s]["easy_acc"] if intra[s]["easy_acc"] is not None else 0
                         for s in seq_names]
            hard_accs = [intra[s]["hard_acc"] if intra[s]["hard_acc"] is not None else 0
                         for s in seq_names]
            hard_fracs = [intra[s]["hard_frac"] for s in seq_names]

            ax = axes[0]
            y_pos = np.arange(len(seq_names))
            ax.barh(y_pos - 0.2, easy_accs, 0.4, label="Easy acc", color="#55A868")
            ax.barh(y_pos + 0.2, hard_accs, 0.4, label="Hard acc", color="#C44E52")
            ax.axvline(0.5, color="black", ls="--", lw=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{s[:20]}\n({h:.0%} hard)"
                                for s, h in zip(seq_names, hard_fracs)], fontsize=7)
            ax.set_xlabel("Accuracy")
            ax.set_title(f"Per-Sequence Easy/Hard Accuracy ({best_fs})")
            ax.legend(fontsize=8)

            ax = axes[1]
            trans_accs = [intra[s]["trans_acc"] if intra[s]["trans_acc"] is not None else 0
                          for s in seq_names]
            n_trans = [intra[s]["transitions"] for s in seq_names]
            colors_t = ["#4C72B0" if t > 0 else "#CCCCCC" for t in n_trans]
            ax.barh(y_pos, trans_accs, color=colors_t)
            ax.axvline(0.5, color="red", ls="--", lw=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{s[:20]} ({n})" for s, n in zip(seq_names, n_trans)],
                               fontsize=7)
            ax.set_xlabel("Transition Accuracy")
            ax.set_title("Transition Accuracy (n transitions)")

            plt.tight_layout()
            pdf.savefig(fig); plt.close()

    print(f"\nSaved PDF: {OUT_PDF}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("SUMMARY — Combined Feature Selection on VisDrone")
    print(f"{'='*60}")
    print(f"Threshold: {threshold:.3f}")
    print(f"MedSample: {med_acc:.1%}")
    for fs_name, r in all_results.items():
        print(f"\n{fs_name}:")
        print(f"  GRU: {r['gru_mean']:.1%} ± {r['gru_std']:.1%}  "
              f"(seeds: {[f'{a:.1%}' for a in r['gru_accs']]})")
        print(f"  RF:  {r['rf_acc']:.1%}")
        print(f"  Intra: {r['mixed_signal']}/{r['mixed_total']} | "
              f"Trans: {r['trans_acc']:.1%}")
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
