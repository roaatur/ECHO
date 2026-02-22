"""
JS4: Density-Proxy Feature Extraction & Correlation Analysis
Extracts 3 new features (contour_count, gradient_orientation_var,
connected_component_count) for 16,592 UAVDT frames, then correlates
13 selected features against num_gt and recall_gap.
"""

import sys
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent          # echo/
UAVDT_FRAMES = ROOT / "UAVDT-NEW" / "UAV-benchmark-M"
UAVDT_DIR = ROOT / "yolo_runs_on_uavdt"

FEATURES_CSV = UAVDT_DIR / "jays_uavdt_frame_features.csv"
RESULTS_CSV = UAVDT_DIR / "uavdt_results.csv"
DENSITY_CSV = Path(__file__).resolve().parent / "js4_density_features.csv"
CORR_CSV = Path(__file__).resolve().parent / "js4_density_correlation.csv"

TEST_SEQUENCES = [
    "M0208", "M0209", "M0403", "M0601", "M0602", "M0606",
    "M0801", "M0802", "M1301", "M1302", "M1303", "M1401",
    "M0203", "M0205", "M1001", "M1007", "M1101", "M0701",
    "M1004", "M1009",
]

# 13 features for correlation analysis
TARGET_FEATURES = [
    "edge_density", "FAST_keypoint_count", "foreground_ratio",
    "spatial_entropy", "local_entropy_std", "high_freq_energy_ratio",
    "spectral_centroid", "LBP_entropy", "colorfulness", "saturation_std",
    "contour_count", "gradient_orientation_var", "connected_component_count",
]

# Features in current 23-feature set
IN_23 = {"local_entropy_std", "saturation_std", "high_freq_energy_ratio"}

DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# ── Part 1: Feature Extraction ──────────────────────────────────

def extract_density_features(gray):
    """Compute 3 density-proxy features from grayscale frame."""
    edges = cv2.Canny(gray, 50, 150)

    # 1. contour_count: external contours on Canny edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    # 2. gradient_orientation_var: circular variance of Sobel angles
    gray_f = gray.astype(np.float64)
    dx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(dy, dx)  # range [-pi, pi]
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    R_len = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    grad_orient_var = 1.0 - R_len

    # 3. connected_component_count: dilated Canny edges → CC
    dilated = cv2.dilate(edges, DILATE_KERNEL)
    num_cc, _ = cv2.connectedComponents(dilated)
    cc_count = num_cc - 1  # exclude background

    return contour_count, grad_orient_var, cc_count


def run_extraction():
    """Extract 3 density features for all frames, with resume support."""
    print("=" * 60)
    print("JS4 Part 1: Density Feature Extraction")
    print("=" * 60)

    records = []
    done_seqs = set()

    if DENSITY_CSV.exists():
        existing = pd.read_csv(DENSITY_CSV)
        done_seqs = set(existing["sequence"].unique())
        records = existing.values.tolist()
        print(f"Resuming: {len(done_seqs)} sequences done ({len(existing)} frames)")

    t0 = time.time()
    global_count = sum(1 for r in records)  # frames already done

    for si, seq in enumerate(TEST_SEQUENCES):
        if seq in done_seqs:
            print(f"[{si+1}/{len(TEST_SEQUENCES)}] Skipping {seq} (done)")
            continue

        seq_dir = UAVDT_FRAMES / seq
        frame_files = sorted(seq_dir.glob("img*.jpg"))
        print(f"[{si+1}/{len(TEST_SEQUENCES)}] {seq}: {len(frame_files)} frames",
              flush=True)

        seq_t0 = time.time()
        for idx, fpath in enumerate(frame_files):
            frame_id = int(fpath.stem.replace("img", ""))
            bgr = cv2.imread(str(fpath))
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            cc, gov, ccc = extract_density_features(gray)
            records.append([seq, frame_id, cc, gov, ccc])
            global_count += 1

            if global_count % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  progress: {global_count} frames, {elapsed:.0f}s elapsed",
                      flush=True)

        seq_elapsed = time.time() - seq_t0
        print(f"  {seq}: done in {seq_elapsed:.1f}s", flush=True)

        # checkpoint after each sequence
        df = pd.DataFrame(records,
                          columns=["sequence", "frame_id", "contour_count",
                                   "gradient_orientation_var",
                                   "connected_component_count"])
        df.to_csv(DENSITY_CSV, index=False)

    total = time.time() - t0
    print(f"\nExtraction done: {global_count} frames, {total:.0f}s")
    print(f"Saved → {DENSITY_CSV}")


# ── Part 2 & 3: Correlation Analysis ────────────────────────────

def run_analysis():
    """Merge features, compute Spearman correlations, output CSV + table."""
    print("\n" + "=" * 60)
    print("JS4 Part 2-3: Correlation Analysis")
    print("=" * 60)

    # Load dataframes
    feat56 = pd.read_csv(FEATURES_CSV)
    density = pd.read_csv(DENSITY_CSV)
    results = pd.read_csv(RESULTS_CSV)

    # Get nano rows for num_gt, weather, recall
    nano = results[results["model"] == "yolo11n.pt"][
        ["sequence", "frame_id", "num_gt", "weather", "recall"]
    ].rename(columns={"recall": "recall_nano"})

    # Get medium rows for recall_gap
    medium = results[results["model"] == "yolo11m.pt"][
        ["sequence", "frame_id", "recall"]
    ].rename(columns={"recall": "recall_medium"})

    # Merge all
    df = feat56.merge(density, on=["sequence", "frame_id"], how="inner")
    df = df.merge(nano, on=["sequence", "frame_id"], how="inner")
    df = df.merge(medium, on=["sequence", "frame_id"], how="inner")
    df["recall_gap"] = df["recall_medium"] - df["recall_nano"]

    print(f"Merged dataset: {len(df)} rows")
    print(f"Weather distribution: {df['weather'].value_counts().to_dict()}")

    # Compute correlations
    weather_groups = {"day": "daylight", "night": "night", "fog": "fog"}
    corr_rows = []

    for feat in TARGET_FEATURES:
        row = {"feature": feat}
        x = df[feat].values

        # vs num_gt
        mask_all = np.isfinite(x) & np.isfinite(df["num_gt"].values)
        rho, p = spearmanr(x[mask_all], df["num_gt"].values[mask_all])
        row["rho_numgt_all"] = rho
        row["p_numgt_all"] = p

        for label, wval in weather_groups.items():
            wdf = df[df["weather"] == wval]
            xw = wdf[feat].values
            mask = np.isfinite(xw) & np.isfinite(wdf["num_gt"].values)
            if mask.sum() > 10:
                rho_w, _ = spearmanr(xw[mask], wdf["num_gt"].values[mask])
            else:
                rho_w = np.nan
            row[f"rho_numgt_{label}"] = rho_w

        # vs recall_gap
        mask_all = np.isfinite(x) & np.isfinite(df["recall_gap"].values)
        rho_g, _ = spearmanr(x[mask_all], df["recall_gap"].values[mask_all])
        row["rho_gap_all"] = rho_g

        for label, wval in weather_groups.items():
            wdf = df[df["weather"] == wval]
            xw = wdf[feat].values
            mask = np.isfinite(xw) & np.isfinite(wdf["recall_gap"].values)
            if mask.sum() > 10:
                rho_gw, _ = spearmanr(xw[mask], wdf["recall_gap"].values[mask])
            else:
                rho_gw = np.nan
            row[f"rho_gap_{label}"] = rho_gw

        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(CORR_CSV, index=False)
    print(f"\nSaved → {CORR_CSV}")

    # ── Summary table ────────────────────────────────────────────
    corr_df["abs_rho_numgt"] = corr_df["rho_numgt_all"].abs()
    corr_df = corr_df.sort_values("abs_rho_numgt", ascending=False)

    def category(row):
        rn = abs(row["rho_numgt_all"])
        rg = abs(row["rho_gap_all"])
        if rn > 0.3 and rg > 0.2:
            return "dual-signal"
        if rn > 0.3 and rg < 0.15:
            return "pure density"
        if rn < 0.15 and rg > 0.2:
            return "gap-only"
        return ""

    print("\n" + "=" * 120)
    print(f"{'Feature':<30} {'Set':>3}  {'rho_gt':>7} {'day':>7} {'night':>7} "
          f"{'fog':>7}  {'rho_gap':>7}  {'Category':<14}")
    print("-" * 120)

    for _, r in corr_df.iterrows():
        feat = r["feature"]
        mark = "\u2713" if feat in IN_23 else "\u2605"
        cat = category(r)
        print(f"{feat:<30} {mark:>3}  {r['rho_numgt_all']:>+7.3f} "
              f"{r['rho_numgt_day']:>+7.3f} {r['rho_numgt_night']:>+7.3f} "
              f"{r['rho_numgt_fog']:>+7.3f}  {r['rho_gap_all']:>+7.3f}  "
              f"{cat:<14}")

    print("=" * 120)
    print(f"\n\u2713 = in current 23-feature set | \u2605 = not in set")
    print("dual-signal: |rho_gt|>0.3 AND |rho_gap|>0.2")
    print("pure density: |rho_gt|>0.3 AND |rho_gap|<0.15")
    print("gap-only: |rho_gt|<0.15 AND |rho_gap|>0.2")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("extract", "all"):
        run_extraction()
    if mode in ("analyze", "all"):
        run_analysis()
