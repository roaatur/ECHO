"""Create dataset for predicting nano's own recall (not recall_gap).

Target: nano_recall (range 0-1) — directly from nano_320 results.
No medium model information needed.

Output: 2lstms_nano_recall_data.csv
  - 14 image features + 9 nano detection features + nano_recall, num_gt, weather
"""

from pathlib import Path
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # res_experiment/

FEAT_CSV = ROOT / "detrac_14features.csv"
DET_CSV  = ROOT / "models_dependent_detection_features.csv"
RES_CSV  = ROOT / "detrac_results.csv"

OUT_DIR = Path(__file__).resolve().parent               # 2lstms/
OUT_CSV = OUT_DIR / "2lstms_nano_recall_data.csv"

IMAGE_FEATURES = [
    "gradient_magnitude_std", "brightness_skewness", "overexposed_ratio",
    "contour_count", "wavelet_detail_energy", "high_freq_energy_ratio",
    "mean_brightness", "spectral_rolloff", "brightness_kurtosis",
    "dark_channel_std", "color_channel_corr", "FAST_keypoint_count",
    "ssim_prev", "temporal_diff_mean",
]

NANO_FEATURES = [
    "nano_num_detections", "nano_mean_confidence", "nano_min_confidence",
    "nano_confidence_std", "nano_mean_box_area", "nano_min_box_area",
    "nano_box_area_std", "nano_detection_density", "nano_edge_box_ratio",
]


def main():
    # ── Load sources ───────────────────────────────────────────────
    print("Loading CSVs...")
    feat = pd.read_csv(FEAT_CSV)
    print(f"  Image features:     {feat.shape[0]:>8} rows x {feat.shape[1]:>3} cols  "
          f"({feat['sequence'].nunique()} sequences)")

    det = pd.read_csv(DET_CSV)
    print(f"  Detection features: {det.shape[0]:>8} rows x {det.shape[1]:>3} cols  "
          f"({det['sequence'].nunique()} sequences)")

    res = pd.read_csv(RES_CSV)
    print(f"  Results:            {res.shape[0]:>8} rows x {res.shape[1]:>3} cols  "
          f"({res['sequence'].nunique()} sequences)")

    # ── Frame overlap check ───────────────────────────────────────
    feat_frames = set(zip(feat["sequence"], feat["frame_id"]))
    det_frames  = set(zip(det["sequence"], det["frame_id"]))
    overlap = feat_frames & det_frames

    print(f"\n  Frame overlap check:")
    print(f"    Image features:     {len(feat_frames)} unique frames")
    print(f"    Detection features: {len(det_frames)} unique frames")
    print(f"    Overlap:            {len(overlap)} frames")
    print(f"    Only in image:      {len(feat_frames - det_frames)} frames")
    print(f"    Only in detection:  {len(det_frames - feat_frames)} frames")

    # ── Extract nano targets (no medium needed) ───────────────────
    targets = res[res["model"] == "nano_320"][
        ["sequence", "frame_id", "recall", "num_gt", "weather"]
    ].copy()
    targets = targets.rename(columns={"recall": "nano_recall"})
    print(f"  Nano targets:         {len(targets)} frames")
    print(f"  nano_recall range:    [{targets['nano_recall'].min():.4f}, "
          f"{targets['nano_recall'].max():.4f}]")

    target_frames = set(zip(targets["sequence"], targets["frame_id"]))
    triple_overlap = overlap & target_frames
    print(f"  Three-way overlap:    {len(triple_overlap)} frames")

    # ── Merge ─────────────────────────────────────────────────────
    print("\nMerging...")
    df = feat.merge(det, on=["sequence", "frame_id"], how="inner")
    print(f"  After feat+det merge: {len(df)} rows")

    df = df.merge(targets, on=["sequence", "frame_id"], how="inner")
    print(f"  After +targets merge: {len(df)} rows")

    nan_total = df.isna().sum().sum()
    print(f"  Total NaN values:     {nan_total}")
    if nan_total > 0:
        for col, cnt in df.isna().sum()[df.isna().sum() > 0].items():
            print(f"    {col}: {cnt}")

    # ── Select columns (nano only) ───────────────────────────────
    keep_cols = ["sequence", "frame_id"] + IMAGE_FEATURES + NANO_FEATURES
    target_cols = ["nano_recall", "num_gt", "weather"]
    out_df = df[keep_cols + target_cols].copy()

    # ── Save ──────────────────────────────────────────────────────
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"  Saved -> {OUT_CSV}")
    print(f"  Shape: {out_df.shape[0]} rows x {out_df.shape[1]} cols")
    print(f"  Sequences: {out_df['sequence'].nunique()}")
    print(f"  Weather: {out_df['weather'].value_counts().to_dict()}")
    print(f"  num_gt range: [{out_df['num_gt'].min()}, {out_df['num_gt'].max()}]")
    print(f"  nano_recall range: [{out_df['nano_recall'].min():.4f}, "
          f"{out_df['nano_recall'].max():.4f}]")
    print(f"  nano_recall mean: {out_df['nano_recall'].mean():.4f}")

    feature_cols = [c for c in out_df.columns
                    if c not in ("sequence", "frame_id", "nano_recall",
                                 "num_gt", "weather")]
    print(f"  Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"    {col}: mean={out_df[col].mean():.4f}, std={out_df[col].std():.4f}")


if __name__ == "__main__":
    main()
