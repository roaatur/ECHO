"""Merge image features + single-model detection features + targets.

Creates two deployment-realistic datasets:
  - 2lstms_nano_data.csv:   14 image + 9 nano detection + targets
  - 2lstms_medium_data.csv: 14 image + 9 medium detection + targets

No cross-model information in either dataset.
"""

from pathlib import Path
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # res_experiment/

FEAT_CSV = ROOT / "detrac_14features.csv"
DET_CSV  = ROOT / "models_dependent_detection_features.csv"
RES_CSV  = ROOT / "detrac_results.csv"

OUT_DIR  = Path(__file__).resolve().parent              # 2lstms/
NANO_CSV   = OUT_DIR / "2lstms_nano_data.csv"
MEDIUM_CSV = OUT_DIR / "2lstms_medium_data.csv"

IMAGE_FEATURES = [
    "gradient_magnitude_std", "brightness_skewness", "overexposed_ratio",
    "contour_count", "wavelet_detail_energy", "high_freq_energy_ratio",
    "mean_brightness", "spectral_rolloff", "brightness_kurtosis",
    "dark_channel_std", "color_channel_corr", "FAST_keypoint_count",
    "ssim_prev", "temporal_diff_mean",
]

DET_SUFFIXES = [
    "num_detections", "mean_confidence", "min_confidence", "confidence_std",
    "mean_box_area", "min_box_area", "box_area_std",
    "detection_density", "edge_box_ratio",
]

NANO_FEATURES   = [f"nano_{s}" for s in DET_SUFFIXES]
MEDIUM_FEATURES = [f"medium_{s}" for s in DET_SUFFIXES]


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

    # ── Extract targets ───────────────────────────────────────────
    nano_res = res[res["model"] == "nano_320"][
        ["sequence", "frame_id", "recall", "num_gt", "weather"]
    ].copy()
    nano_res = nano_res.rename(columns={"recall": "recall_nano"})

    medium_res = res[res["model"] == "medium_640"][
        ["sequence", "frame_id", "recall"]
    ].copy()
    medium_res = medium_res.rename(columns={"recall": "recall_medium"})

    targets = nano_res.merge(medium_res, on=["sequence", "frame_id"])
    targets["recall_gap"] = targets["recall_medium"] - targets["recall_nano"]
    targets = targets[["sequence", "frame_id", "recall_gap", "num_gt", "weather"]]
    print(f"  Targets:              {len(targets)} frames")

    target_frames = set(zip(targets["sequence"], targets["frame_id"]))
    triple_overlap = overlap & target_frames
    print(f"  Three-way overlap:    {len(triple_overlap)} frames")

    # ── Full merge (all features) ─────────────────────────────────
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

    # ── Split into nano and medium datasets ───────────────────────
    keep_cols = ["sequence", "frame_id"] + IMAGE_FEATURES
    target_cols = ["recall_gap", "num_gt", "weather"]

    nano_df = df[keep_cols + NANO_FEATURES + target_cols].copy()
    medium_df = df[keep_cols + MEDIUM_FEATURES + target_cols].copy()

    # ── Save ──────────────────────────────────────────────────────
    nano_df.to_csv(NANO_CSV, index=False)
    medium_df.to_csv(MEDIUM_CSV, index=False)

    for label, out_df, path in [("NANO", nano_df, NANO_CSV),
                                 ("MEDIUM", medium_df, MEDIUM_CSV)]:
        print(f"\n{'='*60}")
        print(f"  {label} dataset saved -> {path}")
        print(f"  Shape: {out_df.shape[0]} rows x {out_df.shape[1]} cols")
        print(f"  Sequences: {out_df['sequence'].nunique()}")
        print(f"  Weather: {out_df['weather'].value_counts().to_dict()}")
        print(f"  num_gt range: [{out_df['num_gt'].min()}, {out_df['num_gt'].max()}]")
        print(f"  recall_gap range: [{out_df['recall_gap'].min():.4f}, "
              f"{out_df['recall_gap'].max():.4f}]")
        feature_cols = [c for c in out_df.columns
                        if c not in ("sequence", "frame_id", "recall_gap",
                                     "num_gt", "weather")]
        print(f"  Feature columns ({len(feature_cols)}):")
        for col in feature_cols:
            print(f"    {col}: mean={out_df[col].mean():.4f}, std={out_df[col].std():.4f}")


if __name__ == "__main__":
    main()
