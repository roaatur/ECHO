"""
UAVDT Drop-Column Importance for 46 Static Features
Trains baseline RF on all features, then retrains with each feature
removed one at a time. Compares against permutation-importance decisions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
ANALYSIS_CSV = ROOT / "js_uavdt_analysis_dataset.csv"
SUMMARY_CSV = ROOT / "js_uavdt_correlation_summary.csv"
OUTPUT_CSV = ROOT / "js2_uavdt_drop_column_importance.csv"

STATIC_FEATURES = [
    # A. Illumination (1-8)
    "mean_brightness", "brightness_std", "brightness_skewness",
    "brightness_kurtosis", "overexposed_ratio", "underexposed_ratio",
    "dynamic_range", "backlight_ratio",
    # B. Contrast (9-12)
    "weber_contrast", "michelson_contrast", "rms_contrast", "local_contrast_mean",
    # C. Haze/Fog (13-16)
    "dark_channel_mean", "dark_channel_std", "transmission_estimate",
    "atmospheric_light",
    # D. Sharpness/Blur (17-20)
    "laplacian_variance", "tenengrad", "brenner_gradient",
    "high_freq_energy_ratio",
    # E. Texture/Complexity (21-30)
    "edge_density", "gradient_magnitude_mean", "gradient_magnitude_std",
    "GLCM_contrast", "GLCM_homogeneity", "GLCM_energy", "GLCM_entropy",
    "GLCM_correlation", "LBP_entropy", "LBP_uniformity",
    # F. Entropy (31-34)
    "spatial_entropy", "color_entropy", "local_entropy_mean", "local_entropy_std",
    # G. Color (35-40)
    "colorfulness", "mean_saturation", "saturation_std", "hue_std",
    "dominant_color_ratio", "color_channel_corr",
    # H. Frequency Domain (41-43)
    "spectral_centroid", "spectral_rolloff", "wavelet_detail_energy",
    # I. Spatial Structure (44-46)
    "FAST_keypoint_count", "connected_components_count", "foreground_ratio",
]

assert len(STATIC_FEATURES) == 46


if __name__ == "__main__":
    print("UAVDT Drop-Column Importance (46 static features)")
    print("=" * 60, flush=True)

    # --- Load & clean ---
    print("\nLoading data...", flush=True)
    df = pd.read_csv(ANALYSIS_CSV)
    cols = STATIC_FEATURES + ["recall_gap"]
    valid = df[cols].dropna()
    X = valid[STATIC_FEATURES].values
    y = valid["recall_gap"].values
    print(f"  Rows after dropna: {len(valid)}", flush=True)

    # --- Baseline RF ---
    print("\nTraining baseline RF on all 46 features...", flush=True)
    rf_params = dict(n_estimators=200, random_state=42, n_jobs=-1)
    rf_base = RandomForestRegressor(**rf_params)
    rf_base.fit(X, y)
    baseline_r2 = rf_base.score(X, y)
    print(f"  Baseline R² = {baseline_r2:.6f}", flush=True)

    # --- Drop-column loop ---
    print(f"\nDrop-column importance ({len(STATIC_FEATURES)} iterations):", flush=True)
    results = []
    feature_indices = {f: i for i, f in enumerate(STATIC_FEATURES)}

    for i, feat in enumerate(STATIC_FEATURES):
        idx = feature_indices[feat]
        X_reduced = np.delete(X, idx, axis=1)

        rf_reduced = RandomForestRegressor(**rf_params)
        rf_reduced.fit(X_reduced, y)
        reduced_r2 = rf_reduced.score(X_reduced, y)
        drop_imp = baseline_r2 - reduced_r2

        results.append({
            "feature": feat,
            "drop_column_importance": drop_imp,
            "baseline_r2": baseline_r2,
            "reduced_r2": reduced_r2,
        })
        print(f"  [{i+1:2d}/46] {feat:40s}  drop_imp={drop_imp:+.6f}  "
              f"R²={reduced_r2:.6f}", flush=True)

    # --- Save CSV ---
    out = pd.DataFrame(results).sort_values("drop_column_importance",
                                            ascending=False)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV}", flush=True)

    # --- Top 20 ---
    print("\nTop 20 features by drop-column importance:")
    print("-" * 70)
    for _, row in out.head(20).iterrows():
        print(f"  {row['feature']:40s}  {row['drop_column_importance']:+.6f}",
              flush=True)

    # --- Disagreement analysis ---
    print("\nDisagreement with permutation-importance selection:")
    print("-" * 70, flush=True)
    summary = pd.read_csv(SUMMARY_CSV)
    # Only static features (temporal have no RF importance)
    static_summary = summary[summary["feature"].isin(STATIC_FEATURES)].copy()
    static_summary = static_summary.set_index("feature")

    drop_dict = dict(zip(out["feature"], out["drop_column_importance"]))

    # Dropped by js1 but positive drop-column importance
    dropped_but_valuable = []
    for feat in STATIC_FEATURES:
        if feat not in static_summary.index:
            continue
        keep = static_summary.loc[feat, "keep"]
        di = drop_dict.get(feat, 0)
        if not keep and di > 0:
            dropped_but_valuable.append((feat, di,
                                         static_summary.loc[feat, "keep_reason"]))

    dropped_but_valuable.sort(key=lambda x: -x[1])
    print(f"\n  Dropped by permutation analysis but POSITIVE drop-column "
          f"importance ({len(dropped_but_valuable)}):")
    if dropped_but_valuable:
        for feat, di, reason in dropped_but_valuable:
            print(f"    {feat:40s}  drop_imp={di:+.6f}  reason: {reason}",
                  flush=True)
    else:
        print("    (none)")

    # Kept by js1 but zero/negative drop-column importance
    kept_but_useless = []
    for feat in STATIC_FEATURES:
        if feat not in static_summary.index:
            continue
        keep = static_summary.loc[feat, "keep"]
        di = drop_dict.get(feat, 0)
        if keep and di <= 0:
            kept_but_useless.append((feat, di,
                                     static_summary.loc[feat, "keep_reason"]))

    kept_but_useless.sort(key=lambda x: x[1])
    print(f"\n  Kept by permutation analysis but ZERO/NEGATIVE drop-column "
          f"importance ({len(kept_but_useless)}):")
    if kept_but_useless:
        for feat, di, reason in kept_but_useless:
            print(f"    {feat:40s}  drop_imp={di:+.6f}  reason: {reason}",
                  flush=True)
    else:
        print("    (none)")

    print("\nDone!", flush=True)
