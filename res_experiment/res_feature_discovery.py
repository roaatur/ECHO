"""
DETRAC Feature Discovery: Extract 53 image features on a 10-sequence subset,
correlate each with recall_gap (medium_640 − nano_320) and num_gt.
Reuses extraction functions from jays_uavdt_extract_features.py (cheap groups),
js4_density_proxy.py (3 density features), adds 5 new candidates.
Drops 11 expensive features: optical flow (3 + flow_coherence), GLCM (5), LBP (2).
"""

import sys
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

# ── Import existing extractors (cheap groups only) ────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # echo/

sys.path.insert(0, str(ROOT / "yolo_runs_on_uavdt"))
from jays_uavdt_extract_features import (
    extract_illumination, extract_contrast, extract_haze, extract_sharpness,
    extract_entropy, extract_color, extract_frequency,
    extract_spatial,
)

sys.path.insert(0, str(ROOT / "yolo_runs_on_uavdt" / "jays"))
from js4_density_proxy import extract_density_features

# ── Paths ─────────────────────────────────────────────────────────
DETRAC_IMAGES = Path("/home/aljouhara/Desktop/Detrac Dataset/DETRAC-Images")
RESULTS_CSV = SCRIPT_DIR / "detrac_results.csv"
FEATURES_CSV = SCRIPT_DIR / "res_discovery_features.csv"
CORR_CSV = SCRIPT_DIR / "res_feature_correlations.csv"

# ── 10-sequence subset (3 sunny, 3 cloudy, 2 night, 2 rainy) ─────
SUBSET_SEQUENCES = [
    "MVI_39361", "MVI_41073", "MVI_39211",       # sunny
    "MVI_40172", "MVI_40191", "MVI_40712",       # cloudy
    "MVI_40992", "MVI_40732",                     # night
    "MVI_63521", "MVI_40892",                     # rainy
]

# ── Feature names: 45 kept + 3 density + 5 new = 53 ──────────────
# Dropped 11 expensive features:
#   optical_flow_mag_mean, optical_flow_mag_std, optical_flow_angle_var,
#   flow_coherence (all need optical flow ~30ms)
#   GLCM_contrast, GLCM_homogeneity, GLCM_energy, GLCM_entropy,
#   GLCM_correlation (~20ms)
#   LBP_entropy, LBP_uniformity (~10ms)

KEPT_FEATURE_NAMES = [
    # A. Illumination (8)
    "mean_brightness", "brightness_std", "brightness_skewness",
    "brightness_kurtosis", "overexposed_ratio", "underexposed_ratio",
    "dynamic_range", "backlight_ratio",
    # B. Contrast (4)
    "weber_contrast", "michelson_contrast", "rms_contrast", "local_contrast_mean",
    # C. Haze/Fog (4)
    "dark_channel_mean", "dark_channel_std", "transmission_estimate", "atmospheric_light",
    # D. Sharpness/Blur (4)
    "laplacian_variance", "tenengrad", "brenner_gradient", "high_freq_energy_ratio",
    # E. Texture — slim (3, no GLCM/LBP)
    "edge_density", "gradient_magnitude_mean", "gradient_magnitude_std",
    # F. Entropy (4)
    "spatial_entropy", "color_entropy", "local_entropy_mean", "local_entropy_std",
    # G. Color (6)
    "colorfulness", "mean_saturation", "saturation_std", "hue_std",
    "dominant_color_ratio", "color_channel_corr",
    # H. Frequency Domain (3)
    "spectral_centroid", "spectral_rolloff", "wavelet_detail_energy",
    # I. Spatial Structure (3)
    "FAST_keypoint_count", "connected_components_count", "foreground_ratio",
    # J. Temporal — slim (6, no optical flow / flow_coherence)
    "frame_difference_mean", "SSIM_prev_frame", "histogram_intersection",
    "brightness_delta", "edge_density_delta", "scene_cut_score",
]

DENSITY_NAMES = ["contour_count", "gradient_orientation_var", "connected_component_count"]
NEW_NAMES = [
    "temporal_diff_std",     # std of frame difference
    "entropy_gradient",      # delta of local_entropy_std
    "saturation_gradient",   # delta of mean_saturation
    "num_vehicles_proxy",    # edge_density on lower half only
    "texture_complexity",    # std of local entropy across 8x8 grid
]
ALL_FEATURE_NAMES = KEPT_FEATURE_NAMES + DENSITY_NAMES + NEW_NAMES

assert len(ALL_FEATURE_NAMES) == 53, f"Expected 53 features, got {len(ALL_FEATURE_NAMES)}"


# ── Slim extraction functions (replacing expensive originals) ─────

def extract_texture_slim(gray):
    """Texture features without GLCM/LBP. Returns 3 features."""
    gray_f = gray.astype(np.float64)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    sx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx ** 2 + sy ** 2)
    return [edge_density, np.mean(grad_mag), np.std(grad_mag)]


def extract_temporal_slim(gray, gray_256, prev_gray, prev_gray_256,
                          prev_brightness, prev_edge_density):
    """Temporal features without optical flow. Returns 6 features."""
    if prev_gray is None:
        return [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    # Frame difference
    diff = cv2.absdiff(gray, prev_gray).astype(np.float64)
    frame_diff_mean = np.mean(diff)

    # SSIM (on 256x256)
    ssim_val = ssim(prev_gray_256, gray_256)

    # Histogram intersection
    hist_curr = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256]).flatten()
    hist_curr = hist_curr / (hist_curr.sum() + 1e-10)
    hist_prev = hist_prev / (hist_prev.sum() + 1e-10)
    hist_inter = np.sum(np.minimum(hist_curr, hist_prev))

    # Brightness delta
    curr_brightness = np.mean(gray)
    brightness_delta = curr_brightness - prev_brightness if prev_brightness is not None else 0.0

    # Edge density delta
    edges = cv2.Canny(gray, 100, 200)
    curr_edge_density = np.mean(edges > 0)
    edge_delta = curr_edge_density - prev_edge_density if prev_edge_density is not None else 0.0

    # Scene cut score
    scene_cut = 1.0 - hist_inter

    return [frame_diff_mean, ssim_val, hist_inter, brightness_delta, edge_delta, scene_cut]


# ── New feature extraction ────────────────────────────────────────

def extract_new_features(gray, bgr, gray_256, prev_gray, prev_local_entropy_std,
                         prev_mean_saturation):
    """Compute 5 new candidate features. Returns (features_list, state_dict)."""

    # 1. temporal_diff_std
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray).astype(np.float64)
        temporal_diff_std = float(np.std(diff))
    else:
        temporal_diff_std = 0.0

    # 2. entropy_gradient (delta of local_entropy_std)
    #    Recompute local_entropy_std here to track across frames
    h256, w256 = gray_256.shape
    patch_h, patch_w = 32, 32
    local_ents = []
    for i in range(0, h256 - patch_h + 1, patch_h):
        for j in range(0, w256 - patch_w + 1, patch_w):
            patch = gray_256[i:i + patch_h, j:j + patch_w]
            ph = cv2.calcHist([patch], [0], None, [256], [0, 256]).flatten()
            ph = ph / (ph.sum() + 1e-10)
            ph = ph[ph > 0]
            local_ents.append(-np.sum(ph * np.log2(ph)))
    curr_local_entropy_std = float(np.std(local_ents)) if local_ents else 0.0

    if prev_local_entropy_std is not None:
        entropy_gradient = curr_local_entropy_std - prev_local_entropy_std
    else:
        entropy_gradient = 0.0

    # 3. saturation_gradient (delta of mean_saturation)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    curr_mean_sat = float(np.mean(hsv[:, :, 1].astype(np.float64)))
    if prev_mean_saturation is not None:
        saturation_gradient = curr_mean_sat - prev_mean_saturation
    else:
        saturation_gradient = 0.0

    # 4. num_vehicles_proxy: edge_density on lower half of frame
    h, w = gray.shape
    lower_half = gray[h // 2:, :]
    edges_lower = cv2.Canny(lower_half, 100, 200)
    num_vehicles_proxy = float(np.mean(edges_lower > 0))

    # 5. texture_complexity: std of local entropy across 8x8 grid
    gh, gw = h // 8, w // 8
    grid_ents = []
    for i in range(8):
        for j in range(8):
            patch = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            ph = cv2.calcHist([patch], [0], None, [256], [0, 256]).flatten()
            ph = ph / (ph.sum() + 1e-10)
            ph = ph[ph > 0]
            grid_ents.append(-np.sum(ph * np.log2(ph)))
    texture_complexity = float(np.std(grid_ents)) if grid_ents else 0.0

    features = [temporal_diff_std, entropy_gradient, saturation_gradient,
                num_vehicles_proxy, texture_complexity]
    state = {
        "local_entropy_std": curr_local_entropy_std,
        "mean_saturation": curr_mean_sat,
    }
    return features, state


# ── Per-sequence processing ───────────────────────────────────────

def process_sequence(seq_name):
    """Extract all 53 features for every frame in a DETRAC sequence."""
    seq_dir = DETRAC_IMAGES / seq_name
    frame_files = sorted(seq_dir.glob("img*.jpg"))
    n_frames = len(frame_files)
    print(f"  {seq_name}: {n_frames} frames", flush=True)

    records = []

    # MOG2 background subtractor — warm up with first 30 frames
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
    init_limit = min(30, n_frames)
    for i in range(init_limit):
        init_frame = cv2.imread(str(frame_files[i]))
        if init_frame is not None:
            mog2.apply(init_frame)

    # Temporal state
    prev_gray = None
    prev_gray_256 = None
    prev_brightness = None
    prev_edge_density = None
    prev_local_entropy_std = None
    prev_mean_saturation = None

    # Timing accumulators (group → total ms)
    timings = {}

    for idx, fpath in enumerate(frame_files):
        frame_id = int(fpath.stem.replace("img", ""))
        bgr = cv2.imread(str(fpath))
        if bgr is None:
            continue

        # Precompute common representations
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        gray_256 = cv2.resize(gray, (256, 256))

        # ── 45 kept features (8 groups) ──
        t0 = time.time()
        feat_illum = extract_illumination(hsv, v_channel)          # 8
        t1 = time.time()
        feat_contrast = extract_contrast(gray)                      # 4
        t2 = time.time()
        feat_haze = extract_haze(bgr)                               # 4
        t3 = time.time()
        feat_sharp = extract_sharpness(gray)                        # 4
        t4 = time.time()
        feat_texture = extract_texture_slim(gray)                   # 3 (no GLCM/LBP)
        t5 = time.time()
        feat_entropy = extract_entropy(gray, bgr, gray_256)         # 4
        t6 = time.time()
        feat_color = extract_color(bgr, hsv)                        # 6
        t7 = time.time()
        feat_freq = extract_frequency(gray)                         # 3
        t8 = time.time()
        feat_spatial = extract_spatial(gray, bgr, mog2 if idx >= init_limit else None)  # 3
        t9 = time.time()
        feat_temporal = extract_temporal_slim(                       # 6 (no optical flow)
            gray, gray_256, prev_gray, prev_gray_256,
            prev_brightness, prev_edge_density,
        )
        t10 = time.time()

        # ── 3 density proxy features ──
        cc, gov, ccc = extract_density_features(gray)
        t11 = time.time()

        # ── 5 new features ──
        feat_new, new_state = extract_new_features(
            gray, bgr, gray_256, prev_gray,
            prev_local_entropy_std, prev_mean_saturation,
        )
        t12 = time.time()

        # Accumulate timings
        groups = [
            ("illumination", t1 - t0), ("contrast", t2 - t1), ("haze", t3 - t2),
            ("sharpness", t4 - t3), ("texture", t5 - t4), ("entropy", t6 - t5),
            ("color", t7 - t6), ("frequency", t8 - t7), ("spatial", t9 - t8),
            ("temporal", t10 - t9), ("density", t11 - t10), ("new", t12 - t11),
        ]
        for name, dt in groups:
            timings[name] = timings.get(name, 0.0) + dt

        if idx == 0:
            print("    Timing (ms, frame 0): " + "  ".join(
                f"{n}={dt*1000:.0f}" for n, dt in groups
            ), flush=True)

        all_features = (feat_illum + feat_contrast + feat_haze + feat_sharp +
                        feat_texture + feat_entropy + feat_color + feat_freq +
                        feat_spatial + feat_temporal +
                        [cc, gov, ccc] + feat_new)

        records.append([seq_name, frame_id] + all_features)

        # Update temporal state
        prev_gray = gray.copy()
        prev_gray_256 = gray_256.copy()
        prev_brightness = float(np.mean(gray))
        edges_prev = cv2.Canny(gray, 100, 200)
        prev_edge_density = float(np.mean(edges_prev > 0))
        prev_local_entropy_std = new_state["local_entropy_std"]
        prev_mean_saturation = new_state["mean_saturation"]

        if (idx + 1) % 200 == 0:
            print(f"    {seq_name}: {idx + 1}/{n_frames}", flush=True)

    # Average timings
    if records:
        n = len(records)
        avg_timings = {k: v / n * 1000 for k, v in timings.items()}
        print(f"    Avg ms/frame: " + "  ".join(
            f"{k}={v:.1f}" for k, v in avg_timings.items()
        ), flush=True)

    return records


# ── Part 1: Feature Extraction ────────────────────────────────────

def run_extraction():
    """Extract 53 features for the 10-sequence DETRAC subset."""
    print("=" * 70)
    print("Part 1: Feature Extraction (53 features × 10 sequences)")
    print("=" * 70)

    columns = ["sequence", "frame_id"] + ALL_FEATURE_NAMES
    all_records = []
    done_seqs = set()

    # Resume support
    if FEATURES_CSV.exists():
        existing = pd.read_csv(FEATURES_CSV)
        done_seqs = set(existing["sequence"].unique())
        all_records = existing.values.tolist()
        print(f"Resuming: {len(done_seqs)} sequences done ({len(existing)} frames)")

    t0 = time.time()

    for i, seq in enumerate(SUBSET_SEQUENCES):
        if seq in done_seqs:
            print(f"\n[{i + 1}/{len(SUBSET_SEQUENCES)}] Skipping {seq} (done)", flush=True)
            continue

        print(f"\n[{i + 1}/{len(SUBSET_SEQUENCES)}] Processing {seq}...", flush=True)
        seq_t0 = time.time()
        records = process_sequence(seq)
        all_records.extend(records)
        elapsed = time.time() - seq_t0

        ms_per_frame = (elapsed / len(records) * 1000) if records else 0
        remaining_seqs = [s for s in SUBSET_SEQUENCES[i + 1:] if s not in done_seqs]
        frames_remaining = sum(
            len(list((DETRAC_IMAGES / s).glob("img*.jpg"))) for s in remaining_seqs
        )
        eta_s = frames_remaining * elapsed / len(records) if records else 0
        print(f"    Done: {len(records)} frames in {elapsed:.1f}s "
              f"({ms_per_frame:.0f} ms/frame, ETA ~{eta_s / 60:.0f} min)", flush=True)

        # Checkpoint
        df = pd.DataFrame(all_records, columns=columns)
        df.to_csv(FEATURES_CSV, index=False)
        print(f"    Checkpoint saved ({len(df)} total frames)", flush=True)

    total = time.time() - t0
    print(f"\nExtraction complete: {len(all_records)} frames, {total:.0f}s")
    print(f"Saved → {FEATURES_CSV}")


# ── Part 2: Correlation Analysis ──────────────────────────────────

def run_analysis():
    """Correlate 53 features against recall_gap and num_gt."""
    print("\n" + "=" * 70)
    print("Part 2: Correlation Analysis")
    print("=" * 70)

    # Load features
    feat_df = pd.read_csv(FEATURES_CSV)
    print(f"Features: {len(feat_df)} frames, {len(ALL_FEATURE_NAMES)} features")

    # Load results and compute recall_gap
    results = pd.read_csv(RESULTS_CSV)
    piv = results.pivot_table(
        index=["sequence", "frame_id"], columns="model",
        values="recall",
    ).reset_index()
    piv.columns.name = None
    piv["recall_gap"] = piv["medium_640"] - piv["nano_320"]

    # Get weather and num_gt from nano rows
    meta = results[results["model"] == "nano_320"][
        ["sequence", "frame_id", "weather", "num_gt"]
    ].drop_duplicates()

    piv = piv.merge(meta, on=["sequence", "frame_id"], how="left")

    # Merge features with gap
    df = feat_df.merge(
        piv[["sequence", "frame_id", "recall_gap", "num_gt", "weather"]],
        on=["sequence", "frame_id"], how="inner",
    )
    print(f"Merged: {len(df)} frames")
    print(f"Weather: {df['weather'].value_counts().to_dict()}")

    # Compute Spearman correlations
    weather_cats = sorted(df["weather"].unique())
    corr_rows = []

    for feat in ALL_FEATURE_NAMES:
        x = df[feat].values.astype(np.float64)
        row = {"feature": feat}

        # vs recall_gap — overall
        mask = np.isfinite(x) & np.isfinite(df["recall_gap"].values)
        if mask.sum() > 10:
            rho, p = spearmanr(x[mask], df["recall_gap"].values[mask])
        else:
            rho, p = np.nan, np.nan
        row["rho_gap"] = rho
        row["p_gap"] = p

        # vs num_gt — overall
        mask = np.isfinite(x) & np.isfinite(df["num_gt"].values)
        if mask.sum() > 10:
            rho_n, p_n = spearmanr(x[mask], df["num_gt"].values[mask])
        else:
            rho_n, p_n = np.nan, np.nan
        row["rho_numgt"] = rho_n
        row["p_numgt"] = p_n

        # Per-weather vs recall_gap
        for w in weather_cats:
            wdf = df[df["weather"] == w]
            xw = wdf[feat].values.astype(np.float64)
            mask_w = np.isfinite(xw) & np.isfinite(wdf["recall_gap"].values)
            if mask_w.sum() > 10:
                rho_w, _ = spearmanr(xw[mask_w], wdf["recall_gap"].values[mask_w])
            else:
                rho_w = np.nan
            row[f"rho_gap_{w}"] = rho_w

        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows)
    corr_df["abs_rho_gap"] = corr_df["rho_gap"].abs()
    corr_df = corr_df.sort_values("abs_rho_gap", ascending=False)
    corr_df.to_csv(CORR_CSV, index=False)
    print(f"\nSaved → {CORR_CSV}")

    # ── Print table ───────────────────────────────────────────────
    print("\n" + "=" * 110)
    header = (f"{'#':>2}  {'Feature':<28} {'rho_gap':>8} {'p_gap':>10} "
              f"{'rho_gt':>8} {'p_gt':>10}")
    for w in weather_cats:
        header += f" {w[:6]:>8}"
    print(header)
    print("-" * 110)

    for rank, (_, r) in enumerate(corr_df.iterrows(), 1):
        line = (f"{rank:>2}  {r['feature']:<28} {r['rho_gap']:>+8.4f} {r['p_gap']:>10.2e} "
                f"{r['rho_numgt']:>+8.4f} {r['p_numgt']:>10.2e}")
        for w in weather_cats:
            val = r.get(f"rho_gap_{w}", np.nan)
            line += f" {val:>+8.4f}" if np.isfinite(val) else f" {'nan':>8}"
        print(line)

    print("=" * 110)

    # Summary
    top10 = corr_df.head(10)
    print(f"\nTop 10 features by |rho_gap|:")
    for _, r in top10.iterrows():
        print(f"  {r['feature']:<28} rho_gap={r['rho_gap']:+.4f}  rho_gt={r['rho_numgt']:+.4f}")

    n_sig = (corr_df["p_gap"] < 0.001).sum()
    print(f"\n{n_sig}/{len(corr_df)} features have p_gap < 0.001")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("extract", "all"):
        run_extraction()
    if mode in ("analyze", "all"):
        run_analysis()
