"""
Extract 56 image features for the 30 UAVDT sequences that don't have features yet.
Reuses all feature functions from jays_uavdt_extract_features.py verbatim.
After extraction, merges with the original 20-sequence CSV into a combined 50-sequence file.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis
# skimage GLCM/LBP and pywt skipped (not in the 24 selected features)
from skimage.metrics import structural_similarity as ssim
import time

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # echo/
UAVDT_FRAMES = ROOT / "UAVDT-NEW" / "UAV-benchmark-M"
JAYS_DIR = Path(__file__).resolve().parent  # jays/
OUTPUT_CSV = JAYS_DIR / "jays_uavdt_remaining30_features.csv"
ORIGINAL_CSV = JAYS_DIR.parent / "jays_uavdt_frame_features.csv"
COMBINED_CSV = JAYS_DIR / "jays_uavdt_all50_features.csv"

# 20 sequences already extracted
DONE_SEQUENCES = {
    "M0208", "M0209", "M0403", "M0601", "M0602", "M0606",
    "M0801", "M0802", "M1301", "M1302", "M1303", "M1401",
    "M0203", "M0205", "M1001", "M1007", "M1101", "M0701",
    "M1004", "M1009",
}

FEATURE_NAMES = [
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
    # J. Motion/Temporal (47-56)
    "optical_flow_mag_mean", "optical_flow_mag_std", "optical_flow_angle_var",
    "frame_difference_mean", "SSIM_prev_frame", "histogram_intersection",
    "brightness_delta", "edge_density_delta", "flow_coherence", "scene_cut_score",
]

assert len(FEATURE_NAMES) == 56


# ── Feature Extraction Functions (copied verbatim) ───────────────

def extract_illumination(hsv, v_channel):
    """Features 1-8: Illumination & Exposure."""
    v_flat = v_channel.flatten().astype(np.float64)
    mean_b = np.mean(v_flat)
    std_b = np.std(v_flat)
    skew_b = skew(v_flat)
    kurt_b = kurtosis(v_flat)
    overexposed = np.mean(v_channel > 240)
    underexposed = np.mean(v_channel < 15)
    dyn_range = float(np.max(v_channel)) - float(np.min(v_channel))

    h, w = v_channel.shape
    top_half = v_channel[:h // 2, :]
    bottom_half = v_channel[h // 2:, :]
    mean_bottom = np.mean(bottom_half)
    backlight = np.mean(top_half) / mean_bottom if mean_bottom > 0 else 1.0

    return [mean_b, std_b, skew_b, kurt_b, overexposed, underexposed,
            dyn_range, backlight]


def extract_contrast(gray):
    """Features 9-12: Contrast."""
    gray_f = gray.astype(np.float64)
    h, w = gray_f.shape

    gh, gw = h // 8, w // 8
    weber_vals = []
    for i in range(8):
        for j in range(8):
            patch = gray_f[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            mu = np.mean(patch)
            if mu > 1:
                weber_vals.append(np.std(patch) / mu)
    weber = np.mean(weber_vals) if weber_vals else 0.0

    i_max, i_min = np.max(gray_f), np.min(gray_f)
    denom = i_max + i_min
    michelson = (i_max - i_min) / denom if denom > 0 else 0.0

    mean_i = np.mean(gray_f)
    rms = np.std(gray_f) / mean_i if mean_i > 0 else 0.0

    local_stds = []
    for i in range(8):
        for j in range(8):
            patch = gray_f[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            local_stds.append(np.std(patch))
    local_contrast = np.mean(local_stds)

    return [weber, michelson, rms, local_contrast]


def extract_haze(bgr):
    """Features 13-16: Haze/Fog/Atmosphere."""
    min_channel = np.min(bgr, axis=2).astype(np.float64)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)

    dc_mean = np.mean(dark_channel)
    dc_std = np.std(dark_channel)

    dc_flat = dark_channel.flatten()
    n_top = max(1, int(len(dc_flat) * 0.001))
    top_indices = np.argpartition(dc_flat, -n_top)[-n_top:]
    atm_light = np.mean(dc_flat[top_indices])

    if atm_light > 0:
        transmission = np.mean(1.0 - dark_channel / atm_light)
    else:
        transmission = 1.0

    return [dc_mean, dc_std, transmission, atm_light]


def extract_sharpness(gray):
    """Features 17-20: Sharpness/Blur."""
    gray_f = gray.astype(np.float64)

    lap = cv2.Laplacian(gray_f, cv2.CV_64F)
    lap_var = np.var(lap)

    sx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(sx ** 2 + sy ** 2)

    shifted = gray_f[:, 2:]
    original = gray_f[:, :shifted.shape[1]]
    brenner = np.mean((shifted - original) ** 2)

    fft = np.fft.fft2(gray_f)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift) ** 2
    h, w = gray_f.shape
    cy, cx = h // 2, w // 2
    radius = min(cy, cx) // 4
    Y, X = np.ogrid[:h, :w]
    low_mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= radius ** 2
    total_energy = np.sum(magnitude)
    low_energy = np.sum(magnitude[low_mask])
    high_freq_ratio = (total_energy - low_energy) / total_energy if total_energy > 0 else 0.0

    return [lap_var, tenengrad, brenner, high_freq_ratio]


def extract_texture(gray, gray_256):
    """Features 21-30: Texture/Complexity."""
    gray_f = gray.astype(np.float64)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)

    sx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx ** 2 + sy ** 2)
    grad_mean = np.mean(grad_mag)
    grad_std = np.std(grad_mag)

    # GLCM skipped (too expensive, not in the 24 selected features)
    glcm_contrast = np.nan
    glcm_homogeneity = np.nan
    glcm_energy = np.nan
    glcm_correlation = np.nan
    glcm_entropy = np.nan

    # LBP_entropy skipped (not in the 24 selected features)
    lbp_entropy = np.nan
    lbp_uniformity = np.nan

    return [edge_density, grad_mean, grad_std, glcm_contrast, glcm_homogeneity,
            glcm_energy, glcm_entropy, glcm_correlation, lbp_entropy, lbp_uniformity]


def extract_entropy(gray, bgr, gray_256):
    """Features 31-34: Entropy/Information Content."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    spatial_ent = -np.sum(hist * np.log2(hist))

    channel_ents = []
    for c in range(3):
        ch_hist = cv2.calcHist([bgr], [c], None, [256], [0, 256]).flatten()
        ch_hist = ch_hist / (ch_hist.sum() + 1e-10)
        ch_hist = ch_hist[ch_hist > 0]
        channel_ents.append(-np.sum(ch_hist * np.log2(ch_hist)))
    color_ent = np.mean(channel_ents)

    h, w = gray_256.shape
    patch_h, patch_w = 32, 32
    local_ents = []
    for i in range(0, h - patch_h + 1, patch_h):
        for j in range(0, w - patch_w + 1, patch_w):
            patch = gray_256[i:i + patch_h, j:j + patch_w]
            ph = cv2.calcHist([patch], [0], None, [256], [0, 256]).flatten()
            ph = ph / (ph.sum() + 1e-10)
            ph = ph[ph > 0]
            local_ents.append(-np.sum(ph * np.log2(ph)))

    local_ent_mean = np.mean(local_ents) if local_ents else 0.0
    local_ent_std = np.std(local_ents) if local_ents else 0.0

    return [spatial_ent, color_ent, local_ent_mean, local_ent_std]


def extract_color(bgr, hsv):
    """Features 35-40: Color Properties."""
    bgr_f = bgr.astype(np.float64)
    R, G, B = bgr_f[:, :, 2], bgr_f[:, :, 1], bgr_f[:, :, 0]

    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    colorfulness = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)

    s_channel = hsv[:, :, 1].astype(np.float64)
    mean_sat = np.mean(s_channel)
    std_sat = np.std(s_channel)

    h_channel = hsv[:, :, 0].astype(np.float64)
    s_mask = s_channel > (0.1 * 255)
    if np.sum(s_mask) > 10:
        hue_vals = h_channel[s_mask]
        hue_rad = hue_vals * (2 * np.pi / 180.0)
        sin_sum = np.mean(np.sin(hue_rad))
        cos_sum = np.mean(np.cos(hue_rad))
        R_len = np.sqrt(sin_sum ** 2 + cos_sum ** 2)
        hue_std_val = np.sqrt(-2 * np.log(max(R_len, 1e-10)))
    else:
        hue_std_val = 0.0

    b_q = (bgr[:, :, 0] // 64).flatten()
    g_q = (bgr[:, :, 1] // 64).flatten()
    r_q = (bgr[:, :, 2] // 64).flatten()
    color_idx = r_q * 16 + g_q * 4 + b_q
    hist_64 = np.bincount(color_idx, minlength=64)
    dominant_ratio = np.max(hist_64) / np.sum(hist_64) if np.sum(hist_64) > 0 else 0.0

    r_flat, g_flat, b_flat = R.flatten(), G.flatten(), B.flatten()
    corr_rg = np.corrcoef(r_flat, g_flat)[0, 1]
    corr_rb = np.corrcoef(r_flat, b_flat)[0, 1]
    corr_gb = np.corrcoef(g_flat, b_flat)[0, 1]
    color_corr = np.mean([corr_rg, corr_rb, corr_gb])

    return [colorfulness, mean_sat, std_sat, hue_std_val, dominant_ratio, color_corr]


def extract_frequency(gray):
    """Features 41-43: Frequency Domain."""
    gray_f = gray.astype(np.float64)
    h, w = gray_f.shape

    fft = np.fft.fft2(gray_f)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift) ** 2

    cy, cx = h // 2, w // 2
    Y, X = np.mgrid[:h, :w]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).flatten()
    mag_flat = magnitude.flatten()

    R_int = R.astype(int)
    max_r = int(np.max(R_int)) + 1
    radial_energy = np.bincount(R_int, weights=mag_flat, minlength=max_r)

    total_energy = np.sum(radial_energy)
    freqs = np.arange(max_r)

    if total_energy > 0:
        spectral_centroid = np.sum(freqs * radial_energy) / total_energy
    else:
        spectral_centroid = 0.0

    cumulative = np.cumsum(radial_energy)
    threshold = 0.85 * total_energy
    rolloff_idx = np.searchsorted(cumulative, threshold)
    spectral_rolloff = float(rolloff_idx)

    # Wavelet skipped (not in the 24 selected features)
    wavelet_ratio = np.nan

    return [spectral_centroid, spectral_rolloff, wavelet_ratio]


def extract_spatial(gray, bgr, mog2):
    """Features 44-46: Spatial Structure."""
    fast = cv2.FastFeatureDetector_create(threshold=20)
    keypoints = fast.detect(gray, None)
    h, w = gray.shape
    fast_count = len(keypoints) / (h * w)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    num_cc, _ = cv2.connectedComponents(opened)
    cc_count = num_cc - 1

    if mog2 is not None:
        fg_mask = mog2.apply(bgr)
        fg_ratio = np.mean(fg_mask > 0)
        if fg_ratio > 0.90:
            fg_ratio = np.nan
    else:
        fg_ratio = np.nan

    return [fast_count, cc_count, fg_ratio]


def extract_temporal(gray_480, gray_256, gray, prev_gray_480, prev_gray_256,
                     prev_gray, prev_brightness, prev_edge_density):
    """Features 47-56: Motion/Temporal."""
    if prev_gray_480 is None:
        return [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_480, gray_480, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_mag_mean = float(np.mean(mag))
    flow_mag_std = float(np.std(mag))

    angle_flat = angle.flatten()
    sin_mean = np.mean(np.sin(angle_flat))
    cos_mean = np.mean(np.cos(angle_flat))
    R_len = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    flow_angle_var = 1.0 - R_len

    diff = cv2.absdiff(gray, prev_gray).astype(np.float64)
    frame_diff_mean = np.mean(diff)

    ssim_val = ssim(prev_gray_256, gray_256)

    hist_curr = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256]).flatten()
    hist_curr = hist_curr / (hist_curr.sum() + 1e-10)
    hist_prev = hist_prev / (hist_prev.sum() + 1e-10)
    hist_inter = np.sum(np.minimum(hist_curr, hist_prev))

    curr_brightness = np.mean(gray)
    brightness_delta = curr_brightness - prev_brightness if prev_brightness is not None else 0.0

    edges = cv2.Canny(gray, 100, 200)
    curr_edge_density = np.mean(edges > 0)
    edge_delta = curr_edge_density - prev_edge_density if prev_edge_density is not None else 0.0

    median_angle = np.median(angle_flat)
    angle_diff = np.abs(angle_flat - median_angle)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    coherence = np.mean(angle_diff < np.radians(15))

    scene_cut = 1.0 - hist_inter

    return [flow_mag_mean, flow_mag_std, flow_angle_var, frame_diff_mean,
            ssim_val, hist_inter, brightness_delta, edge_delta,
            coherence, scene_cut]


# ── Main Pipeline ────────────────────────────────────────────────

def process_sequence(seq_name, global_frame_count):
    """Extract all 56 features for every frame in a sequence."""
    seq_dir = UAVDT_FRAMES / seq_name
    frame_files = sorted(seq_dir.glob("img*.jpg"))
    n_frames = len(frame_files)
    print(f"  {seq_name}: {n_frames} frames", flush=True)

    records = []

    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
    init_limit = min(30, n_frames)
    for i in range(init_limit):
        init_frame = cv2.imread(str(frame_files[i]))
        if init_frame is not None:
            mog2.apply(init_frame)

    prev_gray = None
    prev_gray_480 = None
    prev_gray_256 = None
    prev_brightness = None
    prev_edge_density = None

    for idx, fpath in enumerate(frame_files):
        frame_id = int(fpath.stem.replace("img", ""))
        bgr = cv2.imread(str(fpath))
        if bgr is None:
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        gray_256 = cv2.resize(gray, (256, 256))
        h_orig, w_orig = gray.shape
        scale_480 = 480 / h_orig
        new_w = int(w_orig * scale_480)
        gray_480 = cv2.resize(gray, (max(new_w, 1), 480))

        feat_illum = extract_illumination(hsv, v_channel)
        feat_contrast = extract_contrast(gray)
        feat_haze = extract_haze(bgr)
        feat_sharp = extract_sharpness(gray)
        feat_texture = extract_texture(gray, gray_256)
        feat_entropy = extract_entropy(gray, bgr, gray_256)
        feat_color = extract_color(bgr, hsv)
        feat_freq = extract_frequency(gray)
        feat_spatial = extract_spatial(gray, bgr, mog2 if idx >= init_limit else None)
        feat_temporal = extract_temporal(
            gray_480, gray_256, gray,
            prev_gray_480, prev_gray_256, prev_gray,
            prev_brightness, prev_edge_density
        )

        all_features = (feat_illum + feat_contrast + feat_haze + feat_sharp +
                        feat_texture + feat_entropy + feat_color + feat_freq +
                        feat_spatial + feat_temporal)

        records.append([seq_name, frame_id] + all_features)

        prev_gray = gray.copy()
        prev_gray_480 = gray_480.copy()
        prev_gray_256 = gray_256.copy()
        prev_brightness = float(np.mean(gray))
        edges_prev = cv2.Canny(gray, 100, 200)
        prev_edge_density = float(np.mean(edges_prev > 0))

        total_done = global_frame_count + idx + 1
        if total_done % 1000 == 0:
            print(f"    Progress: {total_done} total frames processed", flush=True)

    return records


if __name__ == "__main__":
    print("UAVDT 56-Feature Extraction — Remaining 30 Sequences", flush=True)
    print("=" * 60, flush=True)

    # Discover remaining sequences
    all_seqs = sorted(d.name for d in UAVDT_FRAMES.iterdir() if d.is_dir())
    remaining_seqs = [s for s in all_seqs if s not in DONE_SEQUENCES]
    print(f"Total sequences on disk: {len(all_seqs)}", flush=True)
    print(f"Already done: {len(DONE_SEQUENCES)}", flush=True)
    print(f"Remaining: {len(remaining_seqs)}", flush=True)
    print(f"Sequences: {remaining_seqs}", flush=True)

    columns = ["sequence", "frame_id"] + FEATURE_NAMES

    # Resume: skip sequences already in output CSV
    all_records = []
    done_in_csv = set()
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        done_in_csv = set(existing["sequence"].unique())
        all_records = existing.values.tolist()
        print(f"Resuming: {len(done_in_csv)} sequences already done "
              f"({len(existing)} frames)", flush=True)

    t0 = time.time()
    global_frame_count = len(all_records)

    for i, seq in enumerate(remaining_seqs):
        if seq in done_in_csv:
            print(f"\n[{i + 1}/{len(remaining_seqs)}] Skipping {seq} (already done)",
                  flush=True)
            continue

        print(f"\n[{i + 1}/{len(remaining_seqs)}] Processing {seq}...", flush=True)
        seq_t0 = time.time()
        records = process_sequence(seq, global_frame_count)
        all_records.extend(records)
        global_frame_count += len(records)
        elapsed = time.time() - seq_t0
        ms_per_frame = (elapsed / len(records) * 1000) if records else 0

        # ETA
        seqs_left = [s for s in remaining_seqs[i + 1:] if s not in done_in_csv]
        frames_remaining = sum(len(list((UAVDT_FRAMES / s).glob("img*.jpg")))
                               for s in seqs_left)
        eta_s = frames_remaining * elapsed / len(records) if records else 0

        print(f"    Done: {len(records)} frames in {elapsed:.1f}s "
              f"({ms_per_frame:.0f} ms/frame, ETA ~{eta_s / 60:.0f} min remaining)",
              flush=True)

        # Checkpoint after each sequence
        df = pd.DataFrame(all_records, columns=columns)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"    Saved checkpoint ({len(df)} total frames, "
              f"{df['sequence'].nunique()} sequences)", flush=True)

    total_time = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"Extraction complete. Total time: {total_time:.0f}s", flush=True)
    print(f"Output: {OUTPUT_CSV} ({global_frame_count} frames)", flush=True)

    # ── Merge with original 20-sequence CSV ──────────────────────
    print(f"\n{'=' * 60}", flush=True)
    print("Merging with original 20-sequence features...", flush=True)

    if not ORIGINAL_CSV.exists():
        print(f"  ERROR: Original CSV not found: {ORIGINAL_CSV}", flush=True)
    else:
        df_orig = pd.read_csv(ORIGINAL_CSV)
        df_new = pd.read_csv(OUTPUT_CSV)

        # Verify columns match
        if list(df_orig.columns) != list(df_new.columns):
            print(f"  ERROR: Column mismatch!", flush=True)
            print(f"    Original: {list(df_orig.columns)[:5]}...", flush=True)
            print(f"    New:      {list(df_new.columns)[:5]}...", flush=True)
        else:
            df_all = pd.concat([df_orig, df_new], ignore_index=True)
            df_all = df_all.sort_values(["sequence", "frame_id"]).reset_index(drop=True)
            df_all.to_csv(COMBINED_CSV, index=False)

            print(f"  Original: {len(df_orig)} frames, "
                  f"{df_orig['sequence'].nunique()} sequences", flush=True)
            print(f"  New:      {len(df_new)} frames, "
                  f"{df_new['sequence'].nunique()} sequences", flush=True)
            print(f"  Combined: {len(df_all)} frames, "
                  f"{df_all['sequence'].nunique()} sequences", flush=True)
            print(f"  Saved to: {COMBINED_CSV}", flush=True)

    print("\nDone.", flush=True)
