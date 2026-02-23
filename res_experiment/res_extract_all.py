"""
Extract 14 selected features on ALL 100 UA-DETRAC sequences (138,252 frames).
All features computed inline for speed (~10-15ms/frame).
Crash-safe: checkpoints after each sequence, resumes on restart.
"""

import time
import cv2
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis
from skimage.metrics import structural_similarity as ssim
import pywt

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DETRAC_IMAGES = Path("/home/aljouhara/Desktop/Detrac Dataset/DETRAC-Images")
OUTPUT_CSV = SCRIPT_DIR / "detrac_14features.csv"

# ── Feature names (column order) ─────────────────────────────────
FEATURE_NAMES = [
    "gradient_magnitude_std",
    "brightness_skewness",
    "overexposed_ratio",
    "contour_count",
    "wavelet_detail_energy",
    "high_freq_energy_ratio",
    "mean_brightness",
    "spectral_rolloff",
    "brightness_kurtosis",
    "dark_channel_std",
    "color_channel_corr",
    "FAST_keypoint_count",
    "ssim_prev",
    "temporal_diff_mean",
]

COLUMNS = ["sequence", "frame_id"] + FEATURE_NAMES
HEADER_LINE = ",".join(COLUMNS) + "\n"

# Pre-create reusable objects
DARK_CHANNEL_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
FAST_DETECTOR = cv2.FastFeatureDetector_create(threshold=20)


def extract_14(bgr, gray, gray_f, hsv, v_flat, gray_256, prev_gray, prev_gray_256):
    """Compute 14 features for a single frame. Returns list of 14 floats."""
    h, w = gray.shape

    # 1. gradient_magnitude_std — Sobel
    sx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx ** 2 + sy ** 2)
    gradient_magnitude_std = float(np.std(grad_mag))

    # 2. brightness_skewness — V channel
    brightness_skewness = float(skew(v_flat))

    # 3. overexposed_ratio — V channel
    v_channel = hsv[:, :, 2]
    overexposed_ratio = float(np.mean(v_channel > 240))

    # 4. contour_count — Canny(50,150) matching js4_density_proxy
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    # 5. wavelet_detail_energy — Haar DWT level 1
    cA, (cH, cV, cD) = pywt.dwt2(gray_f, 'haar')
    detail_energy = np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)
    approx_energy = np.sum(cA ** 2)
    total_wav = detail_energy + approx_energy
    wavelet_detail_energy = float(detail_energy / total_wav) if total_wav > 0 else 0.0

    # Shared FFT for features 6 and 8
    fft = np.fft.fft2(gray_f)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift) ** 2
    cy, cx = h // 2, w // 2

    # 6. high_freq_energy_ratio
    radius = min(cy, cx) // 4
    Y, X = np.ogrid[:h, :w]
    low_mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= radius ** 2
    total_energy = np.sum(magnitude)
    low_energy = np.sum(magnitude[low_mask])
    high_freq_energy_ratio = float((total_energy - low_energy) / total_energy) if total_energy > 0 else 0.0

    # 7. mean_brightness — V channel
    mean_brightness = float(np.mean(v_flat))

    # 8. spectral_rolloff — radial FFT profile
    Y_full, X_full = np.mgrid[:h, :w]
    R = np.sqrt((Y_full - cy) ** 2 + (X_full - cx) ** 2).flatten()
    R_int = R.astype(int)
    max_r = int(np.max(R_int)) + 1
    radial_energy = np.bincount(R_int, weights=magnitude.flatten(), minlength=max_r)
    cumulative = np.cumsum(radial_energy)
    threshold_85 = 0.85 * np.sum(radial_energy)
    spectral_rolloff = float(np.searchsorted(cumulative, threshold_85))

    # 9. brightness_kurtosis — V channel
    brightness_kurtosis_val = float(kurtosis(v_flat))

    # 10. dark_channel_std
    min_channel = np.min(bgr, axis=2).astype(np.float64)
    dark_channel = cv2.erode(min_channel, DARK_CHANNEL_KERNEL)
    dark_channel_std = float(np.std(dark_channel))

    # 11. color_channel_corr
    bgr_f = bgr.astype(np.float64)
    r_flat = bgr_f[:, :, 2].flatten()
    g_flat = bgr_f[:, :, 1].flatten()
    b_flat = bgr_f[:, :, 0].flatten()
    corr_rg = np.corrcoef(r_flat, g_flat)[0, 1]
    corr_rb = np.corrcoef(r_flat, b_flat)[0, 1]
    corr_gb = np.corrcoef(g_flat, b_flat)[0, 1]
    color_channel_corr = float(np.mean([corr_rg, corr_rb, corr_gb]))

    # 12. FAST_keypoint_count — normalized by image area
    keypoints = FAST_DETECTOR.detect(gray, None)
    fast_keypoint_count = len(keypoints) / (h * w)

    # 13. ssim_prev
    if prev_gray_256 is not None:
        ssim_prev = float(ssim(prev_gray_256, gray_256))
    else:
        ssim_prev = 1.0

    # 14. temporal_diff_mean — raw 0-255 scale
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray).astype(np.float64)
        temporal_diff_mean = float(np.mean(diff))
    else:
        temporal_diff_mean = 0.0

    return [
        gradient_magnitude_std, brightness_skewness, overexposed_ratio,
        contour_count, wavelet_detail_energy, high_freq_energy_ratio,
        mean_brightness, spectral_rolloff, brightness_kurtosis_val,
        dark_channel_std, color_channel_corr, fast_keypoint_count,
        ssim_prev, temporal_diff_mean,
    ]


def process_sequence(seq_name):
    """Extract 14 features for all frames in a sequence. Returns list of records."""
    seq_dir = DETRAC_IMAGES / seq_name
    frame_files = sorted(seq_dir.glob("img*.jpg"))
    n_frames = len(frame_files)

    records = []
    prev_gray = None
    prev_gray_256 = None

    for idx, fpath in enumerate(frame_files):
        frame_id = int(fpath.stem.replace("img", ""))
        bgr = cv2.imread(str(fpath))
        if bgr is None:
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_flat = hsv[:, :, 2].flatten().astype(np.float64)
        gray_f = gray.astype(np.float64)
        gray_256 = cv2.resize(gray, (256, 256))

        features = extract_14(bgr, gray, gray_f, hsv, v_flat, gray_256,
                              prev_gray, prev_gray_256)

        records.append([seq_name, frame_id] + features)

        prev_gray = gray
        prev_gray_256 = gray_256

    return records, n_frames


def main():
    # Discover all sequences
    all_seqs = sorted([d.name for d in DETRAC_IMAGES.iterdir() if d.is_dir()])
    n_seqs = len(all_seqs)
    print(f"Found {n_seqs} sequences in {DETRAC_IMAGES}")

    # Resume: find already-done sequences
    done_seqs = set()
    if OUTPUT_CSV.exists():
        import pandas as pd
        existing = pd.read_csv(OUTPUT_CSV)
        done_seqs = set(existing["sequence"].unique())
        print(f"Resuming: {len(done_seqs)} sequences already done ({len(existing)} frames)")
        del existing

    total_frames = 0
    total_time = 0.0
    t_global = time.time()

    for i, seq in enumerate(all_seqs):
        if seq in done_seqs:
            print(f"[{i+1:>3}/{n_seqs}] {seq} — skipped (done)")
            continue

        t0 = time.time()
        records, n_frames = process_sequence(seq)
        elapsed = time.time() - t0
        ms_per_frame = (elapsed / len(records) * 1000) if records else 0

        total_frames += len(records)
        total_time += elapsed

        # Append to CSV
        write_header = not OUTPUT_CSV.exists()
        with open(OUTPUT_CSV, "a") as f:
            if write_header:
                f.write(HEADER_LINE)
            for rec in records:
                f.write(",".join(str(v) for v in rec) + "\n")

        # ETA
        remaining_seqs = n_seqs - i - 1 - len(done_seqs)
        if total_frames > 0:
            avg_ms = total_time / total_frames * 1000
            # Estimate remaining frames
            frames_done_now = total_frames
            remaining_est = remaining_seqs * (len(records))  # rough estimate
            eta_min = remaining_est * avg_ms / 1000 / 60
        else:
            avg_ms = 0
            eta_min = 0

        print(f"[{i+1:>3}/{n_seqs}] {seq}: {len(records)} frames, "
              f"{elapsed:.1f}s ({ms_per_frame:.0f} ms/frame), "
              f"total={total_frames}, ETA ~{eta_min:.0f} min")

    global_elapsed = time.time() - t_global
    print(f"\nDone. {total_frames} frames, {global_elapsed:.0f}s total")
    print(f"Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
