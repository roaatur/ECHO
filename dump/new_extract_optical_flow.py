#!/usr/bin/env python3
"""Extract dense optical flow features on DETRAC.

Uses Farneback optical flow at 320x180 resolution for speed.
Extracts per-frame:
  - flow_magnitude_mean, flow_magnitude_std
  - flow_direction_entropy (36-bin)
  - flow_high_motion_ratio (% pixels with |flow| > threshold)
  - flow_top_magnitude_mean, flow_mid_magnitude_mean, flow_bot_magnitude_mean (strip-level)
  - flow_spatial_variance (how spread out the motion is)
  - flow_acceleration_mean (change in flow magnitude from prev frame)

Appends to existing new_all_features.csv (merges on sequence + frame_id).
Output: new_features/new_optical_flow_features.csv (standalone)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_BASE = Path("/home/aljouhara/Desktop/Detrac Dataset/DETRAC-Images")
EXISTING_CSV = SCRIPT_DIR / "new_all_features.csv"
OUT_CSV = SCRIPT_DIR / "new_optical_flow_features.csv"

# Flow computation resolution
FLOW_W, FLOW_H = 320, 180

# Farneback parameters
FLOW_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)

HIGH_MOTION_THRESH = 2.0  # pixels


def extract_flow_features(flow, prev_mag=None):
    """Extract features from a dense optical flow field."""
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    h, w = mag.shape

    features = {}

    # Global magnitude stats
    features["flow_magnitude_mean"] = float(np.mean(mag))
    features["flow_magnitude_std"] = float(np.std(mag))
    features["flow_magnitude_max"] = float(np.percentile(mag, 99))

    # High motion ratio
    features["flow_high_motion_ratio"] = float((mag > HIGH_MOTION_THRESH).mean())

    # Direction entropy (only for pixels with significant motion)
    mask = mag > 1.0
    if mask.sum() > 100:
        hist_dir = np.histogram(angle[mask], bins=36, range=(-np.pi, np.pi))[0].astype(np.float64)
        hist_dir = hist_dir / hist_dir.sum()
        hist_dir = hist_dir[hist_dir > 0]
        features["flow_direction_entropy"] = float(-np.sum(hist_dir * np.log2(hist_dir)))
    else:
        features["flow_direction_entropy"] = 0.0

    # Strip-level flow (top/mid/bot thirds)
    third = h // 3
    features["flow_top_magnitude"] = float(np.mean(mag[:third]))
    features["flow_mid_magnitude"] = float(np.mean(mag[third:2*third]))
    features["flow_bot_magnitude"] = float(np.mean(mag[2*third:]))

    # Spatial variance of flow (how concentrated vs spread)
    # Divide into 8x8 grid, compute mean mag per cell, then std of cell means
    grid_h, grid_w = 8, 8
    cell_h, cell_w = h // grid_h, w // grid_w
    cell_means = []
    for gi in range(grid_h):
        for gj in range(grid_w):
            cell = mag[gi*cell_h:(gi+1)*cell_h, gj*cell_w:(gj+1)*cell_w]
            cell_means.append(cell.mean())
    features["flow_spatial_variance"] = float(np.std(cell_means))

    # Flow acceleration (change from previous frame's magnitude)
    if prev_mag is not None:
        features["flow_acceleration_mean"] = float(np.mean(mag) - np.mean(prev_mag))
        features["flow_acceleration_abs"] = float(abs(np.mean(mag) - np.mean(prev_mag)))
    else:
        features["flow_acceleration_mean"] = 0.0
        features["flow_acceleration_abs"] = 0.0

    return features, mag


def main():
    # Load existing CSV to get sequence + frame_id list
    print("Loading existing features CSV ...", flush=True)
    existing = pd.read_csv(EXISTING_CSV)
    sequences = sorted(existing["sequence"].unique())
    print(f"  {len(sequences)} sequences, {len(existing)} total frames", flush=True)

    # Build frame list per sequence
    seq_frames = {}
    for seq in sequences:
        frames = sorted(existing[existing["sequence"] == seq]["frame_id"].values)
        seq_frames[seq] = frames

    all_rows = []
    total_done = 0

    for si, seq in enumerate(sequences):
        img_dir = IMG_BASE / seq
        if not img_dir.exists():
            print(f"  [{si+1}/{len(sequences)}] {seq}: directory not found, skipping", flush=True)
            continue

        frames = seq_frames[seq]
        prev_gray = None
        prev_mag = None

        for fi, frame_id in enumerate(frames):
            img_path = img_dir / f"img{frame_id:05d}.jpg"
            if not img_path.exists():
                # Try without leading zeros
                img_path = img_dir / f"img{frame_id}.jpg"
                if not img_path.exists():
                    all_rows.append({"sequence": seq, "frame_id": frame_id,
                                     **{k: 0.0 for k in ["flow_magnitude_mean", "flow_magnitude_std",
                                        "flow_magnitude_max", "flow_high_motion_ratio",
                                        "flow_direction_entropy", "flow_top_magnitude",
                                        "flow_mid_magnitude", "flow_bot_magnitude",
                                        "flow_spatial_variance", "flow_acceleration_mean",
                                        "flow_acceleration_abs"]}})
                    continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (FLOW_W, FLOW_H))

            if prev_gray is None:
                # First frame — no flow
                features = {
                    "flow_magnitude_mean": 0.0, "flow_magnitude_std": 0.0,
                    "flow_magnitude_max": 0.0, "flow_high_motion_ratio": 0.0,
                    "flow_direction_entropy": 0.0,
                    "flow_top_magnitude": 0.0, "flow_mid_magnitude": 0.0,
                    "flow_bot_magnitude": 0.0,
                    "flow_spatial_variance": 0.0,
                    "flow_acceleration_mean": 0.0, "flow_acceleration_abs": 0.0,
                }
                prev_mag = None
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, small_gray, None, **FLOW_PARAMS)
                features, prev_mag = extract_flow_features(flow, prev_mag)

            prev_gray = small_gray

            row = {"sequence": seq, "frame_id": frame_id}
            row.update(features)
            all_rows.append(row)
            total_done += 1

        print(f"  [{si+1}/{len(sequences)}] {seq}: {len(frames)} frames done "
              f"(total: {total_done})", flush=True)

    # Save
    flow_df = pd.DataFrame(all_rows)
    flow_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(flow_df)} rows to {OUT_CSV}", flush=True)

    # Print sample stats
    flow_cols = [c for c in flow_df.columns if c.startswith("flow_")]
    print(f"\nFeature statistics:", flush=True)
    for col in flow_cols:
        vals = flow_df[col].values
        print(f"  {col:35s} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
