"""Diagnostic on js8 data pipeline — no training, just stats."""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # yolo_runs_on_uavdt/
JAYS = ROOT / "jays"

FEAT_CSV_UAVDT = JAYS / "jays_uavdt_all50_features.csv"
FEAT_CSV_VISDRONE = JAYS / "jays_visdrone_features.csv"
RES_CSV_UAVDT = ROOT / "uavdt_all50_results.csv"
RES_CSV_VISDRONE = ROOT / "visdrone_classagnostic_results.csv"

STATIC_FEATURES = [
    "rms_contrast", "weber_contrast", "transmission_estimate", "saturation_std",
    "brightness_std", "mean_brightness", "mean_saturation", "local_entropy_std",
    "dark_channel_std", "dark_channel_mean", "overexposed_ratio", "underexposed_ratio",
    "high_freq_energy_ratio", "colorfulness",
]
TEMPORAL_FEATURES = [
    "optical_flow_mag_mean", "optical_flow_mag_std", "optical_flow_angle_var",
    "frame_difference_mean", "SSIM_prev_frame", "histogram_intersection",
    "brightness_delta", "edge_density_delta", "flow_coherence", "scene_cut_score",
]
FEATURES = STATIC_FEATURES + TEMPORAL_FEATURES

T_CONTEXT = 20
T_HORIZON = 10


def load_merged():
    feat = pd.concat([pd.read_csv(FEAT_CSV_UAVDT), pd.read_csv(FEAT_CSV_VISDRONE)],
                     ignore_index=True)
    res = pd.concat([pd.read_csv(RES_CSV_UAVDT), pd.read_csv(RES_CSV_VISDRONE)],
                    ignore_index=True)

    nano = res[res["model"] == "yolo11n.pt"][["sequence", "frame_id", "recall", "num_gt"]].copy()
    nano = nano.rename(columns={"recall": "recall_nano"})
    medium = res[res["model"] == "yolo11m.pt"][["sequence", "frame_id", "recall"]].copy()
    medium = medium.rename(columns={"recall": "recall_medium"})

    merged = nano.merge(medium, on=["sequence", "frame_id"])
    merged["recall_gap"] = merged["recall_medium"] - merged["recall_nano"]

    df = feat.merge(merged[["sequence", "frame_id", "recall_gap", "num_gt"]],
                    on=["sequence", "frame_id"])
    df = df[df["num_gt"] >= 1]
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df


def main():
    df = load_merged()
    print(f"Loaded: {len(df)} rows, {df['sequence'].nunique()} sequences\n")

    # ── 1. Frame continuity check ────────────────────────────────
    print("=" * 70)
    print("1. FRAME CONTINUITY CHECK")
    print("=" * 70)
    gap_seqs = 0
    for seq, sdf in df.groupby("sequence"):
        fids = sdf.sort_values("frame_id")["frame_id"].values
        diffs = np.diff(fids)
        non_one = diffs[diffs != 1]
        if len(non_one) > 0:
            gap_seqs += 1
            if gap_seqs <= 20:  # print first 20
                print(f"  {seq}: {len(fids)} frames, {len(non_one)} gaps "
                      f"(sizes: {sorted(set(non_one))[:10]}{'...' if len(set(non_one)) > 10 else ''})")
    total_seqs = df["sequence"].nunique()
    print(f"\n  {gap_seqs}/{total_seqs} sequences have frame gaps")
    print(f"  {total_seqs - gap_seqs}/{total_seqs} sequences are fully consecutive\n")

    # ── 2. Window diversity check (stride=1) ─────────────────────
    print("=" * 70)
    print("2. WINDOW DIVERSITY CHECK (stride=1)")
    print("=" * 70)
    total_window = T_CONTEXT + T_HORIZON
    print(f"  Window size: {T_CONTEXT} context + {T_HORIZON} horizon = {total_window} frames")
    print(f"  {'Sequence':<30s} {'Frames':>8} {'Windows':>8} {'Ratio':>8}")
    print("  " + "-" * 56)

    total_frames = 0
    total_windows = 0
    for seq, sdf in sorted(df.groupby("sequence"), key=lambda x: x[0]):
        n = len(sdf)
        w = max(0, n - total_window + 1)
        ratio = w / n if n > 0 else 0
        total_frames += n
        total_windows += w
        if n < total_window:
            print(f"  {seq:<30s} {n:>8d} {w:>8d} {ratio:>8.3f}  *** TOO SHORT")
        # Only print extremes to keep output manageable
    print(f"  {'TOTAL':<30s} {total_frames:>8d} {total_windows:>8d} "
          f"{total_windows/total_frames:.3f}")
    # Summary stats
    seq_stats = []
    for seq, sdf in df.groupby("sequence"):
        n = len(sdf)
        w = max(0, n - total_window + 1)
        seq_stats.append({"seq": seq, "frames": n, "windows": w})
    sdf_stats = pd.DataFrame(seq_stats)
    short = sdf_stats[sdf_stats["windows"] == 0]
    if len(short) > 0:
        print(f"\n  Sequences too short for any window ({total_window} frames needed):")
        for _, r in short.iterrows():
            print(f"    {r['seq']}: {r['frames']} frames")
    print(f"\n  Windows per sequence: min={sdf_stats['windows'].min()}, "
          f"median={sdf_stats['windows'].median():.0f}, "
          f"max={sdf_stats['windows'].max()}, "
          f"mean={sdf_stats['windows'].mean():.1f}\n")

    # ── 3. Target leakage check ──────────────────────────────────
    print("=" * 70)
    print("3. TARGET LEAKAGE CHECK (center=True, window=10)")
    print("=" * 70)
    print()
    print("  With center=True rolling mean (window=10, min_periods=1):")
    print("    smoothed_gap[k] = mean(raw_gap[k-4 : k+6])  (when enough data)")
    print()
    print("  For a sliding window with input frames [i, i+19]:")
    print("    target = smoothed_gap[i+20 : i+30]")
    print()
    print("  Target frame i+20:")
    print("    smoothed_gap[i+20] = mean(raw_gap[i+16 : i+25])")
    print("    Input frames:  i to i+19")
    print("    Raw gap used:  i+16 to i+24")
    print("    OVERLAP:       frames i+16 to i+19  (4 frames)")
    print("    Overlap fraction: 4/9 = 44.4% of target computation")
    print()
    print("  Target frame i+21:")
    print("    smoothed_gap[i+21] = mean(raw_gap[i+17 : i+26])")
    print("    OVERLAP:       frames i+17 to i+19  (3 frames)")
    print("    Overlap fraction: 3/9 = 33.3%")
    print()
    print("  Target frame i+22:")
    print("    smoothed_gap[i+22] = mean(raw_gap[i+18 : i+27])")
    print("    OVERLAP:       frames i+18 to i+19  (2 frames)")
    print("    Overlap fraction: 2/9 = 22.2%")
    print()
    print("  Target frame i+23:")
    print("    smoothed_gap[i+23] = mean(raw_gap[i+19 : i+28])")
    print("    OVERLAP:       frame i+19  (1 frame)")
    print("    Overlap fraction: 1/9 = 11.1%")
    print()
    print("  Target frames i+24 to i+29: NO overlap")
    print()
    print("  SUMMARY: 4 of 10 target steps have leakage from input frames.")
    print("  Average leakage: (4+3+2+1)/(9*10) = 10/90 = 11.1% of target info")
    print("  comes from raw values also seen as input features.")
    print()

    # Empirical verification
    print("  Empirical verification on first sequence:")
    first_seq = sorted(df["sequence"].unique())[0]
    sdf = df[df["sequence"] == first_seq].sort_values("frame_id").reset_index(drop=True)
    raw_gap = sdf["recall_gap"].values.copy()

    # Apply center=True smoothing
    smoothed_center = sdf["recall_gap"].rolling(10, center=True, min_periods=1).mean().values

    # Check a specific window
    i = 0
    if len(sdf) >= T_CONTEXT + T_HORIZON:
        target_start = i + T_CONTEXT  # frame index 20
        target_val = smoothed_center[target_start]

        # What raw values does this use?
        # center=True with window=10: for index k, uses k-4 to k+5
        k = target_start
        raw_start = max(0, k - 4)
        raw_end = min(len(raw_gap), k + 6)
        raw_used = raw_gap[raw_start:raw_end]
        manual_mean = np.mean(raw_used)

        print(f"    Sequence: {first_seq}, window i=0")
        print(f"    Target index {target_start}: smoothed={smoothed_center[target_start]:.6f}")
        print(f"    Manual mean of raw_gap[{raw_start}:{raw_end}] = {manual_mean:.6f}")
        print(f"    Match: {np.isclose(target_val, manual_mean)}")
        print(f"    Input indices: 0 to 19")
        print(f"    Raw gap indices used by target[20]: {raw_start} to {raw_end-1}")
        overlap_indices = set(range(i, i + T_CONTEXT)) & set(range(raw_start, raw_end))
        print(f"    Overlapping indices: {sorted(overlap_indices)}")
        print(f"    Overlap count: {len(overlap_indices)} / {raw_end - raw_start} "
              f"= {len(overlap_indices)/(raw_end-raw_start)*100:.1f}%")
    print()

    # ── 4. Stride=10 check ───────────────────────────────────────
    print("=" * 70)
    print("4. STRIDE=10 WINDOW COUNT")
    print("=" * 70)
    total_windows_s10 = 0
    seq_windows_s10 = []
    for seq, sdf in df.groupby("sequence"):
        n = len(sdf)
        w = len(range(0, max(0, n - total_window + 1), 10))
        total_windows_s10 += w
        seq_windows_s10.append(w)
    print(f"  Stride=1:  {total_windows:>8d} total windows")
    print(f"  Stride=10: {total_windows_s10:>8d} total windows")
    print(f"  Reduction: {total_windows_s10/total_windows*100:.1f}%")
    seq_windows_s10 = np.array(seq_windows_s10)
    print(f"  Per-sequence: min={seq_windows_s10.min()}, "
          f"median={np.median(seq_windows_s10):.0f}, "
          f"max={seq_windows_s10.max()}\n")

    # ── 5. center=False fix ──────────────────────────────────────
    print("=" * 70)
    print("5. FIX: center=False (backward-looking smoothing)")
    print("=" * 70)
    print()
    print("  With center=False rolling mean (window=10, min_periods=1):")
    print("    smoothed_gap[k] = mean(raw_gap[k-9 : k+1])  (backward-looking)")
    print()
    print("  For input frames [i, i+19], target = smoothed_gap[i+20 : i+30]:")
    print()
    print("  Target frame i+20:")
    print("    smoothed_gap[i+20] = mean(raw_gap[i+11 : i+21])")
    print("    Input frames:  i to i+19")
    print("    Raw gap used:  i+11 to i+20")
    print("    OVERLAP:       frames i+11 to i+19  (9 frames)")
    print("    ** STILL LEAKS — backward window extends into input range **")
    print()
    print("  CORRECTION: center=False means gap[k] uses gap[k-9..k].")
    print("  So smoothed_gap[i+20] = mean(raw_gap[i+11 .. i+20]).")
    print("  Input frames are i..i+19. Overlap = i+11..i+19 = 9 frames.")
    print("  This is WORSE, not better.")
    print()
    print("  ACTUAL FIX: use FORWARD-looking smoothing on target only:")
    print("    smoothed_gap[k] = mean(raw_gap[k : k+10])  (forward-looking)")
    print("    Then smoothed_gap[i+20] = mean(raw_gap[i+20 : i+30])")
    print("    Input frames i..i+19 have ZERO overlap with i+20..i+29.")
    print("    This completely eliminates leakage.")
    print()

    # Empirical: verify forward-looking eliminates leakage
    print("  Verification of forward-looking approach:")
    if len(sdf) >= T_CONTEXT + T_HORIZON + 10:
        raw = sdf["recall_gap"].values.copy()
        # Forward-looking: smoothed[k] = mean(raw[k:k+10])
        n = len(raw)
        smoothed_fwd = np.full(n, np.nan)
        for k in range(n):
            end = min(k + 10, n)
            smoothed_fwd[k] = np.mean(raw[k:end])

        i = 0
        target_start = i + T_CONTEXT
        print(f"    Input frames: {i} to {i + T_CONTEXT - 1}")
        print(f"    Target smoothed_fwd[{target_start}] = mean(raw[{target_start}:{target_start+10}])")
        print(f"    Raw frames used: {target_start} to {min(target_start+9, n-1)}")
        overlap = set(range(i, i + T_CONTEXT)) & set(range(target_start, min(target_start + 10, n)))
        print(f"    Overlap with input: {sorted(overlap) if overlap else 'NONE'}")
        print(f"    LEAKAGE ELIMINATED: {len(overlap) == 0}")
    print()

    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("  1. Replace center=True smoothing with forward-looking smoothing")
    print("     to eliminate target leakage from input frames.")
    print("  2. Frame gaps exist in many sequences — current sliding window")
    print("     builder ignores gaps (treats non-consecutive frames as adjacent).")
    print("     Consider splitting sequences at gaps or filtering.")
    print("  3. Stride=10 reduces windows by ~90% — useful for faster iteration")
    print("     but may lose temporal resolution.")


if __name__ == "__main__":
    main()
