"""Benchmark YOLO11-nano@640 on 10 diverse UA-DETRAC training sequences.

Part 1: Run nano@640, compute per-frame recall/precision, compare against
        medium@640 and nano@320 from res_experiment/detrac_results.csv.
Part 2: Failure analysis — spatial features for every false negative.
Part 3: 6-page PDF visualization (spatial_experiments_analysis.pdf).
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # echo/

sys.path.insert(0, str(ROOT / "res_experiment"))
from res_evaluate_detrac import (
    parse_detrac_xml,
    iou_matrix,
    suppress_fp_count,
    VEHICLE_COCO, CONF, IOU_THRESH,
    IMG_BASE, TRAIN_XML,
)

# Output paths
OUT_CSV = SCRIPT_DIR / "spatial_experiments_nano640_results.csv"
MISSES_CSV = SCRIPT_DIR / "spatial_experiments_misses.csv"
OUT_PDF = SCRIPT_DIR / "spatial_experiments_analysis.pdf"
DETRAC_CSV = ROOT / "res_experiment" / "detrac_results.csv"

# Frame resolution (DETRAC standard)
FRAME_W, FRAME_H = 960, 540

# 15 val sequences to exclude
VAL_SEQUENCES = [
    "MVI_39361", "MVI_20011", "MVI_41073", "MVI_39211",
    "MVI_40172", "MVI_40191", "MVI_40712", "MVI_40244",
    "MVI_40992", "MVI_40732", "MVI_39811", "MVI_40775",
    "MVI_63521", "MVI_40892", "MVI_63553",
]

SAMPLES_PER_WEATHER = {"sunny": 3, "cloudy": 3, "night": 2, "rainy": 2}


# ── Sequence selection ─────────────────────────────────────────────
def select_sequences():
    """Randomly sample 10 training sequences (2-3 per weather), seed=42."""
    df = pd.read_csv(DETRAC_CSV)
    seq_weather = (
        df[["sequence", "weather"]]
        .drop_duplicates()
        .set_index("sequence")["weather"]
    )
    val_set = set(VAL_SEQUENCES)
    # Only consider sequences whose XML is in the TRAIN directory
    train_xml_seqs = {p.stem for p in TRAIN_XML.glob("*.xml")}
    train_by_weather = {}
    for seq, w in seq_weather.items():
        if seq in val_set or seq not in train_xml_seqs:
            continue
        train_by_weather.setdefault(w, []).append(seq)

    random.seed(42)
    selected = []
    for weather in ["sunny", "cloudy", "night", "rainy"]:
        pool = sorted(train_by_weather[weather])
        n = SAMPLES_PER_WEATHER[weather]
        chosen = random.sample(pool, n)
        selected.extend(chosen)
        print(f"  {weather}: {chosen}", flush=True)
    return selected


# ── Core matching (detailed variant) ──────────────────────────────
def greedy_match_detailed(pred_boxes, gt_boxes):
    """Greedy IoU matching that returns matched/unmatched GT indices.

    Returns (tp, fp, fn, mean_iou, matched_gt_set, unmatched_gt_list).
    """
    M, N = len(pred_boxes), len(gt_boxes)
    if M == 0 and N == 0:
        return 0, 0, 0, 0.0, set(), []
    if M == 0:
        return 0, 0, N, 0.0, set(), list(range(N))
    if N == 0:
        return 0, M, 0, 0.0, set(), []

    ious = iou_matrix(pred_boxes, gt_boxes)
    flat = np.argsort(ious.ravel())[::-1]
    matched_ious, used_p, used_g = [], set(), set()
    for idx in flat:
        i, j = divmod(int(idx), N)
        v = ious[i, j]
        if v < IOU_THRESH:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched_ious.append(v)

    tp = len(matched_ious)
    unmatched_gt = [j for j in range(N) if j not in used_g]
    miou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return tp, M - tp, N - tp, miou, used_g, unmatched_gt


# ── Spatial features ──────────────────────────────────────────────
def compute_spatial_features(box):
    """Compute spatial features for a GT box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cx_norm = cx / FRAME_W
    cy_norm = cy / FRAME_H

    quadrant = ("top-" if cy_norm < 0.5 else "bottom-") + (
        "left" if cx_norm < 0.5 else "right"
    )
    grid_col = min(int(cx_norm * 3), 2)
    grid_row = min(int(cy_norm * 3), 2)

    dist_center = np.sqrt(
        (cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2
    ) / np.sqrt(0.5)  # normalize so corner = 1.0

    min_edge = min(cx_norm, 1 - cx_norm, cy_norm, 1 - cy_norm)

    return {
        "gt_box_area": round(area / (FRAME_W * FRAME_H), 6),
        "gt_box_width": round(w, 2),
        "gt_box_height": round(h, 2),
        "gt_center_x": round(cx_norm, 4),
        "gt_center_y": round(cy_norm, 4),
        "gt_aspect_ratio": round(w / max(h, 1e-6), 4),
        "quadrant": quadrant,
        "grid_cell": f"r{grid_row}c{grid_col}",
        "grid_row": grid_row,
        "grid_col": grid_col,
        "distance_from_center": round(dist_center, 4),
        "is_edge": min_edge < 0.10,
    }


# ── Per-frame evaluation with failure detail ─────────────────────
def evaluate_frame_detailed(model, img_path, gt_for_frame, ignore_boxes):
    """Run nano@640 on one frame.

    Returns (metrics_dict, miss_records_list, detected_areas_list).
    """
    t0 = time.perf_counter()
    results = model.predict(str(img_path), conf=CONF, iou=IOU_THRESH,
                            imgsz=640, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    det = results[0].boxes
    boxes = det.xyxy.cpu().numpy()
    cls = det.cls.cpu().numpy().astype(int)
    mask = np.isin(cls, list(VEHICLE_COCO))
    pred_boxes = boxes[mask]

    gt_boxes = (
        np.array([t["box"] for t in gt_for_frame], dtype=np.float32)
        if gt_for_frame
        else np.empty((0, 4), dtype=np.float32)
    )

    tp, fp, fn, miou, matched_gt, unmatched_gt = greedy_match_detailed(
        pred_boxes, gt_boxes
    )

    # FP suppression
    if fp > 0 and len(ignore_boxes) > 0:
        suppressed = suppress_fp_count(pred_boxes, gt_boxes, ignore_boxes)
        fp = max(0, fp - suppressed)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    metrics = {
        "tp": tp, "fp": fp, "fn": fn,
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "mean_iou": round(miou, 4),
        "num_gt": len(gt_for_frame),
        "num_detections": int(len(pred_boxes)),
        "latency_ms": round(latency_ms, 2),
    }

    # Spatial features for missed GT boxes
    miss_records = []
    for gt_idx in unmatched_gt:
        gt_entry = gt_for_frame[gt_idx]
        spatial = compute_spatial_features(gt_entry["box"])
        spatial["occ_status"] = gt_entry["occ_status"]
        spatial["truncation_ratio"] = gt_entry["truncation_ratio"]
        miss_records.append(spatial)

    # Areas of detected GT boxes (for Page 3 histogram)
    detected_areas = []
    for gt_idx in matched_gt:
        gt_entry = gt_for_frame[gt_idx]
        b = gt_entry["box"]
        area = (b[2] - b[0]) * (b[3] - b[1]) / (FRAME_W * FRAME_H)
        detected_areas.append(area)

    return metrics, miss_records, detected_areas


# ── Part 1: Benchmark ─────────────────────────────────────────────
def run_benchmark():
    """Run nano@640 on 10 sequences. Returns DataFrames + auxiliary data."""
    print("\n=== Selecting 10 benchmark sequences ===", flush=True)
    benchmark_seqs = select_sequences()
    print(f"\nSelected: {benchmark_seqs}\n", flush=True)

    # Crash-safe resume
    completed_seqs = set()
    existing_frame_records = []
    existing_miss_records = []
    if OUT_CSV.exists():
        existing_df = pd.read_csv(OUT_CSV)
        completed_seqs = set(existing_df["sequence"].unique())
        existing_frame_records = existing_df.to_dict("records")
        print(f"Resuming: {len(completed_seqs)} sequences already done", flush=True)
    if MISSES_CSV.exists():
        existing_miss_df = pd.read_csv(MISSES_CSV)
        existing_miss_records = existing_miss_df.to_dict("records")

    print("Loading YOLO11n...", flush=True)
    model = YOLO(str(ROOT / "yolo11n.pt"))

    # Warmup inference
    dummy = np.zeros((540, 960, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=640, verbose=False)
    print("Model loaded + warmed up.\n", flush=True)

    all_frame_records = list(existing_frame_records)
    all_miss_records = list(existing_miss_records)
    all_detected_areas = []
    total_gt_grid = np.zeros((3, 3))

    n_seqs = len(benchmark_seqs)
    for seq_idx, seq_name in enumerate(benchmark_seqs):
        if seq_name in completed_seqs:
            print(f"[{seq_idx+1}/{n_seqs}] {seq_name}: already done, skipping",
                  flush=True)
            continue

        xml_path = TRAIN_XML / f"{seq_name}.xml"
        img_dir = IMG_BASE / seq_name
        weather, ignore_boxes, gt_by_frame = parse_detrac_xml(xml_path)
        annotated_frames = sorted(gt_by_frame.keys())

        print(f"[{seq_idx+1}/{n_seqs}] {seq_name} ({weather}): "
              f"{len(annotated_frames)} frames", flush=True)

        seq_t0 = time.time()
        seq_frame_records = []
        seq_miss_records = []

        for frame_num in annotated_frames:
            img_path = img_dir / f"img{frame_num:05d}.jpg"
            if not img_path.exists():
                continue

            gt_for_frame = gt_by_frame[frame_num]

            # Accumulate total GT per grid cell (for miss rate)
            for gt_entry in gt_for_frame:
                sp = compute_spatial_features(gt_entry["box"])
                total_gt_grid[sp["grid_row"], sp["grid_col"]] += 1

            metrics, misses, det_areas = evaluate_frame_detailed(
                model, img_path, gt_for_frame, ignore_boxes
            )
            metrics["sequence"] = seq_name
            metrics["frame_id"] = frame_num
            metrics["weather"] = weather
            metrics["model"] = "nano_640"
            metrics["imgsz"] = 640
            seq_frame_records.append(metrics)

            for m in misses:
                m["sequence"] = seq_name
                m["frame_id"] = frame_num
                m["weather"] = weather
            seq_miss_records.extend(misses)

            all_detected_areas.extend(det_areas)

        # Append this sequence to CSVs immediately (crash-safe)
        if seq_frame_records:
            seq_df = pd.DataFrame(seq_frame_records)
            write_header = not OUT_CSV.exists()
            seq_df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)

            if seq_miss_records:
                miss_header = not MISSES_CSV.exists()
                pd.DataFrame(seq_miss_records).to_csv(
                    MISSES_CSV, mode="a", header=miss_header, index=False
                )

        all_frame_records.extend(seq_frame_records)
        all_miss_records.extend(seq_miss_records)

        elapsed = time.time() - seq_t0
        avg_lat = np.mean([r["latency_ms"] for r in seq_frame_records]) if seq_frame_records else 0
        n_miss = len(seq_miss_records)
        print(f"    {len(seq_frame_records)} frames, {n_miss} misses, "
              f"avg latency={avg_lat:.1f}ms, elapsed={elapsed:.1f}s", flush=True)

    frame_df = pd.DataFrame(all_frame_records)
    miss_df = pd.DataFrame(all_miss_records) if all_miss_records else pd.DataFrame()

    print(f"\nSaved {len(frame_df)} frame results -> {OUT_CSV}", flush=True)
    print(f"Saved {len(miss_df)} miss records  -> {MISSES_CSV}", flush=True)

    return frame_df, miss_df, np.array(all_detected_areas), total_gt_grid


# ── Part 2: Comparison summary ────────────────────────────────────
def print_comparison(frame_df):
    """Print three-way recall comparison. Returns comparison DataFrame."""
    seqs = frame_df["sequence"].unique().tolist()
    detrac = pd.read_csv(DETRAC_CSV)

    # Nano@640 per-sequence
    nano640 = frame_df.groupby("sequence").agg(
        weather=("weather", "first"),
        frames=("frame_id", "count"),
        nano640_recall=("recall", "mean"),
        nano640_precision=("precision", "mean"),
        avg_latency=("latency_ms", "mean"),
    ).round(4)

    # Medium@640 from existing CSV
    med = detrac[(detrac["model"] == "medium_640") & (detrac["sequence"].isin(seqs))]
    med640 = med.groupby("sequence").agg(
        med640_recall=("recall", "mean"),
    ).round(4)

    # Nano@320 from existing CSV
    n320 = detrac[(detrac["model"] == "nano_320") & (detrac["sequence"].isin(seqs))]
    nano320 = n320.groupby("sequence").agg(
        nano320_recall=("recall", "mean"),
    ).round(4)

    merged = nano640.join(med640).join(nano320)
    merged["gap_n640_m640"] = (merged["med640_recall"] - merged["nano640_recall"]).round(4)
    merged["improvement_over_320"] = (merged["nano640_recall"] - merged["nano320_recall"]).round(4)

    print("\n" + "=" * 70, flush=True)
    print("Per-Sequence Comparison", flush=True)
    print("=" * 70, flush=True)
    print(merged.to_string(), flush=True)

    mean_n320 = merged["nano320_recall"].mean()
    mean_n640 = merged["nano640_recall"].mean()
    mean_m640 = merged["med640_recall"].mean()

    print(f"\n=== Overall Means ===", flush=True)
    print(f"  nano@320 recall:   {mean_n320:.4f}", flush=True)
    print(f"  nano@640 recall:   {mean_n640:.4f}", flush=True)
    print(f"  medium@640 recall: {mean_m640:.4f}", flush=True)
    print(f"  nano@320 vs medium@640 gap: {mean_m640 - mean_n320:.4f}", flush=True)
    print(f"  nano@640 vs medium@640 gap: {mean_m640 - mean_n640:.4f}", flush=True)
    gap_640 = mean_m640 - mean_n640
    print(f"  Is @640 gap > 0.10? {'YES' if gap_640 > 0.10 else 'NO'} ({gap_640:.4f})",
          flush=True)
    print(f"  Avg latency nano@640: {merged['avg_latency'].mean():.1f}ms", flush=True)

    return merged


# ── Part 3: 6-page PDF ────────────────────────────────────────────
def generate_pdf(frame_df, miss_df, comparison_df, detected_areas, total_gt_grid):
    """Generate spatial_experiments_analysis.pdf (6 pages)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(str(OUT_PDF)) as pdf:
        # ── PAGE 1: Summary table ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis("off")

        col_labels = [
            "Sequence", "Weather", "Frames",
            "Nano@320\nRecall", "Nano@640\nRecall", "Med@640\nRecall",
            "N640-N320\nImprove", "N640-M640\nGap", "Avg Lat\n(ms)",
        ]
        table_data = []
        for seq in comparison_df.index:
            r = comparison_df.loc[seq]
            table_data.append([
                seq, r["weather"], int(r["frames"]),
                f"{r['nano320_recall']:.4f}",
                f"{r['nano640_recall']:.4f}",
                f"{r['med640_recall']:.4f}",
                f"{r['improvement_over_320']:+.4f}",
                f"{r['gap_n640_m640']:.4f}",
                f"{r['avg_latency']:.1f}",
            ])

        # Overall means row
        m = comparison_df.mean(numeric_only=True)
        zero_miss_frac = (frame_df["fn"] == 0).mean()
        table_data.append([
            "OVERALL", "", int(m["frames"] * len(comparison_df)),
            f"{m['nano320_recall']:.4f}",
            f"{m['nano640_recall']:.4f}",
            f"{m['med640_recall']:.4f}",
            f"{m['improvement_over_320']:+.4f}",
            f"{m['gap_n640_m640']:.4f}",
            f"{m['avg_latency']:.1f}",
        ])

        tbl = ax.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.6)

        # Style header row
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#4472C4")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        # Style overall row
        last_row = len(table_data)
        for j in range(len(col_labels)):
            tbl[last_row, j].set_facecolor("#D6E4F0")
            tbl[last_row, j].set_text_props(fontweight="bold")

        fig.suptitle(
            f"Page 1: Nano@640 Benchmark Summary  |  "
            f"Frames with 0 misses: {zero_miss_frac:.1%}",
            fontsize=13, fontweight="bold", y=0.95,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 2: Missed object heatmap ─────────────────────────
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_facecolor("black")

        if len(miss_df) > 0:
            cx_px = miss_df["gt_center_x"] * FRAME_W
            cy_px = miss_df["gt_center_y"] * FRAME_H
            areas = miss_df["gt_box_area"].values * (FRAME_W * FRAME_H)
            sc = ax.scatter(
                cx_px, cy_px,
                c=np.log10(np.clip(areas, 1, None)),
                cmap="plasma", alpha=0.3, s=8, edgecolors="none",
            )
            plt.colorbar(sc, ax=ax, label="log10(area in pixels)")

        # 3x3 grid lines
        for i in range(1, 3):
            ax.axvline(FRAME_W * i / 3, color="white", ls="--", lw=0.5, alpha=0.5)
            ax.axhline(FRAME_H * i / 3, color="white", ls="--", lw=0.5, alpha=0.5)

        ax.set_xlim(0, FRAME_W)
        ax.set_ylim(FRAME_H, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        fig.suptitle(
            f"Page 2: Spatial Distribution of Missed Objects  "
            f"(n={len(miss_df):,})",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 3: Size distribution ─────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if len(miss_df) > 0:
            missed_areas = miss_df["gt_box_area"].values * (FRAME_W * FRAME_H)
            detected_areas_px = detected_areas * (FRAME_W * FRAME_H)

            bins = np.linspace(0, np.log10(max(
                missed_areas.max(), detected_areas_px.max() if len(detected_areas_px) > 0 else 1
            ) + 1), 60)

            ax1.hist(np.log10(np.clip(missed_areas, 1, None)), bins=bins,
                     alpha=0.6, label=f"Missed (n={len(missed_areas):,})",
                     color="red", density=True)
            if len(detected_areas_px) > 0:
                ax1.hist(np.log10(np.clip(detected_areas_px, 1, None)), bins=bins,
                         alpha=0.6, label=f"Detected (n={len(detected_areas_px):,})",
                         color="steelblue", density=True)
            ax1.set_xlabel("log10(area in pixels)")
            ax1.set_ylabel("Density")
            ax1.legend()
            ax1.set_title("Missed vs Detected Object Size")

            # CDF of missed areas
            sorted_areas = np.sort(missed_areas)
            cdf = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
            ax2.plot(sorted_areas, cdf, color="red", lw=2)
            ax2.set_xlabel("Area (pixels)")
            ax2.set_ylabel("CDF")
            ax2.set_title("CDF of Missed Object Areas")
            ax2.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
            median_area = sorted_areas[len(sorted_areas) // 2]
            ax2.axvline(median_area, color="gray", ls="--", lw=0.8, alpha=0.5)
            ax2.text(median_area, 0.52, f"  median={median_area:.0f}px",
                     fontsize=9, color="gray")

        fig.suptitle("Page 3: Size Distribution of Missed vs Detected Objects",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 4: 3x3 grid miss rate ───────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        miss_grid = np.zeros((3, 3))
        if len(miss_df) > 0:
            for _, row in miss_df.iterrows():
                miss_grid[int(row["grid_row"]), int(row["grid_col"])] += 1

        # Left: miss count
        im1 = ax1.imshow(miss_grid, cmap="Reds", interpolation="nearest")
        for i in range(3):
            for j in range(3):
                ax1.text(j, i, f"{int(miss_grid[i, j]):,}",
                         ha="center", va="center", fontsize=14, fontweight="bold",
                         color="white" if miss_grid[i, j] > miss_grid.max() * 0.6 else "black")
        ax1.set_title("Miss Count per Grid Cell")
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(["Left", "Center", "Right"])
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(["Top", "Mid", "Bottom"])
        plt.colorbar(im1, ax=ax1)

        # Right: miss rate (misses / total GT per cell)
        miss_rate = np.divide(
            miss_grid, total_gt_grid,
            out=np.zeros_like(miss_grid), where=total_gt_grid > 0
        )
        im2 = ax2.imshow(miss_rate, cmap="YlOrRd", interpolation="nearest",
                         vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                rate = miss_rate[i, j]
                total = int(total_gt_grid[i, j])
                ax2.text(j, i, f"{rate:.1%}\n({total:,} GT)",
                         ha="center", va="center", fontsize=11, fontweight="bold",
                         color="white" if rate > 0.5 else "black")
        ax2.set_title("Miss Rate per Grid Cell (misses / total GT)")
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(["Left", "Center", "Right"])
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["Top", "Mid", "Bottom"])
        plt.colorbar(im2, ax=ax2, format="%.0f%%")

        fig.suptitle("Page 4: 3x3 Spatial Grid Analysis",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 5: Per-weather breakdown ─────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        weathers = ["sunny", "cloudy", "night", "rainy"]

        # Load existing data for nano@320 and medium@640 per weather
        detrac = pd.read_csv(DETRAC_CSV)
        seqs = frame_df["sequence"].unique().tolist()

        recall_by_weather = {}
        for w in weathers:
            w_seqs = frame_df[frame_df["weather"] == w]["sequence"].unique()
            if len(w_seqs) == 0:
                recall_by_weather[w] = {"nano320": 0, "nano640": 0, "med640": 0}
                continue
            n320_r = detrac[
                (detrac["model"] == "nano_320") & (detrac["sequence"].isin(w_seqs))
            ]["recall"].mean()
            n640_r = frame_df[frame_df["weather"] == w]["recall"].mean()
            m640_r = detrac[
                (detrac["model"] == "medium_640") & (detrac["sequence"].isin(w_seqs))
            ]["recall"].mean()
            recall_by_weather[w] = {
                "nano320": n320_r, "nano640": n640_r, "med640": m640_r,
            }

        x = np.arange(len(weathers))
        width = 0.25
        bars_320 = [recall_by_weather[w]["nano320"] for w in weathers]
        bars_640 = [recall_by_weather[w]["nano640"] for w in weathers]
        bars_med = [recall_by_weather[w]["med640"] for w in weathers]

        ax1.bar(x - width, bars_320, width, label="Nano@320", color="#FF6B6B", alpha=0.8)
        ax1.bar(x, bars_640, width, label="Nano@640", color="#4ECDC4", alpha=0.8)
        ax1.bar(x + width, bars_med, width, label="Med@640", color="#45B7D1", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(weathers)
        ax1.set_ylabel("Mean Recall")
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.set_title("Recall by Weather Category")

        # Right: per-weather miss area distribution (box plot)
        if len(miss_df) > 0:
            weather_areas = []
            weather_labels = []
            for w in weathers:
                areas_w = miss_df[miss_df["weather"] == w]["gt_box_area"].values
                if len(areas_w) > 0:
                    weather_areas.append(areas_w * (FRAME_W * FRAME_H))
                    weather_labels.append(f"{w}\n(n={len(areas_w):,})")
                else:
                    weather_areas.append(np.array([0]))
                    weather_labels.append(f"{w}\n(n=0)")

            bp = ax2.boxplot(weather_areas, labels=weather_labels, patch_artist=True)
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax2.set_ylabel("Missed Object Area (pixels)")
            ax2.set_title("Missed Object Size by Weather")

        fig.suptitle("Page 5: Per-Weather Breakdown",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 6: Frame-level statistics ────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        fn_vals = frame_df["fn"].values

        # Left: histogram
        max_fn = int(fn_vals.max()) if len(fn_vals) > 0 else 10
        bins = range(0, min(max_fn + 2, 30))
        ax1.hist(fn_vals, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        ax1.set_xlabel("False Negatives per Frame")
        ax1.set_ylabel("Frame Count")
        ax1.set_title("Distribution of Misses per Frame")

        frac_0 = (fn_vals == 0).mean()
        frac_12 = ((fn_vals >= 1) & (fn_vals <= 2)).mean()
        frac_3p = (fn_vals >= 3).mean()
        ax1.text(0.95, 0.95,
                 f"0 misses: {frac_0:.1%}\n"
                 f"1-2 misses: {frac_12:.1%}\n"
                 f"3+ misses: {frac_3p:.1%}",
                 transform=ax1.transAxes, fontsize=10,
                 verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # Right: CDF
        sorted_fn = np.sort(fn_vals)
        cdf = np.arange(1, len(sorted_fn) + 1) / len(sorted_fn)
        ax2.plot(sorted_fn, cdf, color="steelblue", lw=2)
        ax2.set_xlabel("False Negatives per Frame")
        ax2.set_ylabel("CDF")
        ax2.set_title("CDF of Misses per Frame")
        ax2.axhline(0.90, color="red", ls="--", lw=0.8, alpha=0.5, label="90th pctl")
        ax2.axhline(0.50, color="orange", ls="--", lw=0.8, alpha=0.5, label="50th pctl")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Page 6: Frame-Level Miss Statistics",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nPDF saved -> {OUT_PDF}", flush=True)


# ── Main ──────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Part 1: Benchmark nano@640
    frame_df, miss_df, detected_areas, total_gt_grid = run_benchmark()

    # Part 2: Comparison summary
    comparison_df = print_comparison(frame_df)

    # Part 3: PDF visualization
    generate_pdf(frame_df, miss_df, comparison_df, detected_areas, total_gt_grid)

    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
