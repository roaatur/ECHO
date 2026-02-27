#!/usr/bin/env python3
"""D²-City Step 1 — YOLO11n@320 vs YOLO11m@640 evaluation.

For each subsampled frame in every validation clip:
  - Load GT from CVAT XML annotation
  - Run YOLO11n (imgsz=320) and YOLO11m (imgsz=640)
  - Filter YOLO detections to COCO_VEHICLE_IDS
  - Greedy IoU match (threshold=0.5) separately for nano and medium
  - Record tp, fp, fn_coco (COCO-matchable GT only), fn_total (all GT),
    recall, precision, num_gt_coco, num_gt_total, recoverable_fn

Output: D2_detection_results.csv
Crash-safe: appends per clip; on restart skips clips already written.
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D2_common import (
    SCRIPT_DIR, CONF, IOU_THRESH, SUBSAMPLE, COCO_VEHICLE_IDS,
    discover_clips, parse_cvat_xml, greedy_match,
)

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed: pip install ultralytics")

# ── Output ───────────────────────────────────────────────────────────────────
OUT_CSV = SCRIPT_DIR / "D2_detection_results.csv"

FIELDNAMES = [
    "clip_id", "frame_idx",
    "num_gt_coco", "num_gt_total",
    "tp_nano",   "fp_nano",   "fn_nano",   "fn_nano_total",
    "tp_medium", "fp_medium", "fn_medium", "fn_medium_total",
    "recoverable_fn",
    "recall_nano",   "precision_nano",
    "recall_medium", "precision_medium",
]

NANO_IMGSZ   = 320
MEDIUM_IMGSZ = 640


# ── YOLO inference ────────────────────────────────────────────────────────────

def run_yolo(model, frame_bgr, imgsz):
    """Run YOLO on a BGR frame.

    Returns list of {x1, y1, x2, y2, coco_id, conf}.
    Only classes in COCO_VEHICLE_IDS are returned.
    """
    results = model.predict(frame_bgr, imgsz=imgsz, conf=CONF,
                            verbose=False, stream=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in COCO_VEHICLE_IDS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "coco_id": cls, "conf": float(box.conf[0])})
    return boxes


# ── Per-clip processing ───────────────────────────────────────────────────────

def process_clip(clip, nano_model, medium_model, writer):
    """Evaluate one clip and write per-sampled-frame rows.

    Returns number of rows written (0 on skip).
    """
    clip_id   = clip["clip_id"]
    vid_path  = clip["video_path"]
    xml_path  = clip["xml_path"]

    # Parse annotations
    frames_coco, frames_total = parse_cvat_xml(xml_path)
    if not frames_coco and not frames_total:
        print(f"  [SKIP] {clip_id}: no annotations found")
        return 0

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        print(f"  [SKIP] {clip_id}: cannot open video {vid_path}")
        return 0

    rows_written = 0
    frame_abs    = 0   # original video frame index

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_abs % SUBSAMPLE != 0:
            frame_abs += 1
            continue

        sampled_idx = frame_abs // SUBSAMPLE

        # --- Ground truth ---
        gt_coco  = frames_coco.get(frame_abs, [])   # COCO-matchable GT
        gt_total = frames_total.get(frame_abs, [])  # all GT

        num_gt_coco  = len(gt_coco)
        num_gt_total = len(gt_total)

        # --- YOLO inference ---
        nano_dets   = run_yolo(nano_model,   bgr, NANO_IMGSZ)
        medium_dets = run_yolo(medium_model, bgr, MEDIUM_IMGSZ)

        # --- Matching against COCO GT only ---
        _, matched_gt_nano   = greedy_match(nano_dets,   gt_coco)
        _, matched_gt_medium = greedy_match(medium_dets, gt_coco)

        tp_nano   = len(matched_gt_nano)
        tp_medium = len(matched_gt_medium)

        fn_nano   = num_gt_coco - tp_nano    # COCO misses
        fn_medium = num_gt_coco - tp_medium

        fp_nano   = max(0, len(nano_dets)   - tp_nano)
        fp_medium = max(0, len(medium_dets) - tp_medium)

        # fn_total: misses including unmapped classes (tricycles, etc.)
        fn_nano_total   = num_gt_total - tp_nano
        fn_medium_total = num_gt_total - tp_medium

        # recoverable_fn: objects nano missed that medium caught (among COCO GT)
        recoverable_fn = max(0, tp_medium - tp_nano)

        recall_nano   = tp_nano   / num_gt_coco if num_gt_coco > 0 else 0.0
        recall_medium = tp_medium / num_gt_coco if num_gt_coco > 0 else 0.0
        prec_nano     = tp_nano   / len(nano_dets)   if nano_dets   else 0.0
        prec_medium   = tp_medium / len(medium_dets) if medium_dets else 0.0

        writer.writerow({
            "clip_id":           clip_id,
            "frame_idx":         sampled_idx,
            "num_gt_coco":       num_gt_coco,
            "num_gt_total":      num_gt_total,
            "tp_nano":           tp_nano,
            "fp_nano":           fp_nano,
            "fn_nano":           fn_nano,
            "fn_nano_total":     fn_nano_total,
            "tp_medium":         tp_medium,
            "fp_medium":         fp_medium,
            "fn_medium":         fn_medium,
            "fn_medium_total":   fn_medium_total,
            "recoverable_fn":    recoverable_fn,
            "recall_nano":       round(recall_nano,   4),
            "precision_nano":    round(prec_nano,     4),
            "recall_medium":     round(recall_medium, 4),
            "precision_medium":  round(prec_medium,   4),
        })
        rows_written += 1
        frame_abs += 1

    cap.release()
    return rows_written


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("D²-CITY EVALUATION: YOLO11n@320 vs YOLO11m@640")
    print("=" * 70)

    clips = discover_clips()
    print(f"\n  Found {len(clips)} validation clips")

    # Crash-safety: skip clips already in the CSV
    done_clips = set()
    if OUT_CSV.exists():
        import pandas as pd
        existing = pd.read_csv(OUT_CSV)
        done_clips = set(existing["clip_id"].unique())
        print(f"  Resuming — {len(done_clips)} clips already done, skipping them")

    todo_clips = [c for c in clips if c["clip_id"] not in done_clips]
    print(f"  Clips to process: {len(todo_clips)}")

    if not todo_clips:
        print("\n  All clips already evaluated. Nothing to do.")
        return

    # Load models once
    print("\n  Loading YOLO11n and YOLO11m ...", flush=True)
    nano_model   = YOLO("yolo11n.pt")
    medium_model = YOLO("yolo11m.pt")

    # Open CSV (append if resuming, create with header if new)
    write_header = not OUT_CSV.exists()
    csv_file = open(OUT_CSV, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    total_frames  = 0
    total_rows    = 0
    clip_times    = []

    try:
        for i, clip in enumerate(todo_clips):
            t0 = time.time()
            print(f"\n  [{i + 1:3d}/{len(todo_clips)}] {clip['clip_id']} ...",
                  end="", flush=True)

            rows = process_clip(clip, nano_model, medium_model, writer)
            csv_file.flush()

            elapsed = time.time() - t0
            clip_times.append(elapsed)
            total_rows   += rows
            total_frames += rows

            if rows > 0:
                fps_eff = rows / elapsed if elapsed > 0 else 0
                avg_t = sum(clip_times) / len(clip_times)
                eta   = avg_t * (len(todo_clips) - i - 1) / 60
                print(f"  {rows} frames in {elapsed:.1f}s "
                      f"({fps_eff:.1f} sampled fps) "
                      f"[ETA {eta:.1f} min]", flush=True)
            else:
                print(f"  SKIPPED", flush=True)

    finally:
        csv_file.close()

    # Summary
    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 70}")

    import pandas as pd
    df = pd.read_csv(OUT_CSV)
    df_all = df                                          # all frames
    nano   = df_all
    medium = df_all

    total_clips = df_all["clip_id"].nunique()
    total_f     = len(df_all)
    mean_gt     = df_all["num_gt_coco"].mean()

    print(f"\n  Clips processed : {total_clips}")
    print(f"  Total frames    : {total_f:,}")
    print(f"  Mean GT / frame : {mean_gt:.2f} (COCO-matchable)")

    mean_recall_nano   = df_all["recall_nano"].mean()
    mean_recall_medium = df_all["recall_medium"].mean()
    recall_gap         = mean_recall_medium - mean_recall_nano

    print(f"\n  Recall nano@320  : {mean_recall_nano:.4f}")
    print(f"  Recall medium@640: {mean_recall_medium:.4f}")
    print(f"  Recall gap       : {recall_gap:.4f}")

    mean_fn_nano      = df_all["fn_nano"].mean()
    mean_fn_medium    = df_all["fn_medium"].mean()
    mean_recoverable  = df_all["recoverable_fn"].mean()

    print(f"\n  Mean fn_nano / frame     : {mean_fn_nano:.3f}")
    print(f"  Mean fn_medium / frame   : {mean_fn_medium:.3f}")
    print(f"  Mean recoverable_fn/frame: {mean_recoverable:.3f}")
    print(f"\n  Output: {OUT_CSV}")


if __name__ == "__main__":
    main()
