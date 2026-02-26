#!/usr/bin/env python3
"""Evaluate YOLO11 Nano@320 vs Medium@640 on METEOR dashcam clips.

Runs both models on every subsampled frame (every 3rd = ~10 fps),
matches detections to GT using IoU >= 0.5, and records per-frame
tp, fp, fn_coco, fn_total.

Output: meteor_validation/meteor_detection_results.csv
Crash-safe: appends per clip, resumes by skipping completed clips.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from meteor_common import (
    CONF,
    ECHO_ROOT,
    IOU_THRESH,
    METEOR_TO_COCO,
    SCRIPT_DIR,
    SUBSAMPLE,
    VEHICLE_COCO,
    discover_clips,
    get_video_frame_count,
    greedy_match,
    parse_clip_annotations,
    read_video_frames,
)

# ── Paths ──────────────────────────────────────────────────────────
OUT_CSV = SCRIPT_DIR / "meteor_detection_results.csv"

# ── Evaluation modes ───────────────────────────────────────────────
MODES = [
    {"model_file": "yolo11n.pt", "imgsz": 320, "label": "nano_320"},
    {"model_file": "yolo11m.pt", "imgsz": 640, "label": "medium_640"},
]

# ── CSV column order ───────────────────────────────────────────────
COL_ORDER = [
    "clip", "frame_id", "model", "imgsz",
    "tp", "fp", "fn_coco", "fn_total",
    "recall", "precision", "f1", "mean_iou",
    "num_gt", "num_gt_coco",
    "weather", "road_type",
]


# ── Per-frame evaluation ──────────────────────────────────────────

def evaluate_frame(model, frame_bgr, gt_objects, model_label, imgsz):
    """Run one YOLO model on one frame, match to GT.

    Args:
        model: loaded YOLO model
        frame_bgr: numpy array (H, W, 3) BGR
        gt_objects: list of dicts from parse_clip_annotations
        model_label: "nano_320" or "medium_640"
        imgsz: inference resolution

    Returns:
        dict with detection metrics
    """
    # Run YOLO
    results = model.predict(frame_bgr, conf=CONF, iou=IOU_THRESH,
                            imgsz=imgsz, verbose=False)
    det = results[0].boxes
    boxes = det.xyxy.cpu().numpy()
    cls = det.cls.cpu().numpy().astype(int)

    # Filter predictions to VEHICLE_COCO classes
    mask = np.isin(cls, list(VEHICLE_COCO))
    pred_boxes = boxes[mask]

    # Split GT into COCO-matchable and all
    gt_coco_boxes = []
    gt_all_boxes = []
    for obj in gt_objects:
        gt_all_boxes.append(obj["box"])
        if obj["coco_id"] is not None and obj["coco_id"] in VEHICLE_COCO:
            gt_coco_boxes.append(obj["box"])

    gt_coco_arr = (
        np.array(gt_coco_boxes, dtype=np.float32)
        if gt_coco_boxes
        else np.empty((0, 4), dtype=np.float32)
    )
    gt_all_arr = (
        np.array(gt_all_boxes, dtype=np.float32)
        if gt_all_boxes
        else np.empty((0, 4), dtype=np.float32)
    )

    # Match predictions to COCO-matchable GT only
    tp, fp, fn_coco, mean_iou = greedy_match(pred_boxes, gt_coco_arr)

    # fn_total includes unmatchable classes (auto-rickshaw etc.)
    fn_total = len(gt_all_arr) - tp

    num_gt = len(gt_all_arr)
    num_gt_coco = len(gt_coco_arr)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn_coco) if (tp + fn_coco) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "model": model_label,
        "imgsz": imgsz,
        "tp": tp,
        "fp": fp,
        "fn_coco": fn_coco,
        "fn_total": fn_total,
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "num_gt": num_gt,
        "num_gt_coco": num_gt_coco,
    }


# ── Final summary ─────────────────────────────────────────────────

def final_summary():
    """Print summary statistics from the completed CSV."""
    df = pd.read_csv(OUT_CSV)
    print(f"\nCSV: {OUT_CSV}", flush=True)
    print(f"Total rows:   {len(df)}", flush=True)
    print(f"Total clips:  {df['clip'].nunique()}", flush=True)
    total_frames = df.groupby(["clip", "frame_id"]).ngroups
    print(f"Total frames: {total_frames}", flush=True)

    print("\n=== Mean Recall per Mode ===", flush=True)
    recall = df.groupby("model")["recall"].mean().round(4)
    print(recall.to_string(), flush=True)

    if "nano_320" in recall.index and "medium_640" in recall.index:
        gap = recall["medium_640"] - recall["nano_320"]
        print(f"\nMean recall gap (medium - nano): {gap:.4f}", flush=True)

    print("\n=== Per-Weather Breakdown ===", flush=True)
    weather_recall = df.pivot_table(
        index="weather", columns="model", values="recall", aggfunc="mean"
    ).round(4)
    if "nano_320" in weather_recall.columns and "medium_640" in weather_recall.columns:
        weather_recall["gap"] = (
            weather_recall["medium_640"] - weather_recall["nano_320"]
        ).round(4)
    print(weather_recall.to_string(), flush=True)

    print("\n=== Per Road-Type Breakdown ===", flush=True)
    road_recall = df.pivot_table(
        index="road_type", columns="model", values="recall", aggfunc="mean"
    ).round(4)
    if "nano_320" in road_recall.columns and "medium_640" in road_recall.columns:
        road_recall["gap"] = (
            road_recall["medium_640"] - road_recall["nano_320"]
        ).round(4)
    print(road_recall.to_string(), flush=True)

    # fn statistics
    nano = df[df["model"] == "nano_320"]
    medium = df[df["model"] == "medium_640"]
    print(f"\n=== Miss Count (fn_coco) ===", flush=True)
    print(f"  Nano mean:   {nano['fn_coco'].mean():.2f}", flush=True)
    print(f"  Medium mean: {medium['fn_coco'].mean():.2f}", flush=True)
    print(f"\n=== Miss Count (fn_total, incl. auto-rickshaw) ===", flush=True)
    print(f"  Nano mean:   {nano['fn_total'].mean():.2f}", flush=True)
    print(f"  Medium mean: {medium['fn_total'].mean():.2f}", flush=True)


# ── Main ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Discover clips ──────────────────────────────────────────────
    clips = discover_clips(max_clips=50)
    if not clips:
        return

    n_clips = len(clips)

    # ── Crash-safe resume ───────────────────────────────────────────
    completed_clips = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        # A clip is complete only if BOTH modes are present
        completed_counts = existing.groupby("clip")["model"].nunique()
        completed_clips = set(
            completed_counts[completed_counts == len(MODES)].index
        )
        print(
            f"Resuming: {len(completed_clips)} clips already completed\n",
            flush=True,
        )

    # ── Load models ─────────────────────────────────────────────────
    print("Loading YOLO models...", flush=True)
    models = {}
    for mode in MODES:
        model_path = ECHO_ROOT / mode["model_file"]
        if not model_path.exists():
            print(f"ERROR: model not found: {model_path}", flush=True)
            return
        models[mode["label"]] = YOLO(str(model_path))
    print("Models loaded.\n", flush=True)

    total_frames_processed = 0

    for clip_idx, clip_info in enumerate(clips):
        clip_name = clip_info["name"]

        if clip_name in completed_clips:
            print(
                f"[{clip_idx + 1}/{n_clips}] {clip_name}: already complete, skipping",
                flush=True,
            )
            continue

        clip_t0 = time.time()

        # Parse annotations
        gt_by_frame = parse_clip_annotations(clip_info["dir"])
        if not gt_by_frame:
            print(
                f"[{clip_idx + 1}/{n_clips}] {clip_name}: no annotations found, skipping",
                flush=True,
            )
            continue

        weather = clip_info["weather"]
        road_type = clip_info["road_type"]
        video_path = clip_info["video"]

        if video_path is None or not video_path.exists():
            print(
                f"[{clip_idx + 1}/{n_clips}] {clip_name}: no video found, skipping",
                flush=True,
            )
            continue

        total_video_frames = get_video_frame_count(video_path)
        expected_sampled = total_video_frames // SUBSAMPLE

        print(
            f"[{clip_idx + 1}/{n_clips}] {clip_name}: "
            f"~{expected_sampled} sampled frames, weather={weather}, "
            f"road={road_type}, GT frames={len(gt_by_frame)}",
            flush=True,
        )

        clip_records = []
        frames_done = 0

        for frame_id, frame_bgr in read_video_frames(video_path, SUBSAMPLE):
            # Get GT for this frame (or empty if not annotated)
            gt_objects = gt_by_frame.get(frame_id, [])

            frames_done += 1

            for mode in MODES:
                model_label = mode["label"]
                imgsz = mode["imgsz"]
                model = models[model_label]

                rec = evaluate_frame(
                    model, frame_bgr, gt_objects, model_label, imgsz,
                )
                rec["clip"] = clip_name
                rec["frame_id"] = frame_id
                rec["weather"] = weather
                rec["road_type"] = road_type
                clip_records.append(rec)

        total_frames_processed += frames_done

        # Crash-safe: append this clip immediately
        if not clip_records:
            print("    skipped (0 evaluable frames)", flush=True)
            continue

        clip_df = pd.DataFrame(clip_records)[COL_ORDER]
        write_header = not OUT_CSV.exists()
        clip_df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)

        clip_elapsed = time.time() - clip_t0
        total_elapsed = time.time() - t0
        fps = frames_done / clip_elapsed if clip_elapsed > 0 else 0
        print(
            f"    done ({frames_done} frames x {len(MODES)} modes, "
            f"{fps:.1f} fps) [{clip_elapsed:.0f}s clip, {total_elapsed:.0f}s total]",
            flush=True,
        )

    # ── Final summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}", flush=True)
    print(f"Evaluation complete. Total time: {time.time() - t0:.1f}s\n",
          flush=True)

    if OUT_CSV.exists():
        final_summary()


if __name__ == "__main__":
    main()
