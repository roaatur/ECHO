"""Evaluate YOLOv11 Nano/Small/Medium on VisDrone VID sequences (class-agnostic, vehicles only).

Scans all three VisDrone-VID splits (train, val, test-dev).
Appends results to CSV after each sequence for crash safety.
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Paths & constants ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
VISDRONE_BASE = Path("/home/aljouhara/Desktop/git-echo/VisDrone-VID dataset")
SPLIT_DIRS = [
    VISDRONE_BASE / "VisDrone2019-VID-train",
    VISDRONE_BASE / "VisDrone2019-VID-val",
    VISDRONE_BASE / "VisDrone2019-VID-test-dev",
]
OUT_CSV = ROOT / "visdrone_classagnostic_results.csv"

MODEL_NAMES = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
CONF = 0.25
IOU_THRESH = 0.5

# COCO vehicle classes to keep from YOLO predictions (same as UAVDT)
VEHICLE_COCO = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# VisDrone vehicle class_ids to keep from GT
# 4=car, 5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus
VISDRONE_VEHICLE_CLASSES = {4, 5, 6, 7, 8, 9}

COL_ORDER = [
    "sequence", "frame_id", "model", "tp", "fp", "fn",
    "recall", "precision", "f1", "mean_iou",
    "weather", "altitude", "camera_view",
    "num_gt", "num_gt_occ1", "num_gt_occ2", "num_gt_occ3", "num_gt_occ4",
    "num_gt_outofview",
]


# ── Helpers ────────────────────────────────────────────────────────
def iou_matrix(a, b):
    """Vectorized IoU between (M,4) and (N,4) boxes -> (M,N)."""
    a = a[:, None, :]
    b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-6)


def greedy_match(pred_boxes, gt_boxes):
    """Class-agnostic greedy matching. Returns (tp, fp, fn, mean_iou_of_tps)."""
    M, N = len(pred_boxes), len(gt_boxes)
    if M == 0 and N == 0:
        return 0, 0, 0, 0.0
    if M == 0:
        return 0, 0, N, 0.0
    if N == 0:
        return 0, M, 0, 0.0

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
        used_p.add(i); used_g.add(j)
        matched_ious.append(v)

    tp = len(matched_ious)
    return tp, M - tp, N - tp, float(np.mean(matched_ious)) if matched_ious else 0.0


# ── VisDrone VID GT parser ─────────────────────────────────────────
def parse_visdrone_vid_gt(ann_path):
    """Parse VisDrone VID annotation file for class-agnostic vehicle evaluation.

    Format: frame_id, target_id, x, y, w, h, score, class_id, truncation, occlusion
      score: 0=ignore, 1=evaluate
      class_id: 0=ignored, 1=pedestrian, ..., 9=bus, 10=motor, 11=others
      occlusion: 0=no, 1=partial, 2=heavy

    Returns:
      gt_boxes:       dict[frame_id -> Nx4 array [x1, y1, x2, y2]]
      occ_info:       dict[frame_id -> array of occlusion values]
      ignore_counts:  dict[frame_id -> int] count of score==0 objects
    """
    gt_frames = defaultdict(list)
    occ_frames = defaultdict(list)
    ignore_counts = defaultdict(int)

    for line in ann_path.read_text().strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 10:
            continue
        frame_id = int(parts[0])
        score = int(parts[6])
        class_id = int(parts[7])
        occlusion = int(parts[9])

        if score == 0:
            ignore_counts[frame_id] += 1
            continue

        if class_id not in VISDRONE_VEHICLE_CLASSES:
            continue

        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        gt_frames[frame_id].append([x, y, x + w, y + h])
        occ_frames[frame_id].append(occlusion)

    gt_boxes = {fid: np.array(boxes) for fid, boxes in gt_frames.items()}
    occ_info = {fid: np.array(occs) for fid, occs in occ_frames.items()}
    return gt_boxes, occ_info, dict(ignore_counts)


# ── Evaluation ─────────────────────────────────────────────────────
def evaluate_frame(model, img_path, gt_boxes, occ_array, num_ignored, model_name):
    """Run model on one frame, return metrics dict."""
    results = model.predict(str(img_path), conf=CONF, iou=IOU_THRESH, verbose=False)
    det = results[0].boxes
    boxes = det.xyxy.cpu().numpy()
    cls = det.cls.cpu().numpy().astype(int)
    mask = np.isin(cls, list(VEHICLE_COCO))
    pred_boxes = boxes[mask]

    tp, fp, fn, miou = greedy_match(pred_boxes, gt_boxes)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    num_gt = len(gt_boxes)
    if num_gt > 0:
        num_gt_occ1 = int((occ_array == 0).sum())  # no occlusion
        num_gt_occ2 = int((occ_array == 2).sum())  # heavy occlusion
        num_gt_occ3 = int((occ_array == 1).sum())  # partial occlusion
    else:
        num_gt_occ1 = num_gt_occ2 = num_gt_occ3 = 0

    return dict(
        tp=tp, fp=fp, fn=fn, model=model_name,
        recall=round(rec, 4), precision=round(prec, 4),
        f1=round(f1, 4), mean_iou=round(miou, 4),
        num_gt=num_gt,
        num_gt_occ1=num_gt_occ1, num_gt_occ2=num_gt_occ2,
        num_gt_occ3=num_gt_occ3, num_gt_occ4=0,
        num_gt_outofview=num_ignored,
    )


# ── Main ───────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Discover sequences across all splits
    sequences = []  # (seq_name, seq_img_dir, ann_path)
    for split_dir in SPLIT_DIRS:
        seq_dir = split_dir / "sequences"
        ann_dir = split_dir / "annotations"
        if not seq_dir.exists():
            print(f"  WARNING: {seq_dir} not found, skipping split", flush=True)
            continue
        for s in sorted(seq_dir.iterdir()):
            if not s.is_dir():
                continue
            ann = ann_dir / (s.name + ".txt")
            if ann.exists():
                sequences.append((s.name, s, ann))
            else:
                print(f"  WARNING: no annotation for {s.name}, skipping", flush=True)

    print(f"Found {len(sequences)} sequences across all splits\n", flush=True)

    # Check for existing output (crash-safe resume)
    completed_seqs = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        completed_seqs = set(existing["sequence"].unique())
        print(f"  Resuming: {len(completed_seqs)} sequences already completed", flush=True)

    # Load models
    model_dir = ROOT.parent
    models = {n: YOLO(str(model_dir / n)) for n in MODEL_NAMES}
    remaining = len(sequences) - len(completed_seqs)
    print(f"Models loaded. Evaluating {remaining} remaining sequences...\n", flush=True)

    for seq_idx, (seq_name, seq_img_dir, ann_path) in enumerate(sequences):
        if seq_name in completed_seqs:
            print(f"  [{seq_idx+1}/{len(sequences)}] {seq_name}: already done, skipping",
                  flush=True)
            continue

        seq_t0 = time.time()

        gt_boxes, occ_info, ignore_counts = parse_visdrone_vid_gt(ann_path)
        annotated_frames = sorted(gt_boxes.keys())
        print(f"  [{seq_idx+1}/{len(sequences)}] {seq_name}: "
              f"{len(annotated_frames)} annotated frames", flush=True)

        seq_records = []
        for fid in annotated_frames:
            img_path = seq_img_dir / f"{fid:07d}.jpg"
            if not img_path.exists():
                continue
            gt_arr = gt_boxes[fid]
            occ_arr = occ_info.get(fid, np.empty(0, dtype=int))
            num_ignored = ignore_counts.get(fid, 0)

            for name, model in models.items():
                rec = evaluate_frame(model, img_path, gt_arr, occ_arr, num_ignored, name)
                rec["sequence"] = seq_name
                rec["frame_id"] = fid
                rec["weather"] = "unknown"
                rec["altitude"] = "unknown"
                rec["camera_view"] = "unknown"
                seq_records.append(rec)

        # Crash-safe: append this sequence's results immediately
        if not seq_records:
            print(f"    skipped (no vehicle GT)", flush=True)
            continue
        seq_df = pd.DataFrame(seq_records)[COL_ORDER]
        write_header = not OUT_CSV.exists()
        seq_df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)

        elapsed = time.time() - seq_t0
        total_elapsed = time.time() - t0
        print(f"    done ({len(annotated_frames)} frames x {len(MODEL_NAMES)} models) "
              f"[{elapsed:.0f}s seq, {total_elapsed:.0f}s total]", flush=True)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluation complete. Total time: {time.time() - t0:.1f}s\n", flush=True)

    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        print(f"CSV: {OUT_CSV}", flush=True)
        print(f"Total rows: {len(df)}", flush=True)
        print(f"Total sequences: {df['sequence'].nunique()}", flush=True)
        print(f"Total unique frames: {df.groupby('sequence')['frame_id'].nunique().sum()}", flush=True)

        summary = df.groupby("model").agg(
            mean_recall=("recall", "mean"),
            mean_precision=("precision", "mean"),
            mean_f1=("f1", "mean"),
            mean_iou=("mean_iou", "mean"),
        ).round(4)
        print("\n=== Per-Model Summary ===", flush=True)
        print(summary.to_string(), flush=True)

        seq_summary = df.groupby("sequence").agg(
            frames=("frame_id", "nunique"),
            mean_recall=("recall", "mean"),
            mean_precision=("precision", "mean"),
            mean_f1=("f1", "mean"),
        ).round(4).sort_values("mean_recall")
        print("\n=== Per-Sequence Summary (sorted by recall, hardest first) ===", flush=True)
        print(seq_summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
