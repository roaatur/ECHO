"""Evaluate YOLOv11 Nano/Small/Medium on cherry-picked VisDrone VID sequences."""

import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Paths & constants ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
VID_BASE = ROOT / "VisDrone2019-VID-Cherry-Picked"
OUT_CSV = ROOT / "vid_subset_results.csv"
SAMPLE_DIR = ROOT / "cherry_samples"

SEQ_DIRS = [
    (VID_BASE / "cherry_train_sequences", VID_BASE / "cherry_train_annotations"),
    (VID_BASE / "cherry_val_sequences",   VID_BASE / "cherry_val_annotations"),
]

MODEL_NAMES = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
CONF = 0.25
IOU_THRESH = 0.5

VISDRONE_TO_COCO = {
    1: 0, 2: 0,        # pedestrian, people -> person
    3: 1,               # bicycle
    4: 2, 5: 2,         # car, van -> car
    6: 7,               # truck
    7: 3, 8: 3, 10: 3,  # tricycle, awning-tricycle, motor -> motorcycle
    9: 5,               # bus
}
RELEVANT_COCO = set(VISDRONE_TO_COCO.values())


# ── Helpers (matching logic copied from evaluate_visdrone.py) ──────
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


def greedy_match(pred_boxes, pred_cls, gt_boxes, gt_cls):
    """Class-aware greedy matching. Returns (tp, fp, fn, mean_iou_of_tps)."""
    M, N = len(pred_boxes), len(gt_boxes)
    if M == 0 and N == 0:
        return 0, 0, 0, 0.0
    if M == 0:
        return 0, 0, N, 0.0
    if N == 0:
        return 0, M, 0, 0.0

    ious = iou_matrix(pred_boxes, gt_boxes)
    class_match = pred_cls[:, None] == gt_cls[None, :]
    ious *= class_match

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


# ── VID-specific GT parser ────────────────────────────────────────
def parse_vid_gt(ann_path):
    """Parse VID annotation file. Returns dict[frame_id, Nx5 array]."""
    frames = defaultdict(list)
    for line in ann_path.read_text().strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 10:
            continue
        frame_id = int(parts[0])
        cat = int(parts[7])  # VID: column 7 is category (not 5 like DET)
        if cat not in VISDRONE_TO_COCO:
            continue
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        frames[frame_id].append([x, y, x + w, y + h, VISDRONE_TO_COCO[cat]])
    return {fid: np.array(boxes) for fid, boxes in frames.items()}


# ── Evaluation & visualization (adapted for pre-parsed GT) ────────
def evaluate_frame(model, img_path, gt_array, model_name):
    """Run model on one frame, return metrics dict."""
    results = model.predict(str(img_path), conf=CONF, iou=IOU_THRESH, verbose=False)
    det = results[0].boxes
    boxes = det.xyxy.cpu().numpy()
    cls = det.cls.cpu().numpy().astype(int)
    mask = np.isin(cls, list(RELEVANT_COCO))
    pred_boxes, pred_cls = boxes[mask], cls[mask]

    gt_boxes = gt_array[:, :4] if len(gt_array) else np.empty((0, 4))
    gt_cls = gt_array[:, 4].astype(int) if len(gt_array) else np.empty(0, dtype=int)

    tp, fp, fn, miou = greedy_match(pred_boxes, pred_cls, gt_boxes, gt_cls)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, model=model_name,
                recall=round(rec, 4), precision=round(prec, 4),
                f1=round(f1, 4), mean_iou=round(miou, 4))


def draw_sample_frame(img_path, gt_boxes, models, out_dir, label):
    """Side-by-side Nano (green) vs Medium (blue) with GT (red dashed)."""
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    nano_res = models["yolo11n.pt"].predict(str(img_path), conf=CONF, iou=IOU_THRESH, verbose=False)[0]
    med_res = models["yolo11m.pt"].predict(str(img_path), conf=CONF, iou=IOU_THRESH, verbose=False)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    for ax, res, title, color in [(ax1, nano_res, "Nano (yolo11n)", "lime"),
                                   (ax2, med_res, "Medium (yolo11m)", "deepskyblue")]:
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        for b in gt_boxes:
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                         linewidth=1, edgecolor="red", facecolor="none", linestyle="--"))
        det = res.boxes
        boxes = det.xyxy.cpu().numpy()
        cls = det.cls.cpu().numpy().astype(int)
        confs = det.conf.cpu().numpy()
        for b, c, conf in zip(boxes, cls, confs):
            if c not in RELEVANT_COCO:
                continue
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                         linewidth=2, edgecolor=color, facecolor="none"))
            ax.text(b[0], b[1] - 2, f"{conf:.2f}", color=color, fontsize=6,
                    bbox=dict(facecolor="black", alpha=0.5, pad=0.3))

    plt.tight_layout()
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{label}_compare.jpg", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────
def main():
    # Collect all sequences (train + val combined)
    sequences = []  # list of (seq_name, seq_img_dir, ann_path)
    for seq_dir, ann_dir in SEQ_DIRS:
        for s in sorted(seq_dir.iterdir()):
            if s.is_dir():
                ann = ann_dir / (s.name + ".txt")
                if ann.exists():
                    sequences.append((s.name, s, ann))
    print(f"Found {len(sequences)} sequences")

    models = {n: YOLO(n) for n in MODEL_NAMES}

    records = []
    all_frames = []  # (seq_name, frame_id, img_path, gt_array) for sampling
    total_frames = 0

    for seq_name, seq_img_dir, ann_path in sequences:
        gt_by_frame = parse_vid_gt(ann_path)
        annotated_frames = sorted(gt_by_frame.keys())
        total_frames += len(annotated_frames)
        print(f"  {seq_name}: {len(annotated_frames)} annotated frames")

        for fid in annotated_frames:
            img_path = seq_img_dir / f"{fid:07d}.jpg"
            if not img_path.exists():
                continue
            gt_array = gt_by_frame[fid]
            all_frames.append((seq_name, fid, img_path, gt_array))

            for name, model in models.items():
                rec = evaluate_frame(model, img_path, gt_array, name)
                rec["sequence"] = seq_name
                rec["frame_id"] = fid
                records.append(rec)

        print(f"    done ({len(annotated_frames)} frames x {len(MODEL_NAMES)} models)")

    df = pd.DataFrame(records)
    col_order = ["sequence", "frame_id", "model", "tp", "fp", "fn",
                 "recall", "precision", "f1", "mean_iou"]
    df = df[col_order]
    df.to_csv(OUT_CSV, index=False)
    print(f"\nCSV saved to {OUT_CSV} ({len(df)} rows, {total_frames} frames)")

    # ── Sample visualizations ──
    random.seed(42)
    sample_frames = random.sample(all_frames, min(10, len(all_frames)))
    for seq_name, fid, img_path, gt_array in sample_frames:
        gt_boxes = gt_array[:, :4] if len(gt_array) else np.empty((0, 4))
        draw_sample_frame(img_path, gt_boxes, models, SAMPLE_DIR,
                          f"{seq_name}_f{fid:07d}")
    print(f"Saved {len(sample_frames)} sample images to {SAMPLE_DIR}/")

    # ── Per-model summary ──
    summary = df.groupby("model").agg(
        mean_recall=("recall", "mean"),
        mean_precision=("precision", "mean"),
        mean_f1=("f1", "mean"),
        mean_iou=("mean_iou", "mean"),
    ).round(4)
    print("\n=== Per-Model Summary ===")
    print(summary.to_string())

    # ── Per-sequence summary ──
    seq_summary = df.groupby("sequence").agg(
        frames=("frame_id", "nunique"),
        mean_recall=("recall", "mean"),
        mean_precision=("precision", "mean"),
        mean_f1=("f1", "mean"),
        mean_iou=("mean_iou", "mean"),
    ).round(4).sort_values("mean_recall")
    print("\n=== Per-Sequence Summary (sorted by recall, hardest first) ===")
    print(seq_summary.to_string())


if __name__ == "__main__":
    main()
