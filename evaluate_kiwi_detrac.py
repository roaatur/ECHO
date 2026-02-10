"""Evaluate YOLOv11 Nano/Small/Medium on UA-DETRAC training sequences."""

import random
import xml.etree.ElementTree as ET
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
DATASET = Path.home() / "Downloads" / "senior project 2.0" / "dataset"
IMG_BASE = DATASET / "DETRAC-Images" / "DETRAC-Images"
ANN_BASE = DATASET / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"
OUT_CSV = ROOT / "kiwi_results.csv"
SAMPLE_DIR = ROOT / "kiwi_samples"

MODEL_NAMES = [str(ROOT / "yolo11n.pt"), str(ROOT / "yolo11s.pt"), str(ROOT / "yolo11m.pt")]
CONF = 0.25
IOU_THRESH = 0.5

# 10 sample videos from the training set
SELECTED_VIDEOS = [
    "MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034",
    "MVI_20051", "MVI_20061", "MVI_20062", "MVI_20063", "MVI_20065",
]

# DETRAC vehicle_type -> COCO class
DETRAC_TO_COCO = {
    "car":    2,  # COCO car
    "van":    2,  # treat as car
    "bus":    5,  # COCO bus
    "others": 2,  # default to car
}
# YOLO vehicle classes we keep from predictions
RELEVANT_COCO = {2, 3, 5, 7}  # car, motorcycle, bus, truck


# ── Helpers (matching logic from evaluate_cherry_visdrone.py) ──────
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


# ── DETRAC XML GT parser ─────────────────────────────────────────
def parse_detrac_gt(xml_path):
    """Parse DETRAC XML annotation. Returns dict[frame_num, Nx5 array]."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames = defaultdict(list)
    for frame_el in root.iter("frame"):
        fnum = int(frame_el.get("num"))
        for target in frame_el.iter("target"):
            box_el = target.find("box")
            attr_el = target.find("attribute")
            if box_el is None or attr_el is None:
                continue
            vtype = attr_el.get("vehicle_type", "others")
            if vtype not in DETRAC_TO_COCO:
                continue
            left = float(box_el.get("left"))
            top = float(box_el.get("top"))
            w = float(box_el.get("width"))
            h = float(box_el.get("height"))
            coco_cls = DETRAC_TO_COCO[vtype]
            frames[fnum].append([left, top, left + w, top + h, coco_cls])
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
    # Collect selected sequences that have both images and annotations
    sequences = []  # list of (seq_name, seq_img_dir, ann_path)
    for vid_name in SELECTED_VIDEOS:
        seq_img_dir = IMG_BASE / vid_name
        ann_path = ANN_BASE / f"{vid_name}.xml"
        if seq_img_dir.is_dir() and ann_path.exists():
            sequences.append((vid_name, seq_img_dir, ann_path))
        else:
            print(f"  WARNING: skipping {vid_name} (missing images or annotation)")
    print(f"Found {len(sequences)} sequences")

    models = {n: YOLO(n) for n in MODEL_NAMES}

    records = []
    all_frames = []  # (seq_name, frame_num, img_path, gt_array) for sampling
    total_frames = 0

    for seq_name, seq_img_dir, ann_path in sequences:
        gt_by_frame = parse_detrac_gt(ann_path)
        annotated_frames = sorted(gt_by_frame.keys())
        total_frames += len(annotated_frames)
        print(f"  {seq_name}: {len(annotated_frames)} annotated frames")

        for fnum in annotated_frames:
            img_path = seq_img_dir / f"img{fnum:05d}.jpg"
            if not img_path.exists():
                continue
            gt_array = gt_by_frame[fnum]
            all_frames.append((seq_name, fnum, img_path, gt_array))

            for name, model in models.items():
                rec = evaluate_frame(model, img_path, gt_array, name)
                rec["sequence"] = seq_name
                rec["frame_id"] = fnum
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
    for seq_name, fnum, img_path, gt_array in sample_frames:
        gt_boxes = gt_array[:, :4] if len(gt_array) else np.empty((0, 4))
        draw_sample_frame(img_path, gt_boxes, models, SAMPLE_DIR,
                          f"{seq_name}_f{fnum:05d}")
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
