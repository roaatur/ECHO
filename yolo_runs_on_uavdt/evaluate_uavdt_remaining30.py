"""Evaluate YOLOv11 Nano/Small/Medium on the 30 remaining UAVDT sequences.

Appends results to uavdt_results.csv (preserving the existing 20 sequences).
"""

import random
import time
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
UAVDT_DIR = ROOT.parent / "UAVDT-NEW"
IMG_BASE = UAVDT_DIR / "UAV-benchmark-M"
GT_DIR = UAVDT_DIR / "UAV-benchmark-MOTD_v1.0" / "GT"
ATTR_DIRS = [UAVDT_DIR / "M_attr" / "train", UAVDT_DIR / "M_attr" / "test"]
OUT_CSV = ROOT / "uavdt_results.csv"
SAMPLE_DIR = ROOT / "uavdt_samples"

MODEL_NAMES = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
CONF = 0.25
IOU_THRESH = 0.5

# COCO vehicle classes to keep from YOLO predictions
VEHICLE_COCO = {2, 3, 5, 7}   # car, motorcycle, bus, truck

# 30 remaining UAVDT sequences (not in the original 20-sequence evaluation)
REMAINING_SEQUENCES = [
    "M0101", "M0201", "M0202", "M0204", "M0206", "M0207", "M0210",
    "M0301",
    "M0401", "M0402",
    "M0501",
    "M0603", "M0604", "M0605",
    "M0702", "M0703", "M0704",
    "M0901", "M0902",
    "M1002", "M1003", "M1005", "M1006", "M1008",
    "M1102",
    "M1201", "M1202",
    "M1304", "M1305", "M1306",
]

# UAVDT object categories we care about
UAVDT_CATEGORIES = {1, 2, 3}  # 1=car, 2=truck, 3=bus


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
    # Class-agnostic: no class_match mask, use raw IoU

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


# ── UAVDT GT parser ───────────────────────────────────────────────
def parse_uavdt_gt(gt_path):
    """Parse gt_whole.txt.

    Format: frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
            out-of-view, occlusion, object_category

    Keeps only object_category in {1,2,3} (car, truck, bus).
    No filtering on occlusion or out-of-view — those are stored for post-analysis.

    Returns dict[frame_id → Nx6 array [x1, y1, x2, y2, occlusion, out_of_view]].
    """
    frames = defaultdict(list)
    for line in gt_path.read_text().strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue
        frame_id = int(parts[0])
        cat = int(parts[8])
        if cat not in UAVDT_CATEGORIES:
            continue
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        out_of_view = int(parts[6])
        occlusion = int(parts[7])
        frames[frame_id].append([x, y, x + w, y + h, occlusion, out_of_view])
    return {fid: np.array(boxes) for fid, boxes in frames.items()}


def parse_ignore_regions(ignore_path):
    """Parse gt_ignore.txt.

    Robustly extracts frame_id and bounding box from each line, regardless of
    exact column semantics (handles both 9-column and other formats).

    Returns dict[frame_id → Nx4 array [x1, y1, x2, y2]].
    """
    if not ignore_path.exists():
        return {}
    frames = defaultdict(list)
    for line in ignore_path.read_text().strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        # Same column layout as gt_whole: frame, id, x, y, w, h, ...
        frame_id = int(parts[0])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        frames[frame_id].append([x, y, x + w, y + h])
    return {fid: np.array(boxes) for fid, boxes in frames.items()}


def parse_attributes(seq_name):
    """Look up attribute file in train/ or test/ dirs.

    Returns (weather, altitude, camera_view) strings.
    """
    attr_file = None
    for d in ATTR_DIRS:
        for name_variant in [f"{seq_name}_attr.txt", f"{seq_name} _attr.txt"]:
            candidate = d / name_variant
            if candidate.exists():
                attr_file = candidate
                break
        if attr_file is not None:
            break
    if attr_file is None:
        return "unknown", "unknown", "unknown"

    bits = [int(x) for x in attr_file.read_text().strip().split(",")]

    # weather: bits 0-2
    if bits[0]:
        weather = "daylight"
    elif bits[1]:
        weather = "night"
    elif bits[2]:
        weather = "fog"
    else:
        weather = "unknown"

    # altitude: bits 3-5
    if bits[3]:
        altitude = "low"
    elif bits[4]:
        altitude = "medium"
    elif bits[5]:
        altitude = "high"
    else:
        altitude = "unknown"

    # camera_view: bits 6-8
    views = []
    if bits[6]:
        views.append("front")
    if bits[7]:
        views.append("side")
    if bits[8]:
        views.append("bird")
    camera_view = "+".join(views) if views else "unknown"

    return weather, altitude, camera_view


# ── Evaluation ─────────────────────────────────────────────────────
def evaluate_frame(model, img_path, gt_array, ignore_boxes, model_name):
    """Run model on one frame, return metrics dict."""
    results = model.predict(str(img_path), conf=CONF, iou=IOU_THRESH, verbose=False)
    det = results[0].boxes
    boxes = det.xyxy.cpu().numpy()
    cls = det.cls.cpu().numpy().astype(int)
    mask = np.isin(cls, list(VEHICLE_COCO))
    pred_boxes = boxes[mask]

    gt_boxes = gt_array[:, :4] if len(gt_array) else np.empty((0, 4))

    tp, fp, fn, miou = greedy_match(pred_boxes, gt_boxes)

    # Suppress FPs that overlap with ignore regions
    if fp > 0 and len(ignore_boxes) > 0:
        # Find which predictions were NOT matched (last fp predictions by index)
        # Re-derive matched set to identify unmatched predictions
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious_pg = iou_matrix(pred_boxes, gt_boxes)
            flat = np.argsort(ious_pg.ravel())[::-1]
            used_p, used_g = set(), set()
            N_gt = len(gt_boxes)
            for idx in flat:
                i, j = divmod(int(idx), N_gt)
                if ious_pg[i, j] < IOU_THRESH:
                    break
                if i in used_p or j in used_g:
                    continue
                used_p.add(i); used_g.add(j)
            unmatched_pred_idx = [i for i in range(len(pred_boxes)) if i not in used_p]
        else:
            unmatched_pred_idx = list(range(len(pred_boxes)))

        if unmatched_pred_idx:
            unmatched_preds = pred_boxes[unmatched_pred_idx]
            ious_ig = iou_matrix(unmatched_preds, ignore_boxes)
            max_iou_with_ignore = ious_ig.max(axis=1)
            suppressed = int((max_iou_with_ignore >= IOU_THRESH).sum())
            fp -= suppressed

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # Occlusion / out-of-view counts for post-analysis
    num_gt = len(gt_array)
    if num_gt > 0:
        occ_col = gt_array[:, 4].astype(int)
        oov_col = gt_array[:, 5].astype(int)
        num_gt_occ1 = int((occ_col == 1).sum())
        num_gt_occ2 = int((occ_col == 2).sum())
        num_gt_occ3 = int((occ_col == 3).sum())
        num_gt_occ4 = int((occ_col == 4).sum())
        num_gt_outofview = int((oov_col > 1).sum())
    else:
        num_gt_occ1 = num_gt_occ2 = num_gt_occ3 = num_gt_occ4 = 0
        num_gt_outofview = 0

    return dict(
        tp=tp, fp=fp, fn=fn, model=model_name,
        recall=round(rec, 4), precision=round(prec, 4),
        f1=round(f1, 4), mean_iou=round(miou, 4),
        num_gt=num_gt,
        num_gt_occ1=num_gt_occ1, num_gt_occ2=num_gt_occ2,
        num_gt_occ3=num_gt_occ3, num_gt_occ4=num_gt_occ4,
        num_gt_outofview=num_gt_outofview,
    )


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
            if c not in VEHICLE_COCO:
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
    t0 = time.time()

    # Parse attributes for all remaining sequences
    seq_attrs = {}
    for seq in REMAINING_SEQUENCES:
        weather, altitude, camera_view = parse_attributes(seq)
        seq_attrs[seq] = (weather, altitude, camera_view)
        print(f"  {seq}: weather={weather}, alt={altitude}, view={camera_view}", flush=True)

    # Load models from repo root
    model_dir = ROOT.parent
    models = {n: YOLO(str(model_dir / n)) for n in MODEL_NAMES}
    print(f"\nModels loaded. Evaluating {len(REMAINING_SEQUENCES)} sequences...\n", flush=True)

    records = []
    all_frames = []  # (seq_name, frame_id, img_path, gt_array) for sampling
    total_frames = 0

    for seq_idx, seq_name in enumerate(REMAINING_SEQUENCES):
        gt_path = GT_DIR / f"{seq_name}_gt_whole.txt"
        ignore_path = GT_DIR / f"{seq_name}_gt_ignore.txt"
        seq_img_dir = IMG_BASE / seq_name

        if not gt_path.exists():
            print(f"  WARNING: GT not found for {seq_name}, skipping", flush=True)
            continue
        if not seq_img_dir.exists():
            print(f"  WARNING: images not found for {seq_name}, skipping", flush=True)
            continue

        gt_by_frame = parse_uavdt_gt(gt_path)
        ignore_by_frame = parse_ignore_regions(ignore_path)
        annotated_frames = sorted(gt_by_frame.keys())
        total_frames += len(annotated_frames)
        weather, altitude, camera_view = seq_attrs[seq_name]
        print(f"  {seq_name}: {len(annotated_frames)} annotated frames "
              f"[{seq_idx+1}/{len(REMAINING_SEQUENCES)}]", flush=True)

        for fid in annotated_frames:
            img_path = seq_img_dir / f"img{fid:06d}.jpg"
            if not img_path.exists():
                continue
            gt_array = gt_by_frame[fid]
            ignore_boxes = ignore_by_frame.get(fid, np.empty((0, 4)))
            all_frames.append((seq_name, fid, img_path, gt_array))

            for name, model in models.items():
                rec = evaluate_frame(model, img_path, gt_array, ignore_boxes, name)
                rec["sequence"] = seq_name
                rec["frame_id"] = fid
                rec["weather"] = weather
                rec["altitude"] = altitude
                rec["camera_view"] = camera_view
                records.append(rec)

        elapsed = time.time() - t0
        print(f"    done ({len(annotated_frames)} frames x {len(MODEL_NAMES)} models) "
              f"[{elapsed:.0f}s elapsed]", flush=True)

    new_df = pd.DataFrame(records)
    col_order = [
        "sequence", "frame_id", "model", "tp", "fp", "fn",
        "recall", "precision", "f1", "mean_iou",
        "weather", "altitude", "camera_view",
        "num_gt", "num_gt_occ1", "num_gt_occ2", "num_gt_occ3", "num_gt_occ4",
        "num_gt_outofview",
    ]
    new_df = new_df[col_order]

    # ── Sample visualizations ──
    random.seed(42)
    sample_frames = random.sample(all_frames, min(10, len(all_frames)))
    for seq_name, fid, img_path, gt_array in sample_frames:
        gt_boxes = gt_array[:, :4] if len(gt_array) else np.empty((0, 4))
        draw_sample_frame(img_path, gt_boxes, models, SAMPLE_DIR,
                          f"{seq_name}_f{fid:06d}")
    print(f"Saved {len(sample_frames)} sample images to {SAMPLE_DIR}/", flush=True)

    # ── Per-model summary (new 30 only) ──
    summary = new_df.groupby("model").agg(
        mean_recall=("recall", "mean"),
        mean_precision=("precision", "mean"),
        mean_f1=("f1", "mean"),
        mean_iou=("mean_iou", "mean"),
    ).round(4)
    print("\n=== Per-Model Summary (new 30 sequences only) ===", flush=True)
    print(summary.to_string(), flush=True)

    # ── Per-sequence summary (new 30 only) ──
    seq_summary = new_df.groupby("sequence").agg(
        frames=("frame_id", "nunique"),
        mean_recall=("recall", "mean"),
        mean_precision=("precision", "mean"),
        mean_f1=("f1", "mean"),
        mean_iou=("mean_iou", "mean"),
    ).round(4).sort_values("mean_recall")
    print("\n=== Per-Sequence Summary (sorted by recall, hardest first) ===", flush=True)
    print(seq_summary.to_string(), flush=True)

    # ── Per-weather-condition summary (new 30 only) ──
    weather_summary = new_df.groupby(["weather", "model"]).agg(
        frames=("frame_id", "nunique"),
        mean_recall=("recall", "mean"),
        mean_precision=("precision", "mean"),
        mean_f1=("f1", "mean"),
        mean_iou=("mean_iou", "mean"),
    ).round(4)
    print("\n=== Per-Weather-Condition Summary ===", flush=True)
    print(weather_summary.to_string(), flush=True)

    # ── Append to existing CSV (preserve the original 20 sequences) ──
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        print(f"\n  Existing CSV: {len(existing)} rows", flush=True)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(OUT_CSV, index=False)
    print(f"  CSV saved to {OUT_CSV} ({len(combined)} total rows, "
          f"{len(new_df)} new rows from {total_frames} frames)", flush=True)

    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
