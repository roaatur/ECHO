"""Evaluate YOLOv11 Nano@320 vs Medium@640 on UA-DETRAC (100 sequences).

Dataset: UA-DETRAC — 100 traffic sequences (60 train + 40 test), ~140K frames.
Two evaluation modes:
  - nano_320:   yolo11n.pt at imgsz=320
  - medium_640: yolo11m.pt at imgsz=640

Matching: class-agnostic greedy IoU matching (threshold=0.5), same as UAVDT eval.
FP suppression: unmatched predictions overlapping ignored regions (IoU >= 0.5) are not FPs.

Occlusion mapping (DETRAC -> UAVDT schema):
  - num_gt_occ1: no <occlusion> element or occlusion_status==1 (occluder, not itself occluded)
  - num_gt_occ2: occlusion_status==0 (by vehicle) or -1 (by ignored region)
  - num_gt_occ3: always 0 (DETRAC has no heavy vs light distinction)
  - num_gt_outofview: truncation_ratio > 0

CSV output: res_experiment/detrac_results.csv
Crash-safe: appends per sequence, resumes by skipping completed sequences.
"""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ECHO_ROOT = ROOT.parent  # echo/ — contains .pt files

IMG_BASE = Path("/home/aljouhara/Desktop/Detrac Dataset/DETRAC-Images")
TRAIN_XML = Path(
    "/home/aljouhara/Desktop/Detrac Dataset"
    "/DETRAC-Train-Annotations-XML"
    "/DETRAC-Train-Annotations-XML"
)
TEST_XML = Path(
    "/home/aljouhara/Desktop/Detrac Dataset"
    "/DETRAC-Test-Annotations-XML"
    "/DETRAC-Test-Annotations-XML"
)
OUT_CSV = ROOT / "detrac_results.csv"

# ── Evaluation modes ───────────────────────────────────────────────
MODES = [
    {"model_file": "yolo11n.pt", "imgsz": 320, "label": "nano_320"},
    {"model_file": "yolo11m.pt", "imgsz": 640, "label": "medium_640"},
]

# ── Detection constants ────────────────────────────────────────────
CONF = 0.25
IOU_THRESH = 0.5
VEHICLE_COCO = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# ── CSV column order ───────────────────────────────────────────────
COL_ORDER = [
    "sequence", "frame_id", "model", "imgsz",
    "tp", "fp", "fn", "recall", "precision", "f1", "mean_iou",
    "weather",
    "num_gt", "num_gt_occ1", "num_gt_occ2", "num_gt_occ3", "num_gt_outofview",
]


# ── IoU & matching (verbatim from evaluate_uavdt.py) ──────────────
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
        used_p.add(i)
        used_g.add(j)
        matched_ious.append(v)

    tp = len(matched_ious)
    return tp, M - tp, N - tp, float(np.mean(matched_ious)) if matched_ious else 0.0


# ── FP suppression via ignored regions ─────────────────────────────
def suppress_fp_count(pred_boxes, gt_boxes, ignore_boxes):
    """Count FPs that overlap ignored regions (IoU >= threshold) and should be suppressed."""
    if len(pred_boxes) == 0 or len(ignore_boxes) == 0:
        return 0

    # Re-derive which predictions were matched to GT
    if len(gt_boxes) > 0:
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
            used_p.add(i)
            used_g.add(j)
        unmatched = [i for i in range(len(pred_boxes)) if i not in used_p]
    else:
        unmatched = list(range(len(pred_boxes)))

    if not unmatched:
        return 0

    unmatched_preds = pred_boxes[unmatched]
    ious_ig = iou_matrix(unmatched_preds, ignore_boxes)
    return int((ious_ig.max(axis=1) >= IOU_THRESH).sum())


# ── UA-DETRAC XML parser ──────────────────────────────────────────
def parse_detrac_xml(xml_path):
    """Parse a UA-DETRAC XML annotation file.

    Returns:
        weather      : str — "sunny", "cloudy", "rainy", or "night"
        ignore_boxes : np.ndarray (K, 4) in [x1, y1, x2, y2]
        gt_by_frame  : dict[int -> list of dicts]
                       Each dict: box (4-list), occ_status (int|None), truncation_ratio (float)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Sequence-level weather
    sa = root.find("sequence_attribute")
    weather = sa.attrib.get("sence_weather", "unknown") if sa is not None else "unknown"

    # Ignored regions (sequence-level, same for every frame)
    ignore_list = []
    ir = root.find("ignored_region")
    if ir is not None:
        for box_el in ir.findall("box"):
            a = box_el.attrib
            x1 = float(a["left"])
            y1 = float(a["top"])
            x2 = x1 + float(a["width"])
            y2 = y1 + float(a["height"])
            ignore_list.append([x1, y1, x2, y2])
    ignore_boxes = (
        np.array(ignore_list, dtype=np.float32)
        if ignore_list
        else np.empty((0, 4), dtype=np.float32)
    )

    # Per-frame ground truth
    gt_by_frame = {}
    for frame_el in root.findall("frame"):
        frame_num = int(frame_el.attrib["num"])
        targets = []
        tl = frame_el.find("target_list")
        if tl is None:
            gt_by_frame[frame_num] = []
            continue

        for target in tl.findall("target"):
            box_el = target.find("box")
            attr_el = target.find("attribute")
            if box_el is None:
                continue

            ba = box_el.attrib
            x1 = float(ba["left"])
            y1 = float(ba["top"])
            x2 = x1 + float(ba["width"])
            y2 = y1 + float(ba["height"])

            truncation = (
                float(attr_el.attrib.get("truncation_ratio", 0.0))
                if attr_el is not None
                else 0.0
            )

            # Occlusion: take worst (most occluded) status across region_overlaps
            occ_status = None
            occ_el = target.find("occlusion")
            if occ_el is not None:
                for ro in occ_el.findall("region_overlap"):
                    s = int(ro.attrib.get("occlusion_status", 1))
                    if occ_status is None or s < occ_status:
                        occ_status = s

            targets.append({
                "box": [x1, y1, x2, y2],
                "occ_status": occ_status,
                "truncation_ratio": truncation,
            })

        gt_by_frame[frame_num] = targets

    return weather, ignore_boxes, gt_by_frame


# ── Per-frame evaluation ──────────────────────────────────────────
def evaluate_frame(model, img_path, gt_for_frame, ignore_boxes, model_label, imgsz):
    """Run one model on one frame. Returns a metrics dict."""
    results = model.predict(str(img_path), conf=CONF, iou=IOU_THRESH,
                            imgsz=imgsz, verbose=False)
    det = results[0].boxes
    boxes = det.xyxy.cpu().numpy()
    cls = det.cls.cpu().numpy().astype(int)
    mask = np.isin(cls, list(VEHICLE_COCO))
    pred_boxes = boxes[mask]

    # Build GT box array
    gt_boxes = (
        np.array([t["box"] for t in gt_for_frame], dtype=np.float32)
        if gt_for_frame
        else np.empty((0, 4), dtype=np.float32)
    )

    tp, fp, fn, miou = greedy_match(pred_boxes, gt_boxes)

    # FP suppression via ignored regions
    if fp > 0 and len(ignore_boxes) > 0:
        suppressed = suppress_fp_count(pred_boxes, gt_boxes, ignore_boxes)
        fp = max(0, fp - suppressed)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Occlusion counts (DETRAC -> UAVDT mapping)
    num_gt = len(gt_for_frame)
    num_gt_occ1 = 0  # no occlusion or is occluder
    num_gt_occ2 = 0  # occluded (by vehicle or ignored region)
    num_gt_occ3 = 0  # always 0 (DETRAC has no heavy/light distinction)
    num_gt_outofview = 0  # truncation_ratio > 0

    for t in gt_for_frame:
        s = t["occ_status"]
        if s is None or s == 1:
            num_gt_occ1 += 1
        else:
            num_gt_occ2 += 1
        if t["truncation_ratio"] > 0:
            num_gt_outofview += 1

    return dict(
        model=model_label,
        imgsz=imgsz,
        tp=tp, fp=fp, fn=fn,
        recall=round(rec, 4),
        precision=round(prec, 4),
        f1=round(f1, 4),
        mean_iou=round(miou, 4),
        num_gt=num_gt,
        num_gt_occ1=num_gt_occ1,
        num_gt_occ2=num_gt_occ2,
        num_gt_occ3=num_gt_occ3,
        num_gt_outofview=num_gt_outofview,
    )


# ── Final summary ─────────────────────────────────────────────────
def final_summary():
    """Print summary statistics from the completed CSV."""
    df = pd.read_csv(OUT_CSV)
    print(f"\nCSV: {OUT_CSV}", flush=True)
    print(f"Total rows:      {len(df)}", flush=True)
    print(f"Total sequences: {df['sequence'].nunique()}", flush=True)
    total_unique_frames = df.groupby(["sequence", "frame_id"]).ngroups
    print(f"Total frames:    {total_unique_frames}", flush=True)

    print("\n=== Mean Recall per Mode ===", flush=True)
    recall = df.groupby("model")["recall"].mean().round(4)
    print(recall.to_string(), flush=True)

    if "nano_320" in recall.index and "medium_640" in recall.index:
        gap = recall["medium_640"] - recall["nano_320"]
        print(f"\nMean recall gap (medium_640 - nano_320): {gap:.4f}", flush=True)

    print("\n=== Per-Weather Breakdown ===", flush=True)
    weather_recall = df.pivot_table(
        index="weather", columns="model", values="recall", aggfunc="mean"
    ).round(4)
    if "nano_320" in weather_recall.columns and "medium_640" in weather_recall.columns:
        weather_recall["gap"] = (
            weather_recall["medium_640"] - weather_recall["nano_320"]
        ).round(4)
    print(weather_recall.to_string(), flush=True)

    print("\n=== Per-Weather Detail ===", flush=True)
    weather_detail = df.groupby(["weather", "model"]).agg(
        sequences=("sequence", "nunique"),
        frames=("frame_id", "count"),
        mean_recall=("recall", "mean"),
        mean_precision=("precision", "mean"),
        mean_f1=("f1", "mean"),
    ).round(4)
    print(weather_detail.to_string(), flush=True)


# ── Main ───────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # ── Discover all 100 sequences ─────────────────────────────────
    sequences = []
    for xml_dir in [TRAIN_XML, TEST_XML]:
        if not xml_dir.exists():
            print(f"WARNING: annotation dir not found: {xml_dir}", flush=True)
            continue
        for xml_path in sorted(xml_dir.glob("*.xml")):
            seq_name = xml_path.stem  # e.g. "MVI_20011"
            img_dir = IMG_BASE / seq_name
            sequences.append((seq_name, xml_path, img_dir))

    print(f"Discovered {len(sequences)} sequences total\n", flush=True)

    # ── Crash-safe resume ──────────────────────────────────────────
    completed_seqs = set()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        # A sequence is complete only if BOTH modes are present
        completed_counts = existing.groupby("sequence")["model"].nunique()
        completed_seqs = set(
            completed_counts[completed_counts == len(MODES)].index
        )
        print(
            f"Resuming: {len(completed_seqs)} sequences already fully completed\n",
            flush=True,
        )

    # ── Load models ────────────────────────────────────────────────
    print("Loading models...", flush=True)
    models = {}
    for mode in MODES:
        models[mode["label"]] = YOLO(str(ECHO_ROOT / mode["model_file"]))
    print("Models loaded.\n", flush=True)

    total_frames_processed = 0
    n_seqs = len(sequences)

    for seq_idx, (seq_name, xml_path, img_dir) in enumerate(sequences):
        if seq_name in completed_seqs:
            print(
                f"[{seq_idx + 1}/{n_seqs}] {seq_name}: already complete, skipping",
                flush=True,
            )
            continue

        if not img_dir.exists():
            print(
                f"[{seq_idx + 1}/{n_seqs}] {seq_name}: WARNING images dir not found, skipping",
                flush=True,
            )
            continue

        seq_t0 = time.time()

        # Parse XML
        weather, ignore_boxes, gt_by_frame = parse_detrac_xml(xml_path)
        annotated_frame_nums = sorted(gt_by_frame.keys())

        print(
            f"[{seq_idx + 1}/{n_seqs}] {seq_name}: "
            f"{len(annotated_frame_nums)} annotated frames, weather={weather}",
            flush=True,
        )

        seq_records = []
        frames_done = 0

        for frame_num in annotated_frame_nums:
            img_path = img_dir / f"img{frame_num:05d}.jpg"
            if not img_path.exists():
                continue

            gt_for_frame = gt_by_frame[frame_num]
            frames_done += 1

            for mode in MODES:
                model_label = mode["label"]
                imgsz = mode["imgsz"]
                model = models[model_label]

                rec = evaluate_frame(
                    model, img_path, gt_for_frame,
                    ignore_boxes, model_label, imgsz,
                )
                rec["sequence"] = seq_name
                rec["frame_id"] = frame_num
                rec["weather"] = weather
                seq_records.append(rec)

        total_frames_processed += frames_done

        # Crash-safe: append this sequence immediately
        if not seq_records:
            print("    skipped (0 evaluable frames)", flush=True)
            continue

        seq_df = pd.DataFrame(seq_records)[COL_ORDER]
        write_header = not OUT_CSV.exists()
        seq_df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)

        seq_elapsed = time.time() - seq_t0
        total_elapsed = time.time() - t0
        print(
            f"    done ({frames_done} frames x {len(MODES)} modes) "
            f"[{seq_elapsed:.0f}s seq, {total_elapsed:.0f}s total]",
            flush=True,
        )

    # ── Final summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}", flush=True)
    print(
        f"Evaluation complete. Total time: {time.time() - t0:.1f}s\n", flush=True
    )

    if OUT_CSV.exists():
        final_summary()


if __name__ == "__main__":
    main()
