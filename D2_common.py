#!/usr/bin/env python3
"""D²-City shared utilities.

Dataset: D²-City (DiDi, arXiv:1904.01975)
  - 1280×720 @ 25 fps dashcam clips
  - CVAT track-based XML annotations (one XML per clip)
  - 12 classes: car, van, bus, truck, person, bicycle, motorcycle,
                open-tricycle, closed-tricycle, forklift, large-block, small-block
  - Validation split: 100 clips in validation-video/ + validation-annotation/

Path layout:
  /Users/meme/Downloads/echo-main2/ D²-City/
    validation-video/{hash}.mp4
    validation-annotation/{hash}.xml
"""

from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
D2_BASE    = Path("/Users/meme/Downloads/echo-main2/ D²-City")
VIDEO_DIR  = D2_BASE / "validation-video"
ANNOT_DIR  = D2_BASE / "validation-annotation"

# ── Evaluation constants (mirror other experiments) ──────────────────────────
CONF        = 0.25
IOU_THRESH  = 0.50
SUBSAMPLE   = 3     # keep every 3rd frame → ~8.3 fps from 25 fps source

# ── D²-City label → COCO class ID ────────────────────────────────────────────
# Labels confirmed from validation-annotation XML meta section:
#   car, van, bus, truck, person, bicycle, motorcycle,
#   open-tricycle, closed-tricycle, forklift, large-block, small-block
#
# Mapping rationale:
#   van → 2 (car): In Chinese urban traffic, vans are passenger-style vehicles
#                  that YOLO11 typically classifies as cars. The paper also
#                  recommends this mapping.
#   open/closed-tricycle → None: no COCO equivalent; excluded from fn_coco
#                                (counted only in fn_total)
#   forklift, large-block, small-block → None: not in COCO

D2_LABEL_TO_COCO = {
    "car":            2,   # COCO: car
    "van":            2,   # COCO: car (closest match)
    "bus":            5,   # COCO: bus
    "truck":          7,   # COCO: truck
    "person":         0,   # COCO: person
    "bicycle":        1,   # COCO: bicycle
    "motorcycle":     3,   # COCO: motorcycle
    # excluded from fn_coco (no COCO equivalent):
    "open-tricycle":  None,
    "closed-tricycle": None,
    "forklift":       None,
    "large-block":    None,
    "small-block":    None,
}

# COCO IDs that YOLO11 (COCO-trained) can detect — used as the evaluation filter
COCO_VEHICLE_IDS = {0, 1, 2, 3, 5, 7}


# ── Annotation parser ────────────────────────────────────────────────────────

def parse_cvat_xml(xml_path):
    """Parse a CVAT track-format XML file.

    Returns:
        frames_coco  : dict {frame_idx: [{x1,y1,x2,y2,coco_id}]}
                       Only boxes with a COCO-mappable class are included.
        frames_total : dict {frame_idx: [{x1,y1,x2,y2,label}]}
                       All boxes regardless of COCO mapping.

    Filtering rules (applied to every <box> element):
      - Skip if outside="1" (object not visible in frame)
      - Skip if occluded="almost" (too occluded to reliably detect)
    """
    root = ET.parse(str(xml_path)).getroot()

    frames_coco  = defaultdict(list)
    frames_total = defaultdict(list)

    for track in root.iter("track"):
        label   = track.get("label", "")
        coco_id = D2_LABEL_TO_COCO.get(label)   # None if unmapped

        for box in track.iter("box"):
            # Object is outside the frame boundary
            if box.get("outside", "0") == "1":
                continue
            # Heavily occluded — unlikely to be detected by either model
            if box.get("occluded", "no") == "almost":
                continue

            frame_idx = int(box.get("frame"))
            entry = {
                "x1":    float(box.get("xtl")),
                "y1":    float(box.get("ytl")),
                "x2":    float(box.get("xbr")),
                "y2":    float(box.get("ybr")),
                "label": label,
            }

            # fn_total counts every visible object including unmapped classes
            frames_total[frame_idx].append(entry)

            # fn_coco counts only COCO-detectable objects
            if coco_id is not None:
                frames_coco[frame_idx].append({**entry, "coco_id": coco_id})

    return dict(frames_coco), dict(frames_total)


# ── Clip discovery ───────────────────────────────────────────────────────────

def discover_clips():
    """Return sorted list of dicts for all paired validation clips.

    Each dict: {clip_id, video_path, xml_path}
    Pairing: validation-video/{hash}.mp4 ↔ validation-annotation/{hash}.xml
    """
    clips = []
    for vid_path in sorted(VIDEO_DIR.glob("*.mp4")):
        clip_id  = vid_path.stem
        xml_path = ANNOT_DIR / f"{clip_id}.xml"
        if not xml_path.exists():
            continue
        clips.append({
            "clip_id":    clip_id,
            "video_path": vid_path,
            "xml_path":   xml_path,
        })
    return clips


# ── IoU utilities (same logic as res_evaluate_detrac.py) ────────────────────

def iou_matrix(boxes_a, boxes_b):
    """Vectorised IoU between all pairs of boxes.

    boxes_a : list of dicts with keys x1, y1, x2, y2  (N boxes)
    boxes_b : list of dicts with keys x1, y1, x2, y2  (M boxes)
    Returns  : (N, M) float64 array
    """
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    def to_arr(boxes):
        return np.array([[b["x1"], b["y1"], b["x2"], b["y2"]]
                         for b in boxes], dtype=np.float64)

    a = to_arr(boxes_a)   # (N, 4)
    b = to_arr(boxes_b)   # (M, 4)

    inter_x1 = np.maximum(a[:, 0:1], b[:, 0])   # (N, M)
    inter_y1 = np.maximum(a[:, 1:2], b[:, 1])
    inter_x2 = np.minimum(a[:, 2:3], b[:, 2])
    inter_y2 = np.minimum(a[:, 3:4], b[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h                   # (N, M)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])   # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])   # (M,)
    union  = area_a[:, None] + area_b[None, :] - inter    # (N, M)

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def greedy_match(det_boxes, gt_boxes, iou_thresh=IOU_THRESH):
    """Greedy highest-IoU matching of detections to GT boxes.

    Returns:
        matched_det_idx : set of matched detection indices
        matched_gt_idx  : set of matched GT indices
    """
    if not det_boxes or not gt_boxes:
        return set(), set()

    ious = iou_matrix(det_boxes, gt_boxes).copy()
    matched_det, matched_gt = set(), set()

    for _ in range(min(len(det_boxes), len(gt_boxes))):
        if ious.size == 0:
            break
        d_idx, g_idx = np.unravel_index(ious.argmax(), ious.shape)
        if ious[d_idx, g_idx] < iou_thresh:
            break
        matched_det.add(int(d_idx))
        matched_gt.add(int(g_idx))
        ious[d_idx, :] = -1.0
        ious[:, g_idx] = -1.0

    return matched_det, matched_gt


# ── Self-test / data discovery ───────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 70)
    print("D²-CITY DATA DISCOVERY")
    print("=" * 70)

    clips = discover_clips()
    print(f"\n  Found {len(clips)} paired clips in validation split")

    # Sample one clip for inspection
    sample = random.choice(clips)
    print(f"\n  Sample clip: {sample['clip_id']}")
    print(f"    Video : {sample['video_path']}")
    print(f"    XML   : {sample['xml_path']}")

    frames_coco, frames_total = parse_cvat_xml(sample["xml_path"])
    print(f"\n  Annotation stats:")
    print(f"    Frames with COCO GT : {len(frames_coco)}")
    print(f"    Frames with any GT  : {len(frames_total)}")

    # Count labels across all tracks
    import xml.etree.ElementTree as ET2
    root = ET.parse(str(sample["xml_path"])).getroot()
    label_counts = defaultdict(int)
    for track in root.iter("track"):
        label_counts[track.get("label", "?")] += sum(
            1 for box in track.iter("box") if box.get("outside", "0") != "1"
        )
    print("\n  Label counts (visible boxes):")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        coco = D2_LABEL_TO_COCO.get(lbl)
        coco_str = f"→ COCO {coco}" if coco is not None else "→ excluded"
        print(f"    {lbl:20s} {cnt:6d} boxes  {coco_str}")

    # Video info
    import cv2
    cap = cv2.VideoCapture(str(sample["video_path"]))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    nfr    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"\n  Video info:")
    print(f"    Resolution : {width}×{height}")
    print(f"    FPS        : {fps}")
    print(f"    Frames     : {nfr} ({nfr / fps:.1f}s)")
    print(f"    Subsampled : {nfr // SUBSAMPLE} frames at effective {fps / SUBSAMPLE:.1f} fps")
