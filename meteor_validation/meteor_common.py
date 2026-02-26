"""Shared utilities for METEOR validation pipeline.

Constants, class mapping, XML parsing, clip discovery, and video reading.
"""

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ECHO_ROOT = SCRIPT_DIR.parent          # echo/ — contains .pt files
METEOR_ROOT = Path("/home/aljouhara/Desktop/METEOR/extracted")

# ── Detection constants ────────────────────────────────────────────
VEHICLE_COCO = {2, 3, 5, 7}            # car, motorcycle, bus, truck
CONF = 0.25
IOU_THRESH = 0.5
SUBSAMPLE = 3                          # every 3rd frame -> ~10 fps

# ── METEOR class -> COCO ID mapping ───────────────────────────────
# Actual METEOR class names discovered from annotations (PascalCase).
# Normalised to lowercase for lookup.
METEOR_TO_COCO = {
    # Vehicles matchable by YOLO (in VEHICLE_COCO)
    "car": 2,
    "bus": 5,
    "truck": 7,
    "motorbike": 3,           # METEOR uses "MotorBike"
    "motorcycle": 3,
    "scooter": 3,             # METEOR "Scooter" -> motorcycle
    "van": 7,                 # METEOR "Van" -> truck (COCO class 7)
    "bicycle": 1,
    # Persons
    "pedestrian": 0,
    "pedestrain": 0,          # known typo in METEOR annotations
    "person": 0,
    # No COCO equivalent — always contributes to fn
    "motorizedtricycle": None, # METEOR "MotorizedTricycle" (auto-rickshaw)
    "auto-rickshaw": None,
    "autorickshaw": None,
    "tractor": None,
    "constructionvehicle": None,
    "multiwheeler": None,
    "animal": None,
    # Skip ego vehicle entirely
    "egovehicle": None,       # camera's own car — skip in parser
}

# ── Import IoU matching from existing code ─────────────────────────
sys.path.insert(0, str(ECHO_ROOT / "res_experiment"))
from res_evaluate_detrac import greedy_match, iou_matrix  # noqa: E402


# ── Pascal VOC per-frame XML parser ───────────────────────────────

def _parse_voc_xml(xml_path):
    """Parse a single Pascal-VOC-style annotation XML.

    Returns list of dicts:
        {"box": [x1,y1,x2,y2], "class_name": str, "coco_id": int|None}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []

    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        class_name = name_el.text.strip()
        class_lower = class_name.lower().replace("_", "").replace("-", "").replace(" ", "")

        # Skip ego vehicle
        if class_lower == "egovehicle":
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        x1 = float(bndbox.find("xmin").text)
        y1 = float(bndbox.find("ymin").text)
        x2 = float(bndbox.find("xmax").text)
        y2 = float(bndbox.find("ymax").text)

        # Try exact lowercase match, then stripped match
        coco_id = METEOR_TO_COCO.get(class_lower)
        if coco_id is None and class_lower not in METEOR_TO_COCO:
            # Unknown class — treat as unmatchable
            coco_id = None

        objects.append({
            "box": [x1, y1, x2, y2],
            "class_name": class_name,
            "coco_id": coco_id,
        })

    return objects


def parse_clip_annotations(clip_dir):
    """Parse all per-frame annotations for a METEOR clip.

    Handles nested structure: clip_dir/Annotations/clip_name/Annotations/frame_*.xml
    Falls back to clip_dir/Annotations/frame_*.xml or clip_dir/frame_*.xml.

    Returns:
        gt_by_frame: dict[int -> list of dicts]
                     frame_number -> list of {"box": [x1,y1,x2,y2],
                                               "class_name": str,
                                               "coco_id": int|None}
    """
    # Find frame_*.xml files recursively under clip_dir
    xml_files = sorted(clip_dir.rglob("frame_*.xml"))
    if not xml_files:
        xml_files = sorted(clip_dir.rglob("frame_*.XML"))

    gt_by_frame = {}
    for xf in xml_files:
        # Extract frame number from filename like "frame_000042.xml"
        stem = xf.stem  # e.g. "frame_000042"
        try:
            frame_num = int(stem.split("_")[-1])
        except (ValueError, IndexError):
            # Fallback: try parsing the whole stem as a number
            try:
                frame_num = int(stem)
            except ValueError:
                continue

        gt_by_frame[frame_num] = _parse_voc_xml(xf)

    return gt_by_frame


# ── Static XML parser (metadata) ──────────────────────────────────

def parse_static_xml(xml_path):
    """Parse METEOR static annotation XML for clip metadata.

    Returns dict with keys: weather, road_type, traffic_density, time_of_day
    (values default to 'unknown' if not found).
    """
    defaults = {
        "weather": "unknown",
        "road_type": "unknown",
        "traffic_density": "unknown",
        "time_of_day": "unknown",
    }
    if xml_path is None or not xml_path.exists():
        return defaults

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return defaults

    # Try common element names / attribute names
    for key in defaults:
        # Direct child element
        el = root.find(key)
        if el is not None and el.text:
            defaults[key] = el.text.strip()
            continue
        # As attribute on root or first child
        for elem in [root] + list(root):
            val = elem.attrib.get(key)
            if val:
                defaults[key] = val
                break

    # Also try tag-based extraction (METEOR may use <tag name="..." value="..."/>)
    for tag in root.iter():
        name_attr = tag.attrib.get("name", "").lower().replace(" ", "_")
        value_attr = tag.attrib.get("value", "")
        if name_attr in defaults and value_attr:
            defaults[name_attr] = value_attr
        # Also check text content
        tag_lower = tag.tag.lower().replace(" ", "_")
        if tag_lower in defaults and tag.text and tag.text.strip():
            defaults[tag_lower] = tag.text.strip()

    return defaults


# ── Clip discovery ─────────────────────────────────────────────────

def _find_video(clip_dir):
    """Find video file in clip directory."""
    for ext in ("*.MP4", "*.mp4", "*.avi", "*.AVI"):
        vids = list(clip_dir.glob(ext))
        if vids:
            return vids[0]
    return None


def _find_static_xml(clip_dir):
    """Find the video-level annotation XML (metadata) in a clip directory.

    METEOR uses VideoXML/annotations.xml for CVAT-format video annotations.
    """
    # Look for VideoXML/annotations.xml (METEOR's actual structure)
    video_xml = clip_dir / "VideoXML" / "annotations.xml"
    if video_xml.exists():
        return video_xml

    # Fallback: look for files with 'static' or 'meta' in the name
    for pattern in ("*static*.xml", "*Static*.xml", "*meta*.xml"):
        matches = list(clip_dir.glob(pattern))
        if matches:
            return matches[0]

    # Look for XML files directly in clip_dir (not frame annotations)
    root_xmls = [f for f in clip_dir.glob("*.xml")
                 if f.parent == clip_dir and "frame" not in f.stem.lower()]
    if len(root_xmls) == 1:
        return root_xmls[0]

    return None


def discover_clips(meteor_root=None, max_clips=50):
    """Discover METEOR clip folders with stratified diversity selection.

    Scans all clips' static XML for metadata, then round-robin selects
    from (weather, road_type) groups to ensure diversity.

    Returns list of dicts:
        {"name": str, "dir": Path, "video": Path|None,
         "static_xml": Path|None, "ann_dir": Path,
         "weather": str, "road_type": str}
    """
    if meteor_root is None:
        meteor_root = METEOR_ROOT

    meteor_root = Path(meteor_root)
    if not meteor_root.exists():
        print(f"ERROR: METEOR data not found at {meteor_root}", flush=True)
        print("\nTo download METEOR chunk_aa (~20.4 GB):", flush=True)
        print("  pip install huggingface_hub", flush=True)
        print("  huggingface-cli download XijunWang/METEOR chunk_aa "
              "--repo-type dataset --local-dir /home/aljouhara/Desktop/METEOR",
              flush=True)
        print("  cd /home/aljouhara/Desktop/METEOR && cat chunk_a* | tar xf -",
              flush=True)
        return []

    # Find clip directories: immediate children of meteor_root that have a video
    all_clips = []
    for d in sorted(meteor_root.iterdir()):
        if not d.is_dir():
            continue
        if _find_video(d) is not None:
            all_clips.append(d)

    if not all_clips:
        print(f"No clip directories found under {meteor_root}", flush=True)
        return []

    print(f"Found {len(all_clips)} clip directories in {meteor_root}", flush=True)

    # Scan metadata for all clips
    clip_infos = []
    for clip_dir in all_clips:
        video = _find_video(clip_dir)
        static_xml = _find_static_xml(clip_dir)
        meta = parse_static_xml(static_xml)

        ann_dir = clip_dir / "Annotations"
        if not ann_dir.exists():
            ann_dir = clip_dir

        clip_infos.append({
            "name": clip_dir.name,
            "dir": clip_dir,
            "video": video,
            "static_xml": static_xml,
            "ann_dir": ann_dir,
            "weather": meta["weather"],
            "road_type": meta["road_type"],
        })

    if len(clip_infos) <= max_clips:
        print(f"Using all {len(clip_infos)} clips (fewer than {max_clips})",
              flush=True)
        _print_diversity_table(clip_infos)
        return clip_infos

    # Stratified selection: group by (weather, road_type), round-robin
    groups = defaultdict(list)
    for ci in clip_infos:
        key = (ci["weather"], ci["road_type"])
        groups[key].append(ci)

    # Shuffle within each group for randomness
    import random
    rng = random.Random(42)
    for key in groups:
        rng.shuffle(groups[key])

    selected = []
    group_iters = {k: iter(v) for k, v in groups.items()}
    keys = list(groups.keys())

    while len(selected) < max_clips:
        added_any = False
        for key in keys:
            if len(selected) >= max_clips:
                break
            try:
                clip = next(group_iters[key])
                selected.append(clip)
                added_any = True
            except StopIteration:
                continue
        if not added_any:
            break  # all groups exhausted

    print(f"Selected {len(selected)} diverse clips from {len(clip_infos)} total",
          flush=True)
    _print_diversity_table(selected)
    return selected


def _print_diversity_table(clips):
    """Print weather x road_type distribution of selected clips."""
    from collections import Counter
    weather_counts = Counter(c["weather"] for c in clips)
    road_counts = Counter(c["road_type"] for c in clips)
    cross = Counter((c["weather"], c["road_type"]) for c in clips)

    print("\n  Clip diversity summary:")
    print(f"    Weather:   {dict(weather_counts)}")
    print(f"    Road type: {dict(road_counts)}")
    if len(cross) <= 20:
        print(f"    Cross:     {dict(cross)}")
    print(flush=True)


# ── Video frame reader ─────────────────────────────────────────────

def read_video_frames(video_path, subsample=SUBSAMPLE):
    """Generator: read frames from video, yielding every Nth frame.

    Yields (frame_id, frame_bgr) where frame_id is the 0-based frame index.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARNING: cannot open video {video_path}", flush=True)
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % subsample == 0:
            yield frame_id, frame
        frame_id += 1

    cap.release()


def get_video_fps(video_path):
    """Get the FPS of a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_video_frame_count(video_path):
    """Get total number of frames in a video."""
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count
