"""Verification script: parse a few UA-DETRAC XMLs, check image paths, print samples."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

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


def parse_detrac_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sa = root.find("sequence_attribute")
    weather = sa.attrib.get("sence_weather", "unknown") if sa is not None else "unknown"
    camera = sa.attrib.get("camera_state", "unknown") if sa is not None else "unknown"

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
            vtype = attr_el.attrib.get("vehicle_type", "?") if attr_el is not None else "?"

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
                "vehicle_type": vtype,
            })

        gt_by_frame[frame_num] = targets

    return weather, camera, ignore_boxes, gt_by_frame


def verify_sequence(xml_path):
    seq_name = xml_path.stem
    img_dir = IMG_BASE / seq_name
    print(f"\n{'='*70}")
    print(f"Sequence: {seq_name}")
    print(f"  XML:      {xml_path}")
    print(f"  Img dir:  {img_dir}  (exists={img_dir.exists()})")

    weather, camera, ignore_boxes, gt_by_frame = parse_detrac_xml(xml_path)
    annotated_frames = sorted(gt_by_frame.keys())
    total_targets = sum(len(v) for v in gt_by_frame.values())

    print(f"  Weather:  {weather}")
    print(f"  Camera:   {camera}")
    print(f"  Ignored regions: {len(ignore_boxes)}")
    for i, box in enumerate(ignore_boxes):
        print(f"    [{i}] x1={box[0]:.1f} y1={box[1]:.1f} x2={box[2]:.1f} y2={box[3]:.1f}")
    print(f"  Annotated frames: {len(annotated_frames)}")
    print(f"  Frame range: {annotated_frames[0]}..{annotated_frames[-1]}" if annotated_frames else "  (no frames)")
    print(f"  Total target instances: {total_targets}")

    # Verify image path resolution for first, middle, last frame
    check_frames = []
    if len(annotated_frames) >= 3:
        check_frames = [annotated_frames[0], annotated_frames[len(annotated_frames)//2], annotated_frames[-1]]
    else:
        check_frames = annotated_frames

    print(f"\n  Image path check (img{{num:05d}}.jpg):")
    for fnum in check_frames:
        img_path = img_dir / f"img{fnum:05d}.jpg"
        exists = img_path.exists()
        print(f"    frame {fnum:5d} -> {img_path.name}  exists={exists}")
        if not exists:
            # Try 6-digit as fallback check
            alt = img_dir / f"img{fnum:06d}.jpg"
            print(f"      !! 5-digit FAILED. 6-digit ({alt.name}) exists={alt.exists()}")

    # Print sample GT boxes from frame 1 (or first available)
    sample_frame = annotated_frames[0] if annotated_frames else None
    if sample_frame is not None:
        targets = gt_by_frame[sample_frame]
        print(f"\n  Sample GT boxes (frame {sample_frame}, {len(targets)} targets):")
        for i, t in enumerate(targets[:5]):
            b = t["box"]
            print(f"    [{i}] box=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]  "
                  f"type={t['vehicle_type']}  occ={t['occ_status']}  trunc={t['truncation_ratio']}")
        if len(targets) > 5:
            print(f"    ... ({len(targets) - 5} more)")

    # Occlusion distribution across all frames
    occ_none = occ_1 = occ_0 = occ_neg1 = trunc_pos = 0
    vtypes = {}
    for targets in gt_by_frame.values():
        for t in targets:
            s = t["occ_status"]
            if s is None:
                occ_none += 1
            elif s == 1:
                occ_1 += 1
            elif s == 0:
                occ_0 += 1
            elif s == -1:
                occ_neg1 += 1
            if t["truncation_ratio"] > 0:
                trunc_pos += 1
            vt = t["vehicle_type"]
            vtypes[vt] = vtypes.get(vt, 0) + 1

    print(f"\n  Occlusion distribution (total={total_targets}):")
    print(f"    No <occlusion>:   {occ_none}  -> occ1")
    print(f"    status=1 (occluder): {occ_1}  -> occ1")
    print(f"    status=0 (by veh):   {occ_0}  -> occ2")
    print(f"    status=-1 (by ign):  {occ_neg1}  -> occ2")
    print(f"    truncation>0:        {trunc_pos}  -> outofview")
    print(f"  Vehicle types: {vtypes}")


def main():
    # Pick 3 sequences: one train, one test, one with known night weather
    test_xmls = [
        TRAIN_XML / "MVI_20011.xml",   # large train sequence
        TRAIN_XML / "MVI_39811.xml",    # night, unstable camera
        TEST_XML / "MVI_39031.xml",     # test sequence
    ]

    for xml_path in test_xmls:
        if not xml_path.exists():
            print(f"WARNING: {xml_path} not found!")
            continue
        verify_sequence(xml_path)

    # Also count total sequences discoverable
    print(f"\n{'='*70}")
    print("Discovery check:")
    train_count = len(list(TRAIN_XML.glob("*.xml")))
    test_count = len(list(TEST_XML.glob("*.xml")))
    img_count = len([d for d in IMG_BASE.iterdir() if d.is_dir()])
    print(f"  Train XMLs: {train_count}")
    print(f"  Test XMLs:  {test_count}")
    print(f"  Total XMLs: {train_count + test_count}")
    print(f"  Image sequence dirs: {img_count}")

    # Check all sequences have matching image dirs
    missing = 0
    for xml_dir in [TRAIN_XML, TEST_XML]:
        for xml_path in xml_dir.glob("*.xml"):
            seq_name = xml_path.stem
            if not (IMG_BASE / seq_name).exists():
                print(f"  MISSING image dir for {seq_name}")
                missing += 1
    print(f"  Sequences missing image dirs: {missing}")


if __name__ == "__main__":
    main()
