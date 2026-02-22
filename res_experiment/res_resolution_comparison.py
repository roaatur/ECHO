"""Quick visual: show what YOLO sees at 320x320 vs 640x640 on 3 UA-DETRAC frames."""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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
OUT_PDF = Path(__file__).resolve().parent / "res_resolution_comparison.pdf"

# Target: 1 sunny, 1 night, 1 rainy — moderate vehicle count (5-15 GT boxes)
TARGET_WEATHER = ["sunny", "night", "rainy"]
TARGET_COUNT_RANGE = (5, 15)


def parse_xml_quick(xml_path):
    """Parse just weather + per-frame target counts and boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sa = root.find("sequence_attribute")
    weather = sa.attrib.get("sence_weather", "unknown") if sa is not None else "unknown"

    frames = {}
    for frame_el in root.findall("frame"):
        fnum = int(frame_el.attrib["num"])
        tl = frame_el.find("target_list")
        if tl is None:
            continue
        boxes = []
        for target in tl.findall("target"):
            box_el = target.find("box")
            if box_el is None:
                continue
            a = box_el.attrib
            x1 = float(a["left"])
            y1 = float(a["top"])
            w = float(a["width"])
            h = float(a["height"])
            boxes.append([x1, y1, w, h])
        if boxes:
            frames[fnum] = boxes
    return weather, frames


def find_candidates():
    """Find one good frame per target weather condition."""
    candidates = {}  # weather -> (seq_name, frame_num, boxes)

    for xml_dir in [TRAIN_XML, TEST_XML]:
        for xml_path in sorted(xml_dir.glob("*.xml")):
            seq_name = xml_path.stem
            weather, frames = parse_xml_quick(xml_path)
            if weather not in TARGET_WEATHER or weather in candidates:
                continue

            # Find a frame with moderate count, prefer middle of sequence
            sorted_fnums = sorted(frames.keys())
            mid = len(sorted_fnums) // 2
            best = None
            best_dist = float("inf")
            for fnum in sorted_fnums:
                n = len(frames[fnum])
                if TARGET_COUNT_RANGE[0] <= n <= TARGET_COUNT_RANGE[1]:
                    dist = abs(sorted_fnums.index(fnum) - mid)
                    if dist < best_dist:
                        best = fnum
                        best_dist = dist
            if best is not None:
                img_path = IMG_BASE / seq_name / f"img{best:05d}.jpg"
                if img_path.exists():
                    candidates[weather] = (seq_name, best, frames[best], img_path)
                    print(f"  {weather}: {seq_name} frame {best}, "
                          f"{len(frames[best])} vehicles, {img_path.name}")

            if len(candidates) == len(TARGET_WEATHER):
                break
        if len(candidates) == len(TARGET_WEATHER):
            break

    return candidates


def draw_page(fig, img_bgr, gt_boxes_xywh, seq_name, frame_num, weather):
    """Draw side-by-side 320 vs 640 with GT boxes scaled to each resolution."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for ax, sz in [(ax1, 320), (ax2, 640)]:
        resized = cv2.resize(img_rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
        ax.imshow(resized)

        sx = sz / w_orig
        sy = sz / h_orig

        for (x, y, w, h) in gt_boxes_xywh:
            rx = x * sx
            ry = y * sy
            rw = w * sx
            rh = h * sy
            rect = patches.Rectangle(
                (rx, ry), rw, rh,
                linewidth=1.5, edgecolor="lime", facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title(f"{sz}x{sz}", fontsize=14, fontweight="bold")
        ax.set_xlim(0, sz)
        ax.set_ylim(sz, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{seq_name}  frame {frame_num}  —  {weather}  —  {len(gt_boxes_xywh)} GT vehicles",
        fontsize=13, y=0.98,
    )


def main():
    print("Finding candidate frames...", flush=True)
    candidates = find_candidates()

    if len(candidates) < len(TARGET_WEATHER):
        missing = set(TARGET_WEATHER) - set(candidates.keys())
        print(f"WARNING: could not find frames for: {missing}")

    with PdfPages(str(OUT_PDF)) as pdf:
        for weather in TARGET_WEATHER:
            if weather not in candidates:
                continue
            seq_name, fnum, gt_boxes, img_path = candidates[weather]
            img_bgr = cv2.imread(str(img_path))
            print(f"  Drawing {weather}: {seq_name} frame {fnum} "
                  f"({img_bgr.shape[1]}x{img_bgr.shape[0]})", flush=True)

            fig = plt.figure(figsize=(14, 7))
            draw_page(fig, img_bgr, gt_boxes, seq_name, fnum, weather)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"\nSaved: {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
