"""Extract 37 per-frame features for cherry-picked VisDrone VID sequences."""

import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN

# ── Paths & constants ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
VID_BASE = ROOT / "VisDrone2019-VID-Cherry-Picked"
OUT_CSV = ROOT / "berry_frame_features.csv"

SEQ_DIRS = [
    (VID_BASE / "cherry_train_sequences", VID_BASE / "cherry_train_annotations"),
    (VID_BASE / "cherry_val_sequences",   VID_BASE / "cherry_val_annotations"),
]

VISDRONE_TO_COCO = {
    1: 0, 2: 0, 3: 1, 4: 2, 5: 2,
    6: 7, 7: 3, 8: 3, 9: 5, 10: 3,
}
EPS = 1e-6


# ── Timing accumulator ────────────────────────────────────────────
feature_timings = defaultdict(float)


def timed(name, func):
    """Call func(), accumulate elapsed time under name, return result."""
    t0 = time.perf_counter()
    result = func()
    feature_timings[name] += time.perf_counter() - t0
    return result


# ── Annotation parsers ────────────────────────────────────────────
def parse_vid_gt_xywh(ann_path):
    """Parse VID annotation → dict[frame_id, Nx4 array [x,y,w,h]] in original coords.
    Same VISDRONE_TO_COCO class filter as evaluation script for consistency."""
    frames = defaultdict(list)
    for line in ann_path.read_text().strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 10:
            continue
        cat = int(parts[7])
        if cat not in VISDRONE_TO_COCO:
            continue
        fid = int(parts[0])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        frames[fid].append([x, y, w, h])
    return {fid: np.array(boxes) for fid, boxes in frames.items()}


# ── IoU helper ─────────────────────────────────────────────────────
def iou_matrix(a, b):
    a = a[:, None, :]; b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + EPS)


# ── Resize ─────────────────────────────────────────────────────────
def resize_640(frame):
    h, w = frame.shape[:2]
    scale = 640.0 / w
    return cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR), scale


# ── Image Quality Features (12) ───────────────────────────────────
def compute_image_features(bgr, gray, gray_u8, grad_mag):
    f = {}
    h, w = gray.shape

    f["mean_brightness"] = timed("mean_brightness", lambda: gray.mean())
    f["brightness_std"] = timed("brightness_std", lambda: gray.std())

    def _dark_channel():
        dark = cv2.erode(bgr.min(axis=2), np.ones((15, 15), np.uint8))
        return dark.mean() / 255.0
    f["dark_channel"] = timed("dark_channel", _dark_channel)

    def _dynamic_range():
        p1, p99 = np.percentile(gray, [1, 99])
        return np.log(p99 + 1) - np.log(p1 + 1)
    f["dynamic_range"] = timed("dynamic_range", _dynamic_range)

    def _weber():
        ph, pw = h // 8, w // 8
        c = []
        for r in range(8):
            for cc in range(8):
                patch = gray[r*ph:(r+1)*ph, cc*pw:(cc+1)*pw]
                c.append((patch.max() - patch.min()) / (patch.mean() + EPS))
        return np.mean(c)
    f["weber_contrast"] = timed("weber_contrast", _weber)

    f["blur_estimate"] = timed("blur_estimate",
                               lambda: cv2.Laplacian(gray, cv2.CV_64F).var())

    def _spatial_freq():
        fft = np.fft.fft2(gray)
        psd = np.abs(np.fft.fftshift(fft)) ** 2
        cy, cx = psd.shape[0] // 2, psd.shape[1] // 2
        Y, X = np.ogrid[:psd.shape[0], :psd.shape[1]]
        freq_norm = np.sqrt(((Y - cy) / psd.shape[0])**2 + ((X - cx) / psd.shape[1])**2)
        return psd[freq_norm < 0.05].sum() / (psd.sum() + EPS)
    f["spatial_freq_low_ratio"] = timed("spatial_freq_low_ratio", _spatial_freq)

    f["grad_mag_mean"] = timed("grad_mag_mean", lambda: grad_mag.mean())
    f["grad_mag_var"] = timed("grad_mag_var", lambda: grad_mag.var())

    def _bimodality():
        flat = gray.ravel()
        return (stats.skew(flat)**2 + 1) / (stats.kurtosis(flat) + 3)
    f["histogram_bimodality"] = timed("histogram_bimodality", _bimodality)

    def _backlight():
        qh, qw = h // 4, w // 4
        center = gray[qh:3*qh, qw:3*qw].mean()
        mask = np.ones_like(gray, dtype=bool)
        mask[qh:3*qh, qw:3*qw] = False
        return gray[mask].mean() / (center + EPS)
    f["backlight_ratio"] = timed("backlight_ratio", _backlight)

    def _edge_density():
        edges = cv2.Canny(gray_u8, 100, 200)
        return edges.astype(bool).sum() / edges.size
    f["edge_density"] = timed("edge_density", _edge_density)

    return f


# ── GT-Derived Features (15) ──────────────────────────────────────
def compute_gt_features(boxes_xywh, orig_h, orig_w):
    f = {}
    n = len(boxes_xywh)
    f["gt_object_count"] = n

    if n == 0:
        for k in ["small_obj_density", "medium_obj_density", "large_obj_density",
                   "mean_gt_area", "max_gt_area", "min_gt_area", "scale_span_ratio",
                   "gt_area_var", "mean_aspect_ratio", "aspect_ratio_extremity",
                   "obj_density_per_area", "mean_pairwise_iou",
                   "boundary_truncation_ratio", "cluster_count"]:
            f[k] = 0.0
        return f

    ws = boxes_xywh[:, 2]
    hs = boxes_xywh[:, 3]
    sizes = np.maximum(ws, hs)
    areas = np.sqrt(ws * hs)
    ars = np.maximum(ws / (hs + EPS), hs / (ws + EPS))

    def _size_densities():
        return ((sizes < 32).sum() / n,
                ((sizes >= 32) & (sizes < 96)).sum() / n,
                (sizes >= 96).sum() / n)
    sm, md, lg = timed("size_densities", _size_densities)
    f["small_obj_density"] = sm
    f["medium_obj_density"] = md
    f["large_obj_density"] = lg

    def _area_stats():
        return areas.mean(), areas.max(), areas.min(), areas.max()/(areas.min()+EPS), areas.var()
    mn, mx, mi, sp, va = timed("area_stats", _area_stats)
    f["mean_gt_area"] = mn; f["max_gt_area"] = mx; f["min_gt_area"] = mi
    f["scale_span_ratio"] = sp; f["gt_area_var"] = va

    f["mean_aspect_ratio"] = timed("aspect_ratio", lambda: ars.mean())
    f["aspect_ratio_extremity"] = ars.max()
    f["obj_density_per_area"] = n / (orig_h * orig_w + EPS)

    def _pairwise_iou():
        if n < 2:
            return 0.0
        xyxy = np.column_stack([boxes_xywh[:, 0], boxes_xywh[:, 1],
                                boxes_xywh[:, 0] + ws, boxes_xywh[:, 1] + hs])
        if n <= 50:
            ious = iou_matrix(xyxy, xyxy)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            return ious[mask].mean()
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=(50, 2), replace=True)
        idx = idx[idx[:, 0] != idx[:, 1]]
        return np.mean([iou_matrix(xyxy[i:i+1], xyxy[j:j+1])[0, 0] for i, j in idx]) if len(idx) else 0.0
    f["mean_pairwise_iou"] = timed("mean_pairwise_iou", _pairwise_iou)

    def _boundary():
        xs, ys = boxes_xywh[:, 0], boxes_xywh[:, 1]
        return ((xs < 5) | (ys < 5) | (xs + ws > orig_w - 5) | (ys + hs > orig_h - 5)).sum() / n
    f["boundary_truncation_ratio"] = timed("boundary_truncation", _boundary)

    def _clusters():
        if n < 2:
            return 0
        cx = boxes_xywh[:, 0] + ws / 2
        cy = boxes_xywh[:, 1] + hs / 2
        labels = DBSCAN(eps=100, min_samples=2).fit_predict(np.column_stack([cx, cy]))
        return labels.max() + 1 if labels.max() >= 0 else 0
    f["cluster_count"] = timed("cluster_count", _clusters)

    return f


# ── Temporal Features (10) ────────────────────────────────────────
def compute_temporal_features(gray_u8, prev_gray_u8, history):
    f = {}
    brightness = gray_u8.astype(np.float64).mean()

    if prev_gray_u8 is not None:
        def _flow():
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray_u8, gray_u8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            return mag.mean(), mag.var()
        flow_mag, flow_var = timed("optical_flow", _flow)
        brightness_change = abs(brightness - history["brightness"][-1])
    else:
        flow_mag = np.nan
        flow_var = np.nan
        brightness_change = np.nan

    f["optical_flow_mag"] = flow_mag
    f["optical_flow_var"] = flow_var
    f["interframe_brightness_change"] = brightness_change

    history["brightness"].append(brightness)
    history["flow_mag"].append(flow_mag)

    for win in [5, 10, 30]:
        buf = history["brightness"]
        f[f"brightness_var_{win}f"] = np.std(buf[-win:]) if len(buf) >= win else np.nan

    for win in [5, 10, 15, 30]:
        buf = history["flow_mag"]
        valid = [v for v in buf[-win:] if not np.isnan(v)]
        f[f"flow_mean_{win}f"] = np.mean(valid) if len(valid) >= win else np.nan

    return f


# ── Main ───────────────────────────────────────────────────────────
def main():
    sequences = []
    for seq_dir, ann_dir in SEQ_DIRS:
        for s in sorted(seq_dir.iterdir()):
            if s.is_dir():
                ann = ann_dir / (s.name + ".txt")
                if ann.exists():
                    sequences.append((s.name, s, ann))
    print(f"Found {len(sequences)} sequences")

    records = []
    total_t0 = time.time()

    for seq_name, seq_img_dir, ann_path in sequences:
        t0 = time.time()
        gt_by_frame = parse_vid_gt_xywh(ann_path)
        annotated_frames = sorted(gt_by_frame.keys())

        first_img = seq_img_dir / f"{annotated_frames[0]:07d}.jpg"
        orig_h, orig_w = cv2.imread(str(first_img)).shape[:2]

        prev_gray_u8 = None
        history = {"brightness": [], "flow_mag": []}

        for fid in annotated_frames:
            img_path = seq_img_dir / f"{fid:07d}.jpg"
            if not img_path.exists():
                continue

            frame_bgr = cv2.imread(str(img_path))

            t_io = time.perf_counter()
            resized, _ = resize_640(frame_bgr)
            feature_timings["io_resize"] += time.perf_counter() - t_io

            t_cache = time.perf_counter()
            gray_u8 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = gray_u8.astype(np.float64) / 255.0
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sx**2 + sy**2)
            feature_timings["cache_gray_grad"] += time.perf_counter() - t_cache

            img_feats = compute_image_features(resized, gray, gray_u8, grad_mag)
            boxes = gt_by_frame.get(fid, np.empty((0, 4)))
            gt_feats = compute_gt_features(boxes, orig_h, orig_w)
            temp_feats = compute_temporal_features(gray_u8, prev_gray_u8, history)
            prev_gray_u8 = gray_u8

            rec = {"sequence": seq_name, "frame_id": fid}
            rec.update(img_feats)
            rec.update(gt_feats)
            rec.update(temp_feats)
            records.append(rec)

        elapsed = time.time() - t0
        print(f"  {seq_name}: {len(annotated_frames)} frames, {elapsed:.1f}s")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    total_elapsed = time.time() - total_t0
    print(f"\nSaved {OUT_CSV} ({len(df)} rows, {len(df.columns)} cols, {total_elapsed:.0f}s total)")

    # ── Timing summary ──
    n_frames = len(df)
    print(f"\n{'Feature':<30s} {'Total (s)':>10s} {'Per-frame (ms)':>15s}")
    print("-" * 57)
    for name, total_s in sorted(feature_timings.items(), key=lambda x: -x[1]):
        print(f"{name:<30s} {total_s:>10.2f} {total_s/n_frames*1000:>15.3f}")


if __name__ == "__main__":
    main()
