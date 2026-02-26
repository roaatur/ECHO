# METEOR Dataset Validation Pipeline

## Context
I'm a PhD student working on adaptive model switching for YOLO object detection on edge devices. My thesis: "Dynamic Model Orchestration via Scene Complexity Forecasting in Edge Vision." I use two YOLO models:
- **nano@320**: lightweight, fast (~34ms), lower recall (~0.715)
- **medium@640**: accurate, slow (~250ms), higher recall (~0.921)

I need to predict per-frame miss count (fn = false negatives = objects the nano model misses) using image-level spatial features, then use an LSTM to forecast when to switch from nano to medium.

**The problem we're solving:** On fixed-camera datasets (UA-DETRAC), spatial features had within-sequence Spearman ρ ≈ 0.1-0.2 vs fn — too weak. The scene never changes in a fixed camera, so features stay constant while fn fluctuates as cars randomly enter/exit. We're switching to a **dashcam dataset (METEOR)** where the moving camera means scene structure genuinely changes within each sequence (highway→intersection→rural→tunnel), which should give much higher within-sequence correlation.

## Goal
Download a subset of METEOR, run both YOLO models, extract spatial features, and compute **within-sequence Spearman correlation** of each feature vs fn. This is a feasibility check — if within-sequence ρ > 0.3-0.4 for multiple features, the dataset works for our approach.

## Step 1: Download METEOR Subset
- Dataset is at: https://huggingface.co/datasets/XijunWang/METEOR/tree/main
- It's 93.4 GB split into 5 chunks (chunk_aa through chunk_ae). These are parts of a single tar archive.
- **For this validation, we only need ~50 video clips.** Download chunk_aa (20.4 GB), extract it, and pick 50 diverse clips.
- Each clip is ~1 minute at 30fps = ~1800 frames. 50 clips = ~90K frames — plenty for validation.
- The dataset structure after extraction: each clip folder named like `REC_YYYY_MM_DD_HH_MM_SS_F/` containing:
  - Video file (MP4)
  - Static XML (metadata: weather, road type, traffic density)
  - Dynamic XML (per-frame annotations: bounding boxes, class IDs, tracking IDs)

## Step 2: Extract Frames
- Use the provided script pattern: `python videos2frames.py` or use ffmpeg/OpenCV to extract all frames from each video clip
- Save frames as JPG/PNG, naming: `frame_000000.jpg`, `frame_000001.jpg`, etc.
- Keep track of which frames belong to which clip (this is our "sequence")

## Step 3: Parse Ground Truth Annotations
- Parse the dynamic XML files to extract per-frame bounding boxes
- For each frame, count:
  - `num_gt`: total number of ground truth bounding boxes (all vehicle classes)
  - Store bbox coordinates for IoU matching later
- Focus on vehicle-related classes: car, bus, truck, van, motorcycle, auto-rickshaw, bicycle (skip pedestrians for now, or include them — whichever classes are present)

## Step 4: Run YOLO Models
- Install ultralytics: `pip install ultralytics`
- Run **YOLOv8n** (nano) at resolution 320 on every frame:
  ```python
  from ultralytics import YOLO
  model_nano = YOLO('yolov8n.pt')
  results = model_nano(frame, imgsz=320, conf=0.25)
  ```
- Run **YOLOv8m** (medium) at resolution 640 on every frame:
  ```python
  model_medium = YOLO('yolov8m.pt')
  results = model_medium(frame, imgsz=640, conf=0.25)
  ```
- For each frame, match YOLO detections to ground truth using IoU ≥ 0.5
- Compute per frame:
  - `tp_nano`, `fp_nano`, `fn_nano` (fn = num_gt - tp_nano = miss count)
  - `tp_medium`, `fp_medium`, `fn_medium`
  - `recall_gap = recall_medium - recall_nano`

**IMPORTANT:** YOLO is pretrained on COCO. METEOR has Indian traffic with classes like auto-rickshaw that COCO doesn't have. Map METEOR classes to COCO classes where possible:
- car → car (COCO class 2)
- bus → bus (COCO class 5)
- truck → truck (COCO class 7)
- motorcycle/two-wheeler → motorcycle (COCO class 3)
- bicycle → bicycle (COCO class 1)
- person/pedestrian → person (COCO class 0)
- auto-rickshaw → no COCO equivalent (will be missed — that's fine, contributes to fn)

## Step 5: Extract Spatial Features (15 features)
For each frame, compute these image-level features from the raw image (NOT from detection results):

```python
import cv2
import numpy as np

def extract_spatial_features(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    features = {}
    
    # 1. Edge density (Canny)
    edges = cv2.Canny(gray, 50, 150)
    features['edge_density'] = np.mean(edges > 0)
    
    # 2-3. Gradient statistics
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    features['gradient_magnitude_mean'] = np.mean(mag)
    features['gradient_magnitude_std'] = np.std(mag)
    
    # 4. Texture (Laplacian variance)
    features['laplacian_variance'] = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 5. FAST keypoint count
    fast = cv2.FastFeatureDetector_create()
    kps = fast.detect(gray, None)
    features['FAST_keypoint_count'] = len(kps)
    
    # 6-7. Brightness statistics
    features['brightness_mean'] = np.mean(gray)
    features['brightness_std'] = np.std(gray)
    
    # 8-9. Color statistics (HSV saturation)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    features['saturation_mean'] = np.mean(hsv[:,:,1])
    features['saturation_std'] = np.std(hsv[:,:,1])
    
    # 10. Connected components (binary threshold)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, _ = cv2.connectedComponents(binary)
    features['connected_components'] = num_labels
    
    # 11-13. Spatial strip features (top/middle/bottom thirds)
    strip_h = h // 3
    for i, name in enumerate(['top', 'middle', 'bottom']):
        strip = gray[i*strip_h:(i+1)*strip_h, :]
        strip_edges = cv2.Canny(strip, 50, 150)
        features[f'{name}_edge_density'] = np.mean(strip_edges > 0)
    
    # 14. Entropy (texture complexity)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log2(hist))
    
    # 15. Optical flow magnitude (will need previous frame)
    # Skip for first frame, use 0
    features['flow_magnitude_mean'] = 0.0  # placeholder, compute separately
    
    return features
```

For **optical flow** (feature 15): compute between consecutive frames using `cv2.calcOpticalFlowFarneback()` and store the mean magnitude. This captures camera/scene motion — should be very informative for dashcam.

## Step 6: Compute Within-Sequence Spearman Correlation (THIS IS THE KEY OUTPUT)

```python
from scipy.stats import spearmanr
import pandas as pd

# df has columns: sequence, frame_idx, fn_nano, feature1, feature2, ...
feature_cols = [col for col in df.columns if col not in ['sequence', 'frame_idx', 'fn_nano', 'fn_medium', 'tp_nano', ...]]

results = []
for feature in feature_cols:
    # Global Spearman (pooled across all sequences)
    global_rho, global_p = spearmanr(df[feature], df['fn_nano'])
    
    # Within-sequence Spearman (averaged across sequences)
    per_seq_rhos = []
    for seq_name, group in df.groupby('sequence'):
        if group['fn_nano'].std() > 0 and group[feature].std() > 0:
            rho, p = spearmanr(group[feature], group['fn_nano'])
            if not np.isnan(rho):
                per_seq_rhos.append(rho)
    
    within_rho = np.mean(per_seq_rhos) if per_seq_rhos else 0
    
    results.append({
        'feature': feature,
        'global_spearman_rho': global_rho,
        'within_seq_spearman_rho': within_rho,
        'signal_drop': global_rho - within_rho,
        'n_valid_sequences': len(per_seq_rhos)
    })

results_df = pd.DataFrame(results).sort_values('within_seq_spearman_rho', ascending=False)
print(results_df.to_string())
```

## Expected Output
A table like:
```
feature                    global_rho  within_seq_rho  signal_drop  n_valid_seqs
optical_flow_mean           0.55        0.42            0.13         48
edge_density                0.48        0.35            0.13         50
gradient_magnitude_std      0.45        0.31            0.14         50
FAST_keypoint_count         0.42        0.28            0.14         50
...
```

## Success Criteria
- **If within_seq_rho > 0.3 for 5+ features**: METEOR works! Proceed with full pipeline.
- **If within_seq_rho 0.2-0.3**: Marginal improvement over DETRAC (was 0.1-0.2). Try delta features.
- **If within_seq_rho < 0.2**: Moving camera didn't help. Fundamental limitation.

## Also compute (bonus but important):
1. **Delta features**: For each feature, compute frame-to-frame difference within each sequence: `df.groupby('sequence')[feature].diff()`. Check within-sequence Spearman of Δ_feature vs Δ_fn.
2. **Basic statistics**: mean fn_nano per sequence, std fn_nano per sequence, number of objects per frame distribution.
3. **Recall comparison**: mean recall_nano vs mean recall_medium across all frames.

## Notes
- Use GPU if available for YOLO inference (much faster)
- Process one clip at a time to manage memory
- Save intermediate results (detections, features) as CSV/parquet so we don't re-run YOLO
- If 30fps is too many frames, subsample to 10fps (every 3rd frame) — still 600 frames per clip
- The validation only needs ~50 clips. Don't process the entire 93GB dataset yet.
