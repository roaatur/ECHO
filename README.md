 currently:
1-lstm1 is an uncertain lstm on synthetic data, missing student-t distribution output and autoregressive structure.

2- the cherry experiment is a baseline evaluation of the yolo11 variants to understand the performance difference, evaluated  on VisDrone,
cherry-picked 17 VID sequences (6582 frames) with varying lighting & crowdness.
per-frame metrics: recall, precision, F1, mean IoU at conf=0.25, IoU=0.5. committed to recall as our metric gap afterwards.
recall gap: +77% — medium finds 77% more objects than nano
precision gap: +18% 
IoU gap: +10% 
recall has the largest relative gap. precision is already high for both models (0.65 and 0.77) — both are conservative, when they fire they're usually right. IoU is also high for both (0.71 and 0.78) — both draw decent boxes.
The one metric that changes dramatically between the 2 is recall.
important note: unweighted recall inflates the gap because sparse frames (2-3 objects) produce extreme recall values that get equal weight as dense frames (50+ objects).
by ground truth weighting: each frame's recall is multiplied by how many objects are in that frame. a frame with 50 objects counts 50x more than a frame with 1 object.
sum(recall_i × gt_count_i) / sum(gt_count_i)
So frame with recall=0.50 and 2 objects contributes 1.0, while a frame with recall=0.20 and 100 objects contributes 20.0.
anyways the gap is still high: 68%

pending:
1-  analysis of 50 raw frame features and their correlation to per-frame recall of each model.
