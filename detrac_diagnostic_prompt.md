# DETRAC: recoverable_fn + Windowed Spearman Analysis

## Context
We already have DETRAC data with per-frame tp_nano, tp_medium, fn values, and 15 spatial features extracted. The METEOR experiment with 40 features showed ρ ≈ 0 for ALL features — moving camera didn't help. We're going back to DETRAC to test whether the target metric and prediction granularity are the issue.

## Goal
Three quick diagnostic experiments on existing DETRAC data. No new feature extraction needed — just post-processing and re-analysis.

---

## Step 1: Compute recoverable_fn on DETRAC

Find the existing DETRAC detection results (likely a CSV or parquet with columns including tp_nano, tp_medium, fn per frame per sequence). Add:

```python
df['recoverable_fn'] = (df['tp_medium'] - df['tp_nano']).clip(lower=0)
```

Report the distribution:
- mean, std, min, max
- percentiles: 25th, 50th, 75th, 90th, 95th
- % of frames with recoverable_fn = 0
- compare to fn_nano distribution side by side

---

## Step 2: Per-frame Spearman — fn_nano vs recoverable_fn

Run within-sequence Spearman ρ for the existing 15 spatial features against BOTH targets:

```python
from scipy.stats import spearmanr
import numpy as np

feature_cols = [list of 15 spatial feature columns]
targets = ['fn_nano', 'recoverable_fn']

results = []
for target in targets:
    for feature in feature_cols:
        per_seq_rhos = []
        for seq, group in df.groupby('sequence'):
            if len(group) < 30:
                continue
            if group[target].std() > 0 and group[feature].std() > 0:
                rho, _ = spearmanr(group[feature], group[target])
                if not np.isnan(rho):
                    per_seq_rhos.append(rho)
        
        within_rho = np.mean(per_seq_rhos) if per_seq_rhos else 0
        results.append({
            'target': target,
            'feature': feature,
            'within_seq_rho': round(within_rho, 3),
            'n_valid_seqs': len(per_seq_rhos)
        })

# Pivot to compare side by side
results_df = pd.DataFrame(results)
comparison = results_df.pivot(index='feature', columns='target', values='within_seq_rho')
comparison['rho_diff'] = comparison['recoverable_fn'] - comparison['fn_nano']
comparison = comparison.sort_values('fn_nano', ascending=False)
print(comparison.to_string())
```

Key question: does recoverable_fn give higher or lower ρ than fn_nano on DETRAC?

---

## Step 3: Windowed Spearman (THE CRITICAL TEST)

Average both targets AND features over 30-frame windows within each sequence. This smooths per-frame noise and tests whether the signal improves at coarser granularity.

```python
WINDOW_SIZE = 30  # 1 second at 30fps

windowed_rows = []
for seq, group in df.groupby('sequence'):
    group = group.sort_values('frame_idx').reset_index(drop=True)
    n_windows = len(group) // WINDOW_SIZE
    
    for w in range(n_windows):
        chunk = group.iloc[w * WINDOW_SIZE : (w + 1) * WINDOW_SIZE]
        row = {'sequence': seq, 'window_idx': w}
        
        # Average targets
        row['fn_nano_w'] = chunk['fn_nano'].mean()
        row['recoverable_fn_w'] = chunk['recoverable_fn'].mean()
        
        # Average features
        for feat in feature_cols:
            row[feat + '_w'] = chunk[feat].mean()
        
        windowed_rows.append(row)

wdf = pd.DataFrame(windowed_rows)

# Now run within-sequence Spearman on windowed data
for target in ['fn_nano_w', 'recoverable_fn_w']:
    print(f"\n=== Windowed Spearman: {target} ===")
    for feature in feature_cols:
        feat_w = feature + '_w'
        per_seq_rhos = []
        for seq, group in wdf.groupby('sequence'):
            if len(group) < 5:  # need at least 5 windows
                continue
            if group[target].std() > 0 and group[feat_w].std() > 0:
                rho, _ = spearmanr(group[feat_w], group[target])
                if not np.isnan(rho):
                    per_seq_rhos.append(rho)
        
        within_rho = np.mean(per_seq_rhos) if per_seq_rhos else 0
        print(f"  {feature:40s} within_seq_rho = {within_rho:.3f}  (n={len(per_seq_rhos)})")
```

Also try window sizes of 10, 30, 60, 90 frames and report how ρ changes with window size. This tells us the optimal prediction granularity.

---

## Step 4: High-GT filter

Repeat step 2 but only on frames where num_gt >= 5 (or whatever threshold gives ~50% of frames). If the signal only exists when there are enough objects to create variance, that's important to know.

```python
df_dense = df[df['num_gt'] >= 5].copy()
# Rerun per-frame Spearman on df_dense for both targets
```

---

## Output

A single summary table with 4 conditions × 15 features × 2 targets = 120 rows:

| condition | target | feature | within_seq_rho |
|-----------|--------|---------|----------------|
| per_frame | fn_nano | edge_density | 0.xxx |
| per_frame | recoverable_fn | edge_density | 0.xxx |
| window_30 | fn_nano | edge_density | 0.xxx |
| window_30 | recoverable_fn | edge_density | 0.xxx |
| high_gt | fn_nano | edge_density | 0.xxx |
| high_gt | recoverable_fn | edge_density | 0.xxx |

Plus a simple PDF or printed summary showing:
1. recoverable_fn distribution on DETRAC
2. Best ρ for each condition (which condition + target gives the strongest signal?)
3. Windowed ρ vs window size curve for the top 3 features
4. Recommendation: which target × granularity to proceed with

## Important

- This is all post-processing on existing data. No YOLO re-runs, no new feature extraction.
- Find the DETRAC data files — likely in the spatial_experiments directory or similar.
- The frame column might be called frame_idx, frame_num, or similar. The sequence column might be called sequence, seq_name, video_name, or similar. Adapt column names to match what exists.
- If tp_medium doesn't exist directly, compute it: tp_medium = num_gt - fn_medium (where fn_medium is medium's false negatives).
