# DETRAC: Dirichlet LSTM Training for Scene Complexity Classification

## Context

We have validated that spatial features predict fn_nano on DETRAC:
- 138,252 frames across 100 sequences
- 15 spatial features extracted per frame
- Windowed fn_nano (window=30) gives best Spearman ρ = 0.377 (FAST_keypoint_count)
- Elastic Net R² = 0.313 confirms linear signal exists
- Now: train an evidential LSTM that classifies windows as easy/hard

## Data

**Spatial features CSV:** `spatial_experiments/spatial_experiments_all_features.csv`
- Columns: sequence, frame_id, weather, fn, num_gt, + 15 feature columns

**Detection results CSV:** `res_experiment/detrac_results.csv`
- Columns: sequence, frame_id, model, imgsz, tp, fp, fn, recall, precision, f1, mean_iou, weather, num_gt, ...
- Models: nano_320, medium_640 — 2 rows per frame
- Extract fn_nano from the nano_320 rows (the `fn` column when model == "nano_320")

**15 Feature columns:**
```python
FEATURE_COLS = [
    "top_laplacian_variance", "top_small_component_count", "top_gradient_magnitude_std",
    "mid_laplacian_variance", "mid_small_component_count", "mid_gradient_magnitude_std",
    "bot_laplacian_variance", "bot_small_component_count", "bot_gradient_magnitude_std",
    "global_total_small_components", "global_overall_edge_density",
    "global_FAST_keypoint_count", "global_mean_brightness",
    "global_dark_channel_mean", "global_temporal_diff_mean",
]
```

---

## Step 1: Data Preparation

### 1a. Merge features with fn_nano

```python
import pandas as pd
import numpy as np

# Load
feat_df = pd.read_csv("spatial_experiments/spatial_experiments_all_features.csv")
det_df = pd.read_csv("res_experiment/detrac_results.csv")

# Get fn_nano per frame
nano_df = det_df[det_df["model"] == "nano_320"][["sequence", "frame_id", "fn"]].rename(columns={"fn": "fn_nano"})

# Merge
df = feat_df.merge(nano_df, on=["sequence", "frame_id"])
```

### 1b. Create windows

Non-overlapping 30-frame windows per sequence. Each window becomes one training sample.

```python
WINDOW_SIZE = 30

windows = []
for seq, group in df.groupby("sequence"):
    group = group.sort_values("frame_id").reset_index(drop=True)
    n_windows = len(group) // WINDOW_SIZE
    
    for w in range(n_windows):
        chunk = group.iloc[w * WINDOW_SIZE : (w + 1) * WINDOW_SIZE]
        
        # Store raw per-frame features as a 2D array (30 x 15) — LSTM input
        frame_features = chunk[FEATURE_COLS].values  # shape: (30, 15)
        
        # Window-level label
        mean_fn = chunk["fn_nano"].mean()
        
        windows.append({
            "sequence": seq,
            "window_idx": w,
            "features": frame_features,      # (30, 15) numpy array
            "mean_fn_nano": mean_fn,
        })

print(f"Total windows: {len(windows)}")
# Expect ~4,600 windows
```

### 1c. Binary classification labels

```python
# Use median as threshold for balanced classes
threshold = np.median([w["mean_fn_nano"] for w in windows])
print(f"Threshold (median): {threshold:.2f}")

for w in windows:
    w["label"] = 0 if w["mean_fn_nano"] < threshold else 1  # 0=easy, 1=hard

# Report class balance
labels = [w["label"] for w in windows]
print(f"Easy: {labels.count(0)}, Hard: {labels.count(1)}")
```

Also try a cost-justified threshold: fn_nano >= 3 means switching recovers enough objects to justify the energy. Report both and pick whichever gives more meaningful classes.

### 1d. Train/val/test split BY SEQUENCE

Critical: split by sequence, not by window. No frames from the same video in both train and test.

```python
from sklearn.model_selection import GroupShuffleSplit

sequences = [w["sequence"] for w in windows]
unique_seqs = list(set(sequences))
np.random.seed(42)
np.random.shuffle(unique_seqs)

n = len(unique_seqs)
train_seqs = unique_seqs[:int(0.7 * n)]
val_seqs = unique_seqs[int(0.7 * n):int(0.85 * n)]
test_seqs = unique_seqs[int(0.85 * n):]

train_windows = [w for w in windows if w["sequence"] in train_seqs]
val_windows = [w for w in windows if w["sequence"] in val_seqs]
test_windows = [w for w in windows if w["sequence"] in test_seqs]

print(f"Train: {len(train_windows)} windows from {len(train_seqs)} sequences")
print(f"Val: {len(val_windows)} windows from {len(val_seqs)} sequences")
print(f"Test: {len(test_windows)} windows from {len(test_seqs)} sequences")
```

### 1e. Normalize features

Per-feature StandardScaler fit on train only, applied to val/test.

```python
from sklearn.preprocessing import StandardScaler

# Flatten train features to fit scaler
train_flat = np.vstack([w["features"] for w in train_windows])  # (N*30, 15)
scaler = StandardScaler().fit(train_flat)

# Apply to all splits
for split in [train_windows, val_windows, test_windows]:
    for w in split:
        w["features"] = scaler.transform(w["features"])  # still (30, 15)
```

---

## Step 2: PyTorch Dataset and DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader

class WindowDataset(Dataset):
    def __init__(self, windows):
        self.X = torch.FloatTensor(np.array([w["features"] for w in windows]))  # (N, 30, 15)
        self.y = torch.LongTensor([w["label"] for w in windows])                # (N,)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = WindowDataset(train_windows)
val_ds = WindowDataset(val_windows)
test_ds = WindowDataset(test_windows)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
```

---

## Step 3: Dirichlet LSTM Model

The model outputs Dirichlet concentration parameters (alpha), NOT softmax probabilities.
This is the evidential deep learning approach from Sensoy et al. 2018.

```python
import torch.nn as nn

class DirichletLSTM(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        # Output: alpha = softplus(fc_output) + 1
        # Ensures alpha > 1 (concentration parameters)
    
    def forward(self, x):
        # x: (batch, seq_len=30, features=15)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        logits = self.fc(last_hidden)     # (batch, num_classes)
        alpha = torch.nn.functional.softplus(logits) + 1  # alpha > 1
        return alpha

model = DirichletLSTM()
```

---

## Step 4: Dirichlet Loss (Sensoy et al. 2018 — Type II Maximum Likelihood)

```python
def dirichlet_loss(alpha, target, epoch, num_epochs, annealing_step=10):
    """
    Type II Maximum Likelihood loss for Dirichlet networks.
    
    alpha: (batch, num_classes) — Dirichlet concentration parameters
    target: (batch,) — class indices
    """
    num_classes = alpha.shape[1]
    
    # One-hot encode target
    y_one_hot = torch.zeros_like(alpha).scatter_(1, target.unsqueeze(1), 1)
    
    # Dirichlet strength
    S = alpha.sum(dim=1, keepdim=True)  # (batch, 1)
    
    # Expected probabilities
    p = alpha / S  # (batch, num_classes)
    
    # Type II ML loss (Eq. 3 from Sensoy et al.)
    # = sum of y_k * (digamma(S) - digamma(alpha_k))
    loss_ml = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    
    # KL divergence regularizer — penalize evidence on wrong classes
    # Annealed: starts at 0, linearly increases to 1
    annealing_coef = min(1.0, epoch / annealing_step)
    
    # Remove evidence for correct class
    alpha_tilde = y_one_hot + (1 - y_one_hot) * (alpha - 1) + 1
    
    # KL(Dir(alpha_tilde) || Dir(1, 1, ..., 1))
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl = torch.lgamma(S_tilde.squeeze()) - torch.lgamma(torch.tensor(float(num_classes))) \
         - torch.sum(torch.lgamma(alpha_tilde), dim=1) \
         + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1)
    
    loss = loss_ml + annealing_coef * kl
    return loss.mean()
```

---

## Step 5: Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
num_epochs = 100

best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 20

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        alpha = model(X_batch)
        loss = dirichlet_loss(alpha, y_batch, epoch, num_epochs)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(y_batch)
    train_loss /= len(train_ds)
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            alpha = model(X_batch)
            loss = dirichlet_loss(alpha, y_batch, epoch, num_epochs)
            val_loss += loss.item() * len(y_batch)
    val_loss /= len(val_ds)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_dirichlet_lstm.pt")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## Step 6: Evaluation

Load best model and evaluate on test set.

### 6a. Predictions and uncertainty extraction

```python
model.load_state_dict(torch.load("best_dirichlet_lstm.pt"))
model.eval()

all_alphas = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        alpha = model(X_batch)
        all_alphas.append(alpha.numpy())
        all_labels.append(y_batch.numpy())

alphas = np.concatenate(all_alphas)       # (N_test, 2)
labels = np.concatenate(all_labels)       # (N_test,)

# Dirichlet outputs
S = alphas.sum(axis=1)                    # Dirichlet strength (total evidence)
probs = alphas / S[:, None]               # Expected probabilities
predictions = probs.argmax(axis=1)        # Predicted class

# Uncertainty metrics
epistemic_uncertainty = 2.0 / S           # Higher = less evidence = more uncertain
                                           # (num_classes / S)
aleatoric_uncertainty = -np.sum(probs * np.log(probs + 1e-8), axis=1)  # Entropy of expected probs
```

### 6b. Classification metrics

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(labels, predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(labels, predictions, target_names=["easy", "hard"]))
print(f"\nConfusion Matrix:")
print(confusion_matrix(labels, predictions))
```

### 6c. Uncertainty quality — do misclassified windows have higher uncertainty?

```python
correct_mask = predictions == labels
wrong_mask = ~correct_mask

print(f"\nUncertainty Analysis:")
print(f"  Mean epistemic uncertainty (correct):   {epistemic_uncertainty[correct_mask].mean():.4f}")
print(f"  Mean epistemic uncertainty (incorrect): {epistemic_uncertainty[wrong_mask].mean():.4f}")
print(f"  Mean aleatoric uncertainty (correct):   {aleatoric_uncertainty[correct_mask].mean():.4f}")
print(f"  Mean aleatoric uncertainty (incorrect): {aleatoric_uncertainty[wrong_mask].mean():.4f}")

# Key test: if uncertainty is higher for wrong predictions, the evidential approach works
# This means the MPC can use uncertainty to be conservative on unfamiliar scenes
```

### 6d. Calibration — does P(hard) = 0.7 mean hard 70% of the time?

```python
# Bin predictions by P(hard) and check actual fraction
p_hard = probs[:, 1]
bins = np.linspace(0, 1, 11)
for i in range(len(bins) - 1):
    mask = (p_hard >= bins[i]) & (p_hard < bins[i+1])
    if mask.sum() > 0:
        actual = labels[mask].mean()
        predicted = p_hard[mask].mean()
        print(f"  P(hard) in [{bins[i]:.1f}, {bins[i+1]:.1f}): n={mask.sum():4d}, predicted={predicted:.3f}, actual={actual:.3f}")
```

### 6e. Baselines

```python
# Random baseline
random_acc = max(labels.mean(), 1 - labels.mean())
print(f"\nBaselines:")
print(f"  Always majority class: {random_acc:.4f}")
print(f"  Always easy (always nano): {(labels == 0).mean():.4f}")
print(f"  Always hard (always medium): {(labels == 1).mean():.4f}")

# Single-feature threshold baseline (FAST keypoints)
# Use the best single feature with optimal threshold on train set
print(f"  Dirichlet LSTM: {accuracy:.4f}")
```

---

## Step 7: PDF Report

Generate a PDF with the following pages:

### Page 1: Training Overview
- Loss curves (train and val over epochs)
- Data split summary (sequences, windows per split)
- Class balance histogram
- Threshold used for easy/hard

### Page 2: Test Results
- Confusion matrix heatmap
- Classification report (precision, recall, F1)
- ROC curve with AUC
- Accuracy vs baselines bar chart

### Page 3: Uncertainty Quality
- Histogram of epistemic uncertainty: correct vs incorrect predictions (overlaid)
- Histogram of aleatoric uncertainty: correct vs incorrect predictions (overlaid)
- Scatter: P(hard) vs epistemic uncertainty, colored by correct/incorrect
- Box plot: uncertainty by prediction outcome (correct-easy, correct-hard, wrong-easy, wrong-hard)

### Page 4: Calibration & MPC Interface
- Calibration plot (predicted P(hard) vs actual fraction hard)
- Decision regions: plot showing P(hard) vs uncertainty with suggested MPC thresholds
- Per-sequence accuracy (bar chart sorted by accuracy, shows which sequences are hardest to classify)
- Summary text with key numbers

---

## Output Files

- `dirichlet_lstm_results.pdf` — 4-page report
- `best_dirichlet_lstm.pt` — saved model weights
- `dirichlet_lstm_results.csv` — per-window predictions with alpha, probs, uncertainty, label

Save all to the same directory as the training script.

## Verification

- Test accuracy should beat majority-class baseline
- Incorrect predictions should have higher epistemic uncertainty than correct ones
- Calibration should be reasonable (not wildly off-diagonal)
- Loss should converge (not diverge or flat-line)
- Class balance should be roughly 50/50 if using median threshold
- No NaN in alphas (softplus + 1 prevents this, but verify)
