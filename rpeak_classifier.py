import pandas as pd
import numpy as np
import wfdb
from scipy.signal import resample, butter, filtfilt, find_peaks
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prompt user for sample frequency
try:
    user_input = input("Enter target sample frequency (Hz, e.g., 100, 250, 500) [default 500]: ").strip()
    if user_input:
        target_fs = int(user_input)
        if target_fs <= 0 or target_fs > 500:
            print("Sample frequency must be positive and less than 500. Using default 500.")
            target_fs = 500
    else:
        target_fs = 500
except Exception as e:
    print(f"Invalid input ({e}), using default 500.")
    target_fs = 500

    
fs = target_fs
# ─────────────────────────────────────────────────────────────────────────────
# 1) READ + SPLIT METADATA
# ─────────────────────────────────────────────────────────────────────────────
path = "./"  # adjust this to your ptbxl root
meta = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")

# folds 1–8 = train, 9 = val, 10 = test
# train_df = meta[meta.strat_fold.between(1, 8)].sample(frac=0.1, random_state=42)
# val_df   = meta[meta.strat_fold == 9].sample(frac=0.1, random_state=42)
# test_df  = meta[meta.strat_fold == 10].sample(frac=0.1, random_state=42)
train_df = meta[meta.strat_fold.between(1, 8)]
val_df   = meta[meta.strat_fold == 9]
test_df  = meta[meta.strat_fold == 10]

print(f"Number of rows in train set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2) HELPERS: load & preprocess signals + annotations
# ─────────────────────────────────────────────────────────────────────────────
def load_filtered_signal(fname, target_fs=target_fs):
    """
    1) wfdb.rdsamp reads the .dat/.hea pair
    2) select Lead II (channel index 1)
    3) resample → target_fs
    4) butterworth band-pass 0.5–40 Hz
    """
    sig, fields = wfdb.rdsamp(path + fname)
    lead2 = sig[:, 1].astype(float)
    orig_fs = fields["fs"]
    
    # resample
    if orig_fs != target_fs:
        lead2 = resample(lead2, int(len(lead2) * target_fs / orig_fs))

    # band-pass
    nyq = 0.5 * target_fs
    b, a = butter(3, [0.5/nyq, 40/nyq], btype="bandpass")
    return filtfilt(b, a, lead2)

# Pan–Tompkins QRS detection
def detect_qrs_pantompkins(sig, fs=100):
    """
    Implements a simplified Pan–Tompkins QRS detection pipeline:
    1. Band-pass filter (5–15 Hz)
    2. Differentiate
    3. Square
    4. Moving-window integration
    5. Peak detection
    """
    # 1. Band-pass filter (5–15 Hz)
    nyq = 0.5 * fs
    b, a = butter(3, [5/nyq, 15/nyq], btype="bandpass")
    filtered = filtfilt(b, a, sig)

    # 2. Differentiate
    diff = np.ediff1d(filtered, to_begin=0)

    # 3. Square
    squared = diff ** 2

    # 4. Moving-window integration (window ~150 ms)
    win_size = int(0.15 * fs)
    mwa = np.convolve(squared, np.ones(win_size)/win_size, mode='same')

    # 5. Peak detection
    min_dist = int(0.25 * fs)  # 250 ms minimum distance between peaks
    threshold = np.mean(mwa) + 0.5 * np.std(mwa)  # adaptive threshold
    peaks, _ = find_peaks(mwa, distance=min_dist, height=threshold)
    return peaks, mwa

def refine_rpeaks(sig, rpeaks, search_radius=10):
    """Refine R-peak locations by searching for the local maximum in the raw ECG signal."""
    refined = []
    for p in rpeaks:
        start = max(0, p - search_radius)
        end = min(len(sig), p + search_radius)
        local_max = np.argmax(sig[start:end]) + start
        refined.append(local_max)
    return np.array(refined)

def extract_windows(sig, rpeaks, window_size=128, neg_ratio=5):
    
    half = window_size // 2
    X, y = [], []
    # Positive samples (centered on R-peaks)
    for p in rpeaks:
        if p - half >= 0 and p + half < len(sig):
            X.append(sig[p-half:p+half])
            y.append(1)
    # Negative samples (random, not near R-peaks)
    negs = []
    exclusion = np.zeros(len(sig), dtype=bool)
    for p in rpeaks:
        exclusion[max(0, p-half):min(len(sig), p+half)] = True
    candidates = np.where(~exclusion)[0]
    np.random.shuffle(candidates)
    for idx in candidates:
        if idx - half >= 0 and idx + half < len(sig):
            negs.append(idx)
        if len(negs) >= len(rpeaks) * neg_ratio:
            break
    for n in negs:
        X.append(sig[n-half:n+half])
        y.append(0)
    return np.array(X), np.array(y)

# ─────────────────────────────────────────────────────────────────────────────
# 3) BUILD DATASETS FOR CLASSIFICATION (Lead II, refined peaks)
# ─────────────────────────────────────────────────────────────────────────────
window_size = 128
neg_ratio = 5
fs = target_fs

def build_set(df, window_size=128, neg_ratio=5, fs=target_fs):
    X, y = [], []
    for fname in df.filename_hr:
        sig = load_filtered_signal(fname, fs)
        rpeaks, _ = detect_qrs_pantompkins(sig, fs)
        rpeaks_refined = refine_rpeaks(sig, rpeaks)
        X1, y1 = extract_windows(sig, rpeaks_refined, window_size, neg_ratio)
        X.append(X1)
        y.append(y1)
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

print("Extracting train set windows...")
X_train, y_train = build_set(train_df, window_size, neg_ratio, fs)
print("Extracting val set windows...")
X_val, y_val = build_set(val_df, window_size, neg_ratio, fs)
print("Extracting test set windows...")
X_test, y_test = build_set(test_df, window_size, neg_ratio, fs)

# ─────────────────────────────────────────────────────────────────────────────
# 4) PYTORCH DATASET/LOADER
# ─────────────────────────────────────────────────────────────────────────────
class RPeakWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y[i], dtype=torch.float32)
        return x, y

train_ds = RPeakWindowDataset(X_train, y_train)
val_ds = RPeakWindowDataset(X_val, y_val)
test_ds = RPeakWindowDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# ─────────────────────────────────────────────────────────────────────────────
# 5) CLASSIFIER MODEL, TRAINING, EVAL
# ─────────────────────────────────────────────────────────────────────────────
class SimpleClassifier(nn.Module):
    def __init__(self, wsize=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear((wsize//4)*32, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = SimpleClassifier(wsize=window_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCELoss()

def compute_f1(model, loader, threshold=0.8):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).cpu().numpy()
            y_pred.extend((out > threshold).astype(int))
            y_true.extend(y.numpy().astype(int))
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return f1

history = {"train_loss": [], "val_loss": [], "val_f1": []}
best_f1 = 0
best_epoch = 0
patience = 5
patience_counter = 0
best_model_state = None
for epoch in range(1, 31):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(train_ds)
    history["train_loss"].append(avg_loss)
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            val_loss += loss.item() * x.size(0)
    avg_val_loss = val_loss / len(val_ds)
    history["val_loss"].append(avg_val_loss)
    # Compute val F1
    val_f1 = compute_f1(model, val_loader, threshold=0.8)
    history["val_f1"].append(val_f1)
    print(f"Epoch {epoch:02d}  train_loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  val_f1={val_f1:.4f}")
    # Early stopping
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        best_model_state = deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best val F1: {best_f1:.4f} at epoch {best_epoch}.")
            break
# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Plot loss and val F1
plot_dir = './plot'
os.makedirs(plot_dir, exist_ok=True)
fig, ax1 = plt.subplots(1,1, figsize=(8,4))
ax1.plot(history["train_loss"], label="Train Loss")
ax1.plot(history["val_loss"], label="Val Loss")
ax2 = ax1.twinx()
ax2.plot(history["val_f1"], label="Val F1", color='g')
ax1.set_title("Loss and Validation F1")
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.savefig(os.path.join(plot_dir, 'classifier_loss_valf1_plot.png'))
plt.close()

# Evaluate on test set
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x).cpu().numpy()
        y_pred.extend((out > 0.8).astype(int))
        y_true.extend(y.numpy().astype(int))
pr, re, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
print(f"Test Precision: {pr:.3f}  Recall: {re:.3f}  F1: {f1:.3f}")

# --- Compare refined peaks vs classifier peaks on 5 test examples ---
for i, example in enumerate(test_df.filename_hr.iloc[:5]):
    sig = load_filtered_signal(example, fs)
    # 1. Refined Pan–Tompkins peaks
    rpeaks_ref, _ = detect_qrs_pantompkins(sig, fs)
    rpeaks_refined = refine_rpeaks(sig, rpeaks_ref)
    # 2. Classifier-based peaks
    half = window_size // 2
    pred_scores = []
    positions = []
    model.eval()
    with torch.no_grad():
        for center in range(half, len(sig) - half):
            #window = torch.tensor(sig[center-half:center+half].copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            segment = sig[center-half:center+half]
            if len(segment) != window_size:
                continue  # skip this window if it's not the right size
            window = torch.tensor(segment.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            score = model(window).item()
            pred_scores.append(score)
            positions.append(center)
    pred_scores = np.array(pred_scores)
    positions = np.array(positions)
    # Use the classifier scores as a 1D signal, find local maxima above threshold, and enforce minimum distance
    peaks, _ = find_peaks(pred_scores, height=0.85, distance=int(0.3*fs))
    predicted_peaks = positions[peaks]
    # 3. Plot
    plt.figure(figsize=(12, 4))
    plt.plot(sig, label="Lead II")
    plt.scatter(rpeaks_refined, sig[rpeaks_refined], c='g', s=30, label="Refined R-peaks (reference)", marker='o')
    plt.scatter(predicted_peaks, sig[predicted_peaks], c='r', s=15, label="Classifier R-peaks", marker='x')
    plt.legend()
    plt.title(f"Comparison of Refined vs Classifier R-peaks (Example {i+1})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'compare_refined_vs_classifier_rpeaks_{i+1}.png'))
    plt.close()

# --- Output classifier R-peak results for test set to CSV ---
print("Extracting classifier R-peak results for test set...")
results = []
for fname in test_df.filename_hr:
    sig = load_filtered_signal(fname, fs)
    half = window_size // 2
    pred_scores = []
    positions = []
    model.eval()
    with torch.no_grad():
        for center in range(half, len(sig) - half):
            segment = sig[center-half:center+half]
            if len(segment) != window_size:
                continue  # skip this window if it's not the right size
            window = torch.tensor(segment.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            score = model(window).item()
            pred_scores.append(score)
            positions.append(center)
    pred_scores = np.array(pred_scores)
    positions = np.array(positions)
    peaks, _ = find_peaks(pred_scores, height=0.85, distance=int(0.3*fs))
    r_peaks_samples = positions[peaks]
    r_peaks_seconds = r_peaks_samples / fs
    results.append({
        'filename': fname,
        #'r_peaks_samples': list(r_peaks_samples),
        'r_peaks_seconds': list(r_peaks_seconds)
    })
# Write to CSV
with open('classifier_rpeaks_testset.csv', 'w', newline='') as csvfile:
    fieldnames = ['filename', 'r_peaks_seconds']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        # Convert lists to string for CSV
        #row['r_peaks_samples'] = ','.join(map(str, row['r_peaks_samples']))
        row['r_peaks_seconds'] = ','.join(map(str, row['r_peaks_seconds']))
        writer.writerow(row)
print('Classifier R-peak results for test set saved to classifier_rpeaks_testset.csv')