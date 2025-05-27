import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample, butter, filtfilt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, brier_score_loss
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import csv

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds.")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# -----------------------------
# 1. CONFIGURATION
# -----------------------------
PTBXL_PATH = './'  # Set to your PTB-XL root
ANNOTATION_FILE = os.path.join(PTBXL_PATH, 'ptbxl_database.csv')
# SCP_STATEMENTS_FILE = os.path.join(PTBXL_PATH, 'scp_statements.csv')

# Define the top 5 SCP codes to use for training
TOP5_SCP_CODES = ['NORM', 'IMI', 'ASMI', 'LVH', 'NDT']
# NORM: 9514 records
# IMI: 2676 records
# ASMI: 2357 records
# LVH: 2132 records
# NDT: 1825 records
# -----------------------------
# 2. DATA LOADING & PREPROCESSING
# -----------------------------
# Accept user input for sample rate
try:
    user_input = input("Enter target sample frequency (Hz, e.g., 100, 250, 500) [default 500]: ").strip()
    if user_input:
        user_sample_rate = int(user_input)
        if user_sample_rate <= 0 or user_sample_rate > 500:
            print("Sample frequency must be positive and less than or equal to 500. Using default 500.")
            user_sample_rate = 500
    else:
        user_sample_rate = 500
except Exception as e:
    print(f"Invalid input ({e}), using default 500.")
    user_sample_rate = 500

# After user_sample_rate is set
window_size = user_sample_rate * 10  # 10 seconds
print("Due to limitation of hardware, the training is only focus on top 5 SCP codes, which are NORM, IMI, ASMI, LVH, NDT.")
def load_lead2_signal(fname, target_fs=None):
    # Always load at raw 500 Hz
    sig, fields = wfdb.rdsamp(os.path.join(PTBXL_PATH, fname))
    lead2 = sig[:, 1].astype(float)
    orig_fs = fields['fs']
    if orig_fs != 500:
        raise ValueError("Raw data is expected to be 500 Hz!")
    if target_fs is None:
        target_fs = user_sample_rate
    if target_fs != 500:
        lead2 = resample(lead2, int(len(lead2) * target_fs / 500))
    # Band-pass filter (0.5–40 Hz)
    nyq = 0.5 * target_fs
    b, a = butter(3, [0.5/nyq, 40/nyq], btype='bandpass')
    filtered = filtfilt(b, a, lead2)
    return filtered

# Load metadata
meta = pd.read_csv(ANNOTATION_FILE, index_col='ecg_id')

# Parse SCP_CODEs for each record
meta['scp_codes_parsed'] = meta.scp_codes.apply(eval)

def extract_top5_scp_codes(scp_dict):
    return set([k for k, v in scp_dict.items() if v > 0.0 and k in TOP5_SCP_CODES])
meta['scp_codes_set'] = meta['scp_codes_parsed'].apply(extract_top5_scp_codes)

# Filter out records with no positive label among the top 5 codes
meta = meta[meta['scp_codes_set'].apply(len) > 0]

# MultiLabelBinarizer for top 5 SCP codes
mlb = MultiLabelBinarizer(classes=TOP5_SCP_CODES)
mlb.fit(meta['scp_codes_set'])

# After meta is defined and filtered
train_df = meta[meta.strat_fold.between(1, 8)]
val_df   = meta[meta.strat_fold == 9]
test_df  = meta[meta.strat_fold == 10]

# -----------------------------
# 3. DATASET & DATALOADER
# -----------------------------
class LeadIISCPDataset(Dataset):
    def __init__(self, df, window_size=5000, target_fs=500):
        self.df = df
        self.window_size = window_size
        self.target_fs = target_fs
        self.labels = mlb.transform(df['scp_codes_set'])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = load_lead2_signal(row.filename_hr, self.target_fs)
        # Pad or truncate to window_size
        if len(sig) < self.window_size:
            pad = np.zeros(self.window_size)
            pad[:len(sig)] = sig
            sig = pad
        else:
            sig = sig[:self.window_size]
        x = torch.tensor(sig.copy(), dtype=torch.float32).unsqueeze(0)  # (1, L)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# -----------------------------
# 4. MODEL
# -----------------------------
class SCPCodeClassifier(nn.Module):
    def __init__(self, window_size, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        # Dynamically compute the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, window_size)
            n_flat = self.features(dummy).view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(n_flat, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# 5. TRAINING & EVALUATION
# -----------------------------
def compute_metrics(y_true, y_pred, y_prob, k=3):
    metrics = {}
    # AUROC, AUPRC (macro)
    y_true_sum = y_true.sum(axis=0)
    valid_classes = y_true_sum > 0
    if valid_classes.any():
        auroc = roc_auc_score(y_true[:, valid_classes], y_prob[:, valid_classes], average='macro')
    else:
        auroc = float('nan')
    try:
        metrics['AUROC'] = auroc
    except:
        metrics['AUROC'] = np.nan
    try:
        metrics['AUPRC'] = average_precision_score(y_true, y_prob, average='macro')
    except:
        metrics['AUPRC'] = np.nan
    # F1-micro, F1-macro
    metrics['F1-micro'] = f1_score(y_true, y_pred, average='micro')
    metrics['F1-macro'] = f1_score(y_true, y_pred, average='macro')
    # Accuracy@k
    topk = np.argsort(-y_prob, axis=1)[:, :k]
    acc_k = np.mean([y_true[i, topk[i]].any() for i in range(len(y_true))])
    metrics[f'Accuracy@{k}'] = acc_k
    # Brier score (mean over all labels)
    metrics['Brier'] = np.mean([brier_score_loss(y_true[:,i], y_prob[:,i]) for i in range(y_true.shape[1])])
    return metrics

# -----------------------------
# 6. MAIN TRAINING LOOP (EXAMPLE)
# -----------------------------
def train_scp_classifier():
    # Use the stratified splits defined above
    batch_size = 32
    n_classes = len(TOP5_SCP_CODES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Number of rows in train set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")
    print(f"Number of rows in test set: {len(test_df)}")

    train_ds = LeadIISCPDataset(train_df, window_size, target_fs=user_sample_rate)
    val_ds = LeadIISCPDataset(val_df, window_size, target_fs=user_sample_rate)
    test_ds = LeadIISCPDataset(test_df, window_size, target_fs=user_sample_rate)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SCPCodeClassifier(window_size, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    best_val_f1 = 0
    best_state = None
    patience = 7  # Early stopping patience
    patience_counter = 0
    for epoch in range(1, 31):  # Train for up to 30 epochs
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
        # Validation
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_prob.append(probs)
                y_true.append(y.numpy())
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        y_pred = (y_prob > 0.5).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        print(f"Epoch {epoch:02d}  Val F1-micro: {metrics['F1-micro']:.4f}  AUROC: {metrics['AUROC']:.4f}")
        # print("Val label counts:", np.sum(mlb.transform(val_df['scp_codes_set']), axis=0))
        if metrics['F1-micro'] > best_val_f1:
            best_val_f1 = metrics['F1-micro']
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best val F1-micro: {best_val_f1:.4f}.")
                break
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    # Test evaluation
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.append(probs)
            y_true.append(y.numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = (y_prob > 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return model, metrics, mlb

# -----------------------------
# 7. PREDICTION FUNCTION
# -----------------------------
def predict_scp_codes(model, ecg_snippet, sample_rate, mlb, window_size=None, device=None):
    if window_size is None:
        window_size = user_sample_rate * 10
    if sample_rate != user_sample_rate:
        ecg_snippet = resample(ecg_snippet, int(len(ecg_snippet) * user_sample_rate / sample_rate))
    nyq = 0.5 * user_sample_rate
    b, a = butter(3, [0.5/nyq, 40/nyq], btype='bandpass')
    filtered = filtfilt(b, a, ecg_snippet)
    if len(filtered) < window_size:
        pad = np.zeros(window_size)
        pad[:len(filtered)] = filtered
        filtered = pad
    else:
        filtered = filtered[:window_size]
    x = torch.tensor(filtered.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    scp_codes = mlb.classes_
    pred_dict = {scp: float(prob) for scp, prob in zip(scp_codes, probs)}
    return pred_dict

# -----------------------------
# 8. USAGE EXAMPLE (Uncomment to run)
# -----------------------------
model, metrics, mlb = train_scp_classifier()
# Example: load a real test sample
print("Writing results to scp_code_predictions.csv")
prediction_rows = []
for ecg_id, row in test_df.iterrows():
    ecg_snippet = load_lead2_signal(row.filename_hr, target_fs=user_sample_rate)
    sample_rate = user_sample_rate
    pred = predict_scp_codes(model, ecg_snippet, sample_rate, mlb, window_size=window_size)
    pred_filtered = {k: v for k, v in pred.items() if v > 0.5}
    # print(f"Sample {ecg_id} ({row.filename_hr}): {pred_filtered}")
    # Prepare row for CSV
    code_str = "; ".join([f"{k}:{v*100:.1f}" for k, v in pred_filtered.items()])
    prediction_rows.append({
        'ecg_id': ecg_id,
        'filename_hr': row.filename_hr,
        'predicted_codes': code_str
    })
# Save to CSV
with open('scp_code_predictions.csv', 'w', newline='') as csvfile:
    fieldnames = ['ecg_id', 'filename_hr', 'predicted_codes']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in prediction_rows:
        writer.writerow(r)


