# File: C:\vehicle_health_factory\train_factory.py
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

# Connect to src
sys.path.append('src')
from data_loader import IndustrialDataLoader
from models import LSTMAutoencoder

# --- CONFIGURATION ---
MODULES = ["engine", "body", "battery", "transmission", "tyre"]
DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"
EPOCHS = 50       
BATCH_SIZE = 32
WINDOW_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_threshold_and_stats(model, val_loader):
    model.eval()
    errors = []
    criterion = nn.L1Loss(reduction='none') 

    print("      📊 Calibrating Model Statistics...")
    with torch.no_grad():
        for x in val_loader:
            # FIX: Ensure tuple unpacking if loader returns (x,)
            if isinstance(x, list) or isinstance(x, tuple):
                x = x[0]
            
            x = x.to(DEVICE)
            out = model(x)
            loss = torch.mean(criterion(out, x), dim=(1, 2)) 
            errors.extend(loss.cpu().numpy())

    errors = np.array(errors)
    threshold = np.percentile(errors, 99.5)
    
    stats = {
        "threshold": float(threshold),
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "error_max": float(np.max(errors)),
        "error_min": float(np.min(errors))
    }
    
    print(f"      ✅ Stats: Mean={stats['error_mean']:.4f}, Max={stats['error_max']:.4f}")
    print(f"      🎯 Threshold Set: {threshold:.4f}")
    return stats, errors

def train_module(module_name):
    print(f"\n🏭 PROCESSING MODULE: {module_name.upper()}")
    
    csv_file = os.path.join(DATA_DIR, f"{module_name}.csv")
    save_path = os.path.join(ARTIFACT_DIR, module_name)
    os.makedirs(save_path, exist_ok=True)
    
    if not os.path.exists(csv_file):
        print(f"❌ Missing {csv_file}")
        return

    # 1. Load & Engineer
    loader = IndustrialDataLoader(csv_file, window_size=WINDOW_SIZE)
    train_scaled, val_scaled, feature_cols = loader.load_and_prep()
    loader.save_scaler(os.path.join(save_path, "scaler.pkl"))

    # 2. Tensor Prep
    X_train = loader.create_sliding_windows(train_scaled)
    X_val = loader.create_sliding_windows(val_scaled)
    
    # --- CRITICAL FIX: Convert to Float32 Tensor explicitly ---
    tensor_train = torch.tensor(X_train, dtype=torch.float32)
    tensor_val = torch.tensor(X_val, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(tensor_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(TensorDataset(tensor_val), batch_size=BATCH_SIZE)

    # 3. Train LSTM
    input_dim = X_train.shape[2]
    model = LSTMAutoencoder(input_dim, WINDOW_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"      🚀 Training LSTM ({input_dim} features)...")
    history = {'train': [], 'val': []}
    
    for epoch in tqdm(range(EPOCHS), leave=False):
        model.train()
        train_loss = 0
        for x in train_loader:
            # FIX: TensorDataset returns a tuple (tensor,), we need just tensor
            x = x[0].to(DEVICE)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                x = x[0].to(DEVICE)
                out = model(x)
                val_loss += criterion(out, x).item()
        
        history['train'].append(train_loss / len(train_loader))
        history['val'].append(val_loss / len(val_loader))

    torch.save(model.state_dict(), f"{save_path}/lstm_model.pt")

    # 4. Calibration & Stats
    calib_stats, val_errors = calculate_threshold_and_stats(model, val_loader)

    # 5. Save Metadata
    model_meta = {
        "calibration": calib_stats,
        "window_size": WINDOW_SIZE,
        "input_dim": input_dim,
        "features": feature_cols
    }
    with open(f"{save_path}/model_meta.json", "w") as f:
        json.dump(model_meta, f, indent=4)

    # 6. Train Isolation Forest
    print("      🌲 Training Isolation Forest...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    if_model = IsolationForest(contamination=0.01, n_jobs=-1, random_state=42)
    if_model.fit(X_train_flat)
    joblib.dump(if_model, f"{save_path}/iforest.pkl")

    # 7. Evidence Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'], label='Val')
    plt.title("Loss Curve")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(val_errors, bins=50, alpha=0.7, color='orange')
    plt.axvline(calib_stats['threshold'], color='red', linestyle='dashed', label='Threshold')
    plt.title("Anomaly Distribution")
    plt.legend()
    
    plt.savefig(f"{save_path}/training_report.png")
    plt.close()

    print(f"✅ {module_name.upper()} Complete.")

if __name__ == "__main__":
    print(f"🚀 Factory Started on {DEVICE}")
    for mod in MODULES:
        train_module(mod)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\n🎉 ALL MODULES TRAINED. Artifacts Ready.")