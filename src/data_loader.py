# File: C:\vehicle_health_factory\src\data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # <--- CHANGED FROM ROBUSTSCALER
import joblib
import json
import os

class IndustrialDataLoader:
    def __init__(self, csv_path, window_size=10, test_split=0.2):
        self.csv_path = csv_path
        self.window_size = window_size
        self.test_split = test_split
        self.scaler = MinMaxScaler()  # <--- FORCE 0-1 RANGE
        self.feature_cols = None

    def _add_rolling_features(self, df):
        """
        SHARED LOGIC: This function MUST be used by both Training and Inference.
        """
        # Select only numeric columns
        df_numeric = df.select_dtypes(include=[np.number]).copy()
        base_cols = df_numeric.columns.tolist()
        
        for col in base_cols:
            # 1. Volatility
            df_numeric[f'{col}_std'] = df_numeric[col].rolling(window=5).std()
            # 2. Trend
            df_numeric[f'{col}_mean'] = df_numeric[col].rolling(window=5).mean()
        
        # Drop NaNs created by the rolling window
        df_numeric = df_numeric.dropna()
        return df_numeric

    def load_and_prep(self):
        print(f"   └── 📂 Loading {os.path.basename(self.csv_path)}...")
        df = pd.read_csv(self.csv_path)

        # 1. Clean Metadata
        drop_cols = ['date', 'source_id', 'timestamp', 'row_hash', 'vehicle_id', 'module', 'ingest_ts']
        existing_drop = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing_drop)

        # 2. Force Numeric & Fill Gaps
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()  # Safe Pandas 2.0 syntax
        
        # 3. Apply Rolling Features
        print("   └── 🛠️  Generating Rolling Features...")
        df_eng = self._add_rolling_features(df)
        self.feature_cols = df_eng.columns.tolist()
        
        # 4. Temporal Split
        split_idx = int(len(df_eng) * (1 - self.test_split))
        train_df = df_eng.iloc[:split_idx]
        test_df = df_eng.iloc[split_idx:]
        
        # 5. Fit Scaler (On Train Only)
        print("   └── ⚖️  Fitting MinMaxScaler (0-1)...")
        train_scaled = self.scaler.fit_transform(train_df)
        test_scaled = self.scaler.transform(test_df)
        
        return train_scaled, test_scaled, self.feature_cols

    def create_sliding_windows(self, data):
        # Converts 2D Matrix -> 3D Tensor
        X = []
        for i in range(len(data) - self.window_size):
            X.append(data[i : i + self.window_size])
        return np.array(X)

    def save_scaler(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.scaler, save_path)
        
        # Save Features as JSON
        meta_path = save_path.replace("scaler.pkl", "features.json")
        with open(meta_path, "w") as f:
            json.dump(self.feature_cols, f)