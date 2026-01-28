#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


ROOT_DIR = "main/id_analysis/results"
OUTPUT_ROOT = "main/linear_probe_experiment/results_500epochs_100patience"

MODELS = [
    "qwen2.5_0.5B", "qwen2.5_1.5B", "qwen2.5_3B", 
    "qwen2.5_7B", "qwen2.5_14B", "qwen2.5_32B", "qwen2.5_72B"
]

TRAIN_CONFIG = {
    "batch_size": 256,
    "lr": 1e-3,
    "epochs": 500,
    "patience": 100,       
    "weight_decay": 0.01,  # high-dimension vector normalization
    "val_split": 0.2,
    "seed": 42
}

# ==========================================
# 1. data loader
# ==========================================

class LayerDataManager:
    def __init__(self, model_root):
        self.base_dir = os.path.join(model_root, "activations")
        self.index = {} 
        self._scan_files()
        
    def _scan_files(self):
        sample_path = os.path.join(self.base_dir, "known", "baseline")
        if not os.path.exists(sample_path):
            print(f"Warning: Directory not found {sample_path}")
            self.available_layers = []
            return

        all_files = glob.glob(os.path.join(sample_path, "*.npy"))
        layers = set()
        for f in all_files:
            try:
                parts = os.path.basename(f).split('_')
                if parts[1].isdigit():
                    layers.add(int(parts[1]))
            except:
                continue
        self.available_layers = sorted(list(layers))

        for layer in self.available_layers:
            self.index[layer] = {}
            for ds in ["known", "unknown", "ambiguous"]:
                self.index[layer][ds] = {}
                for rt in ["baseline", "counterfactual"]:
                    path = os.path.join(self.base_dir, ds, rt)
                    files = sorted(glob.glob(os.path.join(path, f"layer_{layer:03d}_worker_*.npy")))
                    if not files:
                        files = sorted(glob.glob(os.path.join(path, f"layer_{layer:03d}_single.npy")))
                    self.index[layer][ds][rt] = files

    def load_data(self, layer, dataset, run_type):
        files = self.index.get(layer, {}).get(dataset, {}).get(run_type, [])
        if not files: return None
        arrays = [np.load(f) for f in files]
        return np.concatenate(arrays, axis=0) if arrays else None

# ==========================================
# 2. Linear Probe Model
# ==========================================

class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # w^T * x + b
        # Geometric interpretation: w is the normal vector of the knowledge direction, and b is used to absorb the residual after bias correction.

        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# ==========================================
# 3. Training engine
# ==========================================

def train_and_evaluate(X_train, y_train, X_val, y_val, device):
    input_dim = X_train.shape[1]
    model = LinearProbe(input_dim).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"])
    
    # Use Cosine Annealing to make the loss converge more deeply, which works excellently with 500 epochs.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])
    
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)
    
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze()
            val_loss = criterion(val_logits, y_val_t).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
        if patience_counter >= TRAIN_CONFIG["patience"]:
            break
            
    if best_state:
        model.load_state_dict(best_state)
        
    return model

def predict_prob(model, X, device):
    model.eval()
    if X is None or len(X) == 0: return np.array([])
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            batch = torch.from_numpy(X[i:i+1024]).float().to(device)
            logits = model(batch).squeeze()
            if logits.ndim == 0: logits = logits.unsqueeze(0)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)

# ==========================================
# 4. main process
# ==========================================

def process_single_model(model_name, device):
    print(f"\n{'='*40}")
    print(f"Processing Model: {model_name}")
    print(f"{'='*40}")
    
    model_dir = os.path.join(ROOT_DIR, model_name)
    save_dir = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    dm = LayerDataManager(model_dir)
    if not hasattr(dm, 'available_layers') or not dm.available_layers:
        print(f"Skipping {model_name}: No layers found.")
        return

    results_df = []
    all_probes_weights = {} # Used to store the probing weights for all layers of the model.

    
    for layer in tqdm(dm.available_layers, desc=f"Probing {model_name}"):
        # 1. Load the four necessary data blocks

        X_known_corr = dm.load_data(layer, "known", "baseline")
        X_unknown_corr = dm.load_data(layer, "unknown", "baseline")
        X_known_ctx = dm.load_data(layer, "known", "counterfactual")     
        X_unknown_ctx = dm.load_data(layer, "unknown", "counterfactual") 
        
        if any(x is None for x in [X_known_corr, X_unknown_corr, X_known_ctx, X_unknown_ctx]):
            continue

        # 2. Calculate geometric centroids - the basis of translation invariance assumption
        base_corr = np.mean(X_unknown_corr, axis=0)
        base_ctx = np.mean(X_unknown_ctx, axis=0)
        
        # 3. Construct de-biased training data
        min_len = min(len(X_known_corr), len(X_unknown_corr))
        np.random.seed(TRAIN_CONFIG["seed"])
        idx_k = np.random.choice(len(X_known_corr), min_len, replace=False)
        idx_u = np.random.choice(len(X_unknown_corr), min_len, replace=False)
        
        # Core geometric operation: Translate the vector to the "ignorance" origin
        X_train_pos = X_known_corr[idx_k] - base_corr
        X_train_neg = X_unknown_corr[idx_u] - base_corr 
        
        X_train_full = np.concatenate([X_train_pos, X_train_neg], axis=0)
        y_train_full = np.concatenate([np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))], axis=0)
        
        # Compute the scaler using only the training set to prevent data leakage
        X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
            X_train_full, y_train_full, test_size=TRAIN_CONFIG["val_split"], 
            random_state=TRAIN_CONFIG["seed"], stratify=y_train_full
        )
        
        # Calculate statistics
        scaler_mean = X_tr_raw.mean(axis=0)
        scaler_std = X_tr_raw.std(axis=0) + 1e-6
        
        # Apply normalization
        X_tr = (X_tr_raw - scaler_mean) / scaler_std
        X_val = (X_val_raw - scaler_mean) / scaler_std
        
        # 4. Train the probe
        probe = train_and_evaluate(X_tr, y_tr, X_val, y_val, device)
        
        # Save the trained probing weights
        # We save the state_dict (weights and biases) as well as the corresponding scaler parameters for this layer,
        # so that when loading later, the new data can be correctly preprocessed.
        all_probes_weights[f"layer_{layer}"] = {
            "model_state": probe.state_dict(),
            "scaler_mean": scaler_mean, 
            "scaler_std": scaler_std,   
            "base_corr": base_corr,     
            "base_ctx": base_ctx        
        }
        
        # 5. Cross-domain testing
        # Geometric hypothesis testing: In the Context coordinate system, test whether the vector still points in the "knowledge" direction.
        X_test_conflict = X_known_ctx - base_ctx
        
        # The projection must be done using the scaler from the training set.
        X_test_conflict_norm = (X_test_conflict - scaler_mean) / scaler_std
        
        # Evaluate
        val_probs = predict_prob(probe, X_val, device)
        val_preds = (val_probs >= 0.5)
        acc_baseline = accuracy_score(y_val, val_preds) 
        
        conflict_probs = predict_prob(probe, X_test_conflict_norm, device)
        suppression_rate = np.mean(conflict_probs < 0.5)
        
        results_df.append({
            "layer": layer,
            "acc_baseline": acc_baseline,
            "suppression_rate": suppression_rate
        })
        
        del probe, X_known_corr, X_unknown_corr, X_known_ctx, X_unknown_ctx
        del X_train_full, X_tr, X_val, X_test_conflict_norm
        torch.cuda.empty_cache()

    if not results_df:
        print(f"No results generated for {model_name}")
        return

    # --- save results ---
    
    df = pd.DataFrame(results_df)
    csv_path = os.path.join(save_dir, "metrics_debiased.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    weights_path = os.path.join(save_dir, "probe_weights.pt")
    torch.save(all_probes_weights, weights_path)
    print(f"Probe weights saved to {weights_path}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if 'acc_baseline' in df.columns:
        plt.plot(df['layer'], df['acc_baseline'], label='Probe Reliability')
        plt.title(f"{model_name}: Baseline Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if 'suppression_rate' in df.columns:
        plt.plot(df['layer'], df['suppression_rate'], color='red', label='Suppression Rate')
        plt.title(f"{model_name}: Knowledge Suppression")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_plot_debiased.png"))
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
        
    for model_name in MODELS:
        process_single_model(model_name, device)

if __name__ == "__main__":
    main()