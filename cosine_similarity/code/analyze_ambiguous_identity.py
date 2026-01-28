#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
from tqdm import tqdm

BASE_ROOT_DIR = "main/id_analysis/results"
OUTPUT_DIR = "main/cosine_similarity/aggregate_results"
MODEL_SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]
# ===========================================

def load_layer_data(base_dir, layer_idx):
    pattern_worker = os.path.join(base_dir, f"layer_{layer_idx:03d}_worker_*.npy")
    files = sorted(glob.glob(pattern_worker))
    if not files:
        pattern_single = os.path.join(base_dir, f"layer_{layer_idx:03d}_single.npy")
        files = sorted(glob.glob(pattern_single))
    if not files: return None
    arrays = []
    for f in files:
        try:
            arrays.append(np.load(f))
        except: pass
    if not arrays: return None
    return np.concatenate(arrays, axis=0)

def compute_cosine_similarity(A, B):
    if A.shape != B.shape:
        min_len = min(A.shape[0], B.shape[0])
        A, B = A[:min_len], B[:min_len]
    norm_a = np.linalg.norm(A, axis=1)
    norm_b = np.linalg.norm(B, axis=1)
    denom = norm_a * norm_b
    denom[denom < 1e-10] = 1e-10
    return np.sum(A * B, axis=1) / denom

def get_model_curves(model_size):
    print(f"Processing {model_size}...")
    activations_dir = os.path.join(BASE_ROOT_DIR, f"qwen2.5_{model_size}", "activations")
    
    # Only the layers of Known are needed as index references
    dir_base = os.path.join(activations_dir, "known", "baseline")
    if not os.path.exists(dir_base): return None
    
    files = glob.glob(os.path.join(dir_base, "layer_*_worker_0.npy"))
    if not files: files = glob.glob(os.path.join(dir_base, "layer_*_single.npy"))
    if not files: return None
    
    layer_indices = sorted([int(os.path.basename(f).split('_')[1]) for f in files])
    curves = {}
    
    for ds in ["known", "unknown", "ambiguous"]:
        means = []
        dir_b = os.path.join(activations_dir, ds, "baseline")
        dir_c = os.path.join(activations_dir, ds, "counterfactual")
        
        for layer_idx in layer_indices:
            act_A = load_layer_data(dir_b, layer_idx)
            act_B = load_layer_data(dir_c, layer_idx)
            if act_A is not None and act_B is not None:
                val = np.mean(compute_cosine_similarity(act_A, act_B))
                means.append(val)
            else:
                means.append(np.nan)
        
        # Interpolation filling
        s = pd.Series(means)
        curves[ds] = s.interpolate().to_numpy()
        
    return curves, layer_indices

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    analysis_data = []

    for model in MODEL_SIZES:
        res = get_model_curves(model)
        if not res: continue
        curves, _ = res
        
        # Calculate the similarity between Ambiguous and Known/Unknown (Pearson Correlation)
        # This represents the similarity in terms of "curve shape"
        corr_with_known = np.corrcoef(curves['ambiguous'], curves['known'])[0, 1]
        corr_with_unknown = np.corrcoef(curves['ambiguous'], curves['unknown'])[0, 1]
        
        # Calculate the Euclidean distance between Ambiguous and Known/Unknown
        # This represents the proximity in terms of "absolute value"
        dist_to_known = np.linalg.norm(curves['ambiguous'] - curves['known'])
        dist_to_unknown = np.linalg.norm(curves['ambiguous'] - curves['unknown'])
        
        analysis_data.append({
            "Model Size": model,
            "Corr(Amb, Known)": corr_with_known,
            "Corr(Amb, Unknown)": corr_with_unknown,
            "Dist(Amb, Known)": dist_to_known,
            "Dist(Amb, Unknown)": dist_to_unknown
        })
    
    df = pd.DataFrame(analysis_data)
    print("\nAnalysis Data:")
    print(df)

    # 2. Plotting: Correlation Tug-of-War
    # This plot answers the question: Which entity does the trend of Ambiguous resemble more?
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df["Model Size"]))
    width = 0.35
    
    plt.bar(x - width/2, df["Corr(Amb, Known)"], width, label='Similarity to Known', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, df["Corr(Amb, Unknown)"], width, label='Similarity to Unknown', color='#e74c3c', alpha=0.8)
    
    plt.xticks(x, df["Model Size"], fontsize=12)
    plt.ylabel("Pearson Correlation (Curve Shape)", fontsize=12)
    plt.title("Identity Crisis: Does 'Ambiguous' behave like Known or Unknown?", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.ylim(0, 1.1)
    
    save_path_corr = os.path.join(OUTPUT_DIR, "ambiguous_correlation_bar.png")
    plt.savefig(save_path_corr, dpi=300, bbox_inches='tight')
    print(f"Correlation plot saved to {save_path_corr}")

    # 3. Plotting: Distance Trend
    # This plot answers the question: As the model grows, which entity does Ambiguous get closer to?
    plt.figure(figsize=(10, 6))
    
    plt.plot(df["Model Size"], df["Dist(Amb, Known)"], marker='o', linewidth=3, label="Distance to Known", color='#3498db')
    plt.plot(df["Model Size"], df["Dist(Amb, Unknown)"], marker='o', linewidth=3, label="Distance to Unknown", color='#e74c3c')
    
    plt.ylabel("Euclidean Distance (Lower is Closer)", fontsize=12)
    plt.title("Trajectory Distance: Ambiguous vs (Known/Unknown)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path_dist = os.path.join(OUTPUT_DIR, "ambiguous_distance_trend.png")
    plt.savefig(save_path_dist, dpi=300, bbox_inches='tight')
    print(f"Distance plot saved to {save_path_dist}")

if __name__ == "__main__":
    main()