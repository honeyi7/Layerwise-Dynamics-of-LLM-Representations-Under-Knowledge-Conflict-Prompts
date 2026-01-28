#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm import tqdm

BASE_ROOT_DIR = "main/id_analysis/results"
OUTPUT_DIR = "main/cosine_similarity/aggregate_results"

MODEL_SIZES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]
DATASETS = ["known", "unknown", "ambiguous"]


COLOR_PALETTE = sns.color_palette("flare", n_colors=len(MODEL_SIZES))
# ===========================================

def load_layer_data(base_dir, layer_idx):
    pattern_worker = os.path.join(base_dir, f"layer_{layer_idx:03d}_worker_*.npy")
    files = sorted(glob.glob(pattern_worker))
    
    if not files:
        pattern_single = os.path.join(base_dir, f"layer_{layer_idx:03d}_single.npy")
        files = sorted(glob.glob(pattern_single))
    
    if not files:
        return None
    
    arrays = []
    for f in files:
        try:
            arr = np.load(f)
            arrays.append(arr)
        except Exception as e:
            return None
            
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)

def compute_cosine_similarity(A, B):
    if A.shape != B.shape:
        min_len = min(A.shape[0], B.shape[0])
        A = A[:min_len]
        B = B[:min_len]
    
    norm_a = np.linalg.norm(A, axis=1)
    norm_b = np.linalg.norm(B, axis=1)
    dot_product = np.sum(A * B, axis=1)
    denominator = norm_a * norm_b
    denominator[denominator < 1e-10] = 1e-10
    return dot_product / denominator

def collect_model_data(model_size):
    """
    Collect all layer data of a single model
    Returns: { 'known': {'x': [], 'y': [], 'std': []}, ... }
    """
    print(f"Collecting data for Qwen2.5-{model_size}...")
    activations_dir = os.path.join(BASE_ROOT_DIR, f"qwen2.5_{model_size}", "activations")
    
    results = {ds: {'x': [], 'y': [], 'std': []} for ds in DATASETS}
    
    for ds in DATASETS:
        dir_baseline = os.path.join(activations_dir, ds, "baseline")
        dir_counterfactual = os.path.join(activations_dir, ds, "counterfactual")
        
        if not os.path.exists(dir_baseline):
            print(f"  [Warning] Missing directory for {model_size} - {ds}")
            continue
            
        # Scan the file to determine the number of layers
        sample_files = glob.glob(os.path.join(dir_baseline, "layer_*_worker_0.npy"))
        if not sample_files:
            sample_files = glob.glob(os.path.join(dir_baseline, "layer_*_single.npy"))
        
        if not sample_files:
            continue
            
        layer_indices = sorted([int(os.path.basename(f).split('_')[1]) for f in sample_files])
        max_layer = max(layer_indices) if layer_indices else 1
        
        means = []
        stds = []
        rel_depths = []
        
        for layer_idx in tqdm(layer_indices, desc=f"  {ds} layers", leave=False):
            act_A = load_layer_data(dir_baseline, layer_idx)
            act_B = load_layer_data(dir_counterfactual, layer_idx)
            
            if act_A is None or act_B is None:
                continue
            
            sims = compute_cosine_similarity(act_A, act_B)
            means.append(np.mean(sims))
            stds.append(np.std(sims))
            # Normalize depth: 0.0 ~ 1.0
            rel_depths.append(layer_idx / max_layer)
            
        results[ds]['x'] = rel_depths
        results[ds]['y'] = means
        results[ds]['std'] = stds
        
    return results

def plot_aggregated_results(all_data):

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": "--"})
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    
    line_styles = ['-', '-', '-', '-', '-', '-', '-'] 
    
    for ax_idx, ds in enumerate(DATASETS):
        ax = axes[ax_idx]
        
        for m_idx, model_size in enumerate(MODEL_SIZES):
            if model_size not in all_data:
                continue
                
            data = all_data[model_size][ds]
            if not data['x']:
                continue
            
            ax.plot(data['x'], data['y'], 
                    label=f"{model_size}", 
                    color=COLOR_PALETTE[m_idx], 
                    linewidth=2.5, 
                    alpha=0.9)

        ax.set_title(f"Dataset: {ds.capitalize()}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Relative Network Depth (0=Input, 1=Output)", fontsize=14)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        
        if ax_idx == 0:
            ax.set_ylabel("Cosine Similarity (Param vs Counterfactual)", fontsize=14)
        
        if ax_idx == 2:
            ax.legend(title="Model Size", fontsize=10, title_fontsize=12, loc='lower left')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "aggregated_cosine_similarity.pdf")
    png_path = os.path.join(OUTPUT_DIR, "aggregated_cosine_similarity.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated plots to:\n  {save_path}\n  {png_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_root", type=str, default=BASE_ROOT_DIR, help="Root dir of geometry results")
    args = parser.parse_args()
    
    all_models_data = {}
    
    print("Starting Aggregation...")
    for model in MODEL_SIZES:
        all_models_data[model] = collect_model_data(model)
        
    plot_aggregated_results(all_models_data)
    print("Done.")

if __name__ == "__main__":
    main()