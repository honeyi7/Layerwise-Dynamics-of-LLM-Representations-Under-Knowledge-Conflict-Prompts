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
DATASETS = ["known", "ambiguous", "unknown"]


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
    activations_dir = os.path.join(BASE_ROOT_DIR, f"qwen2.5_{model_size}", "activations")
    curves = {}
    
    dir_base = os.path.join(activations_dir, "known", "baseline")
    if not os.path.exists(dir_base): return None
    
    files = glob.glob(os.path.join(dir_base, "layer_*_worker_0.npy"))
    if not files: files = glob.glob(os.path.join(dir_base, "layer_*_single.npy"))
    if not files: return None
    
    layer_indices = sorted([int(os.path.basename(f).split('_')[1]) for f in files])
    
    for ds in DATASETS:
        means = []
        dir_b = os.path.join(activations_dir, ds, "baseline")
        dir_c = os.path.join(activations_dir, ds, "counterfactual")
        
        for layer_idx in tqdm(layer_indices, desc=f"{model_size} {ds}", leave=False):
            act_A = load_layer_data(dir_b, layer_idx)
            act_B = load_layer_data(dir_c, layer_idx)
            if act_A is not None and act_B is not None:
                means.append(np.mean(compute_cosine_similarity(act_A, act_B)))
            else:
                means.append(np.nan)
        
        s = pd.Series(means)
        curves[ds] = s.interpolate().to_numpy()
        
    return curves, layer_indices

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    correlation_records = []
    
    print("Calculating Metrics...")
    for model in MODEL_SIZES:
        res = get_model_curves(model)
        if not res: continue
        curves, _ = res
        
        # Calculate Pearson correlation coefficient (measuring the curve shape)
        r_ku = np.corrcoef(curves['known'], curves['unknown'])[0, 1]
        r_ka = np.corrcoef(curves['known'], curves['ambiguous'])[0, 1]
        
        # Calculate Euclidean Distance (measuring the gap between curves)
        d_ku = np.linalg.norm(curves['known'] - curves['unknown'])
        d_ka = np.linalg.norm(curves['known'] - curves['ambiguous'])
        
        correlation_records.append({
            "Model": model,
            "r_Known_Unknown": r_ku,
            "r_Known_Ambiguous": r_ka,
            "d_Known_Unknown": d_ku,
            "d_Known_Ambiguous": d_ka
        })

    df = pd.DataFrame(correlation_records)
    print("\n=== Metric Table ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_DIR, "dataset_metrics.csv"), index=False)
    print("Generating Per-Model Comparison Plots...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 10)) 
    axes = axes.flatten()
    
    colors = {"known": "#1f77b4", "unknown": "#d62728", "ambiguous": "#2ca02c"}
    
    global_handles = []
    global_labels = []
    
    for i, model in enumerate(MODEL_SIZES):
        ax = axes[i]
        res = get_model_curves(model)
        if not res: continue
        curves, layers = res

        x_axis = np.array(layers) / max(layers)
        
        for ds in DATASETS:
            line, = ax.plot(x_axis, curves[ds], label=ds.capitalize(), color=colors[ds], linewidth=2.5)
            if i == 0:
                global_handles.append(line)
                global_labels.append(ds.capitalize())
            
        # Get the metrics of the model
        row = df[df['Model'] == model].iloc[0]
        
        # Construct the text content
        text_str = (f"{model}\n"
                    f"d(K,U)={row['d_Known_Unknown']:.3f}\n"
                    f"d(K,A)={row['d_Known_Ambiguous']:.3f}")
        
        ax.text(0.04, 0.04, text_str, 
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                verticalalignment='bottom', 
                horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="lightgray"))
        
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        
        if i >= 4: 
            ax.set_xlabel("Relative Depth", fontweight='bold', fontsize=14)
        
        if i % 4 == 0:
            ax.set_ylabel("Cosine Similarity", fontweight='bold', fontsize=14)
        else:
            ax.set_yticklabels([])

    legend_ax = axes[7]
    legend_ax.axis('off') 
    
    if global_handles:
        legend = legend_ax.legend(global_handles, global_labels, 
                         loc='center', 
                         title="Dataset",
                         fontsize=16, 
                         title_fontsize=16, 
                         frameon=True, 
                         fancybox=True, 
                         shadow=True,
                         borderpad=1.5)

        legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    per_model_path = os.path.join(OUTPUT_DIR, "per_model_trajectories.png")
    plt.savefig(per_model_path, dpi=300, bbox_inches='tight')
    print(f"Per-Model plots saved to: {per_model_path}")

if __name__ == "__main__":
    main()