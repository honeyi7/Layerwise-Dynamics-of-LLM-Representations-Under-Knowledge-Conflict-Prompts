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

def load_layer_data(base_dir, layer_idx):
    """
    Load the file for the specified layer (adaptively supports Distributed Worker mode and Single GPU mode)
    """
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
            print(f"Error loading {f}: {e}")
            return None
            
    if not arrays:
        return None
        
    # Concatenate by rows (Samples dimension)
    return np.concatenate(arrays, axis=0)

def compute_cosine_similarity(A, B):
    """
    A, B shape: [N_samples, Hidden_Dim]
    Return: [N_samples]
    """
    if A.shape != B.shape:
        min_len = min(A.shape[0], B.shape[0])
        A = A[:min_len]
        B = B[:min_len]
    
    # Calculate the norm
    norm_a = np.linalg.norm(A, axis=1)
    norm_b = np.linalg.norm(B, axis=1)
    
    # Calculate the dot product
    dot_product = np.sum(A * B, axis=1)
    
    # Prevent division by zero
    denominator = norm_a * norm_b
    denominator[denominator < 1e-10] = 1e-10
    
    sims = dot_product / denominator
    return sims

def main():
    parser = argparse.ArgumentParser(description="Calculate Layer-wise Cosine Similarity")
    parser.add_argument("--activations_dir", type=str, required=True, 
                        help="Path to the root of activations")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets = ["known", "unknown", "ambiguous"]
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    colors = {"known": "blue", "unknown": "red", "ambiguous": "green"}
    
    global_max_layer = 0
    has_plotted_any = False

    for ds in datasets:
        print(f"Processing dataset: {ds}...")
        
        dir_baseline = os.path.join(args.activations_dir, ds, "baseline")
        dir_counterfactual = os.path.join(args.activations_dir, ds, "counterfactual")
        
        if not os.path.exists(dir_baseline) or not os.path.exists(dir_counterfactual):
            print(f"Skipping {ds}: directories not found.")
            continue
            
        # Scan the number of layers: first try Worker mode, then try Single mode
        sample_files = glob.glob(os.path.join(dir_baseline, "layer_*_worker_0.npy"))
        if not sample_files:
            sample_files = glob.glob(os.path.join(dir_baseline, "layer_*_single.npy"))
            
        if not sample_files:
            print(f"No files found for {ds}")
            continue
            
        # Parse layer index
        layer_indices = sorted([int(os.path.basename(f).split('_')[1]) for f in sample_files])
        
        if not layer_indices:
            continue

        # Update the global maximum number of layers
        current_max = max(layer_indices)
        if current_max > global_max_layer:
            global_max_layer = current_max
        
        sim_means = []
        sim_stds = []
        valid_layers = []
        
        for layer_idx in tqdm(layer_indices, desc=f"Calc Sim {ds}"):
            # load baseline (Run A)
            act_A = load_layer_data(dir_baseline, layer_idx)
            # load counterfactual (Run B)
            act_B = load_layer_data(dir_counterfactual, layer_idx)
            
            if act_A is None or act_B is None:
                continue
                
            sims = compute_cosine_similarity(act_A, act_B)
            
            sim_means.append(np.mean(sims))
            sim_stds.append(np.std(sims))
            valid_layers.append(layer_idx)
        
        if not valid_layers:
            continue
            
        has_plotted_any = True
        
        plt.plot(valid_layers, sim_means, label=f"{ds.upper()}", color=colors[ds], linewidth=2.5, marker='o', markersize=4)
        plt.fill_between(valid_layers, 
                         np.array(sim_means) - np.array(sim_stds), 
                         np.array(sim_means) + np.array(sim_stds), 
                         color=colors[ds], alpha=0.15)

    if has_plotted_any:
        plt.title(f"Layer-wise Cosine Similarity: Parametric Memory vs. counterfactual State", fontsize=16)
        plt.xlabel("Layer Index", fontsize=14)
        plt.ylabel("Cosine Similarity", fontsize=14)
        
        plt.xlim(0, global_max_layer)
        plt.ylim(0, 1.05) 
        
        
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        save_path = os.path.join(args.output_dir, "cosine_similarity_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Analysis Done. Plot saved to {save_path}")
    else:
        print("No valid data found to plot.")

if __name__ == "__main__":
    main()