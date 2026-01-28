#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def load_results(path):
    print(f"Loading {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # data structure: { 'known': {'indirect_effects': [[...], [...]]}, ... }
    processed = {}
    for ds_name, ds_content in data.items():
        # Get the indirect effects matrix [Samples, Layers]
        ie_matrix = np.array(ds_content['indirect_effects'])
        # process NaN
        if np.isnan(ie_matrix).any():
            ie_matrix = np.nan_to_num(ie_matrix)
        
        # Mean over samples -> Shape: [Layers]
        mean_ie = np.mean(ie_matrix, axis=0)
        processed[ds_name] = mean_ie
        
    return processed

def plot_verification(mlp_data, attn_data, state_data, output_dir, model_name):
    """State vs. CumSum(MLP + Attn)"""
    sns.set_theme(style="whitegrid")
    datasets = mlp_data.keys()
    
    for ds in datasets:
        if ds not in attn_data or ds not in state_data:
            print(f"Skipping {ds}: missing in some files.")
            continue
            
        mlp_curve = mlp_data[ds]
        attn_curve = attn_data[ds]
        state_curve = state_data[ds] # Ground Truth
        
        # Calculate the cumulative sum
        total_updates = mlp_curve + attn_curve
        reconstructed_state = np.cumsum(total_updates)
        
        # Zero-center alignment
        reconstructed_state = reconstructed_state - reconstructed_state[0] + state_curve[0]

        # compute correlation
        if len(state_curve) > 1:
            corr, _ = pearsonr(state_curve, reconstructed_state)
        else:
            corr = 0.0
        
        plt.figure(figsize=(12, 6))
        
        # 1. real State curve (Target)
        plt.plot(state_curve, label='Actual State Effect (Ground Truth)', 
                 color='black', linewidth=2.5, alpha=0.8)
        
        # 2. reconstructed curve (Hypothesis)
        plt.plot(reconstructed_state, label=f'Cumulative Sum (MLP + Attn)', 
                 color='red', linestyle='--', linewidth=2.5)
        
        # 3. single-layer contribution
        plt.plot(mlp_curve, label='Single Layer MLP (Derivative)', color='blue', alpha=0.2, linewidth=1)
        plt.plot(attn_curve, label='Single Layer Attn (Derivative)', color='green', alpha=0.2, linewidth=1)
        
        plt.title(f"[{model_name}] Mechanism Verification: {ds.upper()} (Correlation: {corr:.4f})", fontsize=14)
        plt.xlabel("Layer Index")
        plt.ylabel("Average Indirect Effect (AIE)")
        plt.legend(loc='best')
        
        # Automatically find the approximate location of Late Integration (the point where State decreases the fastest)
        # This is only for auxiliary labeling to prevent hardcoding layer 60, which may not be suitable for smaller models
        try:
            late_start = np.where(state_curve < state_curve[0] - 2.0)[0][0]
            plt.axvline(x=late_start, color='gray', linestyle=':', alpha=0.5)
            plt.text(late_start + 1, plt.ylim()[0] + (plt.ylim()[1]-plt.ylim()[0])*0.1, "Late Integration Start", fontsize=10, color='gray')
        except:
            pass

        save_path = os.path.join(output_dir, f"verify_mechanism_{ds}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[{ds}] Correlation: {corr:.4f}. Plot saved to {save_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Validate the Residual Stream Dominance Hypothesis (multi-model version)")
    
    parser.add_argument("--model_name", type=str, required=True, 
                        help="model name, for example qwen2.5_7B")
    
    parser.add_argument("--base_dir", type=str, 
                        default="main/causal_trace/results",
                        help="root directory containing all model results")
    
    parser.add_argument("--output_dir", type=str, 
                        default="main/attn+mlp_vs_state/results",
                        help="output directory for verification results")
    
    args = parser.parse_args()
    
    if "32B" in args.model_name or "72B" in args.model_name:
        pkl_filename = "final_results_merged.pkl"
    else:
        pkl_filename = "checkpoint.pkl"
        
    print(f"Model: {args.model_name} | Using filename: {pkl_filename}")

    model_result_dir = os.path.join(args.base_dir, args.model_name)
    mlp_path = os.path.join(model_result_dir, "mlp", pkl_filename)
    attn_path = os.path.join(model_result_dir, "attn", pkl_filename)
    state_path = os.path.join(model_result_dir, "state", pkl_filename)
    
    save_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        for p in [mlp_path, attn_path, state_path]:
            if not os.path.exists(p):
                print(f"Error: Required file not found: {p}")
                return

        mlp_res = load_results(mlp_path)
        attn_res = load_results(attn_path)
        state_res = load_results(state_path)
        
        plot_verification(mlp_res, attn_res, state_res, save_dir, args.model_name)
        
    except Exception as e:
        print(f"Error processing {args.model_name}: {e}")
        return

    print("-" * 50)

if __name__ == "__main__":
    main()