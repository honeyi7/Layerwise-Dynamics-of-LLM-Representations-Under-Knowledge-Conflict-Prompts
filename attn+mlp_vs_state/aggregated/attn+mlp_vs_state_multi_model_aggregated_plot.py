#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.ticker as ticker

# ==========================================
# Visual Setup
# ==========================================

def setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'grid.color': '#dddddd',
        'grid.linestyle': ':',
        'grid.linewidth': 1.0,
        'scatter.edgecolors': 'w',
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

def extract_model_size(model_name):
    """Extract the number of parameters from the model name (e.g., 'qwen2.5_72B' -> 72.0)"""
    match = re.search(r'(\d+\.?\d*)B', model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def load_pkl_data(path):
    if not os.path.exists(path): return None
    try:
        with open(path, 'rb') as f: data = pickle.load(f)
        processed = {}
        for ds_name, ds_content in data.items():
            ie_matrix = np.array(ds_content['indirect_effects'])
            if np.isnan(ie_matrix).any(): ie_matrix = np.nan_to_num(ie_matrix)
            processed[ds_name] = np.mean(ie_matrix, axis=0)
        return processed
    except Exception as e:
        print(f"Error loading {path}: {e}"); return None


def plot_correlation_scaling_scatter_optimized(stats_data, output_dir):
    rows = []
    for model, datasets in stats_data.items():
        size = extract_model_size(model)
        for ds, metrics in datasets.items():
            rows.append({
                "Model": model,
                "Size (B)": size,
                "Dataset": ds.upper(),
                "Correlation": metrics['correlation']
            })
    df = pd.DataFrame(rows)
    
    df = df.sort_values(by="Size (B)")
    
    plt.figure(figsize=(10, 7))
    
    color_map = {
        "KNOWN": "#006400",     
        "AMBIGUOUS": "#1F77B4", 
        "UNKNOWN": "#D35400"    
    }
    markers_dict = {
        "KNOWN": "o",   
        "AMBIGUOUS": "s", 
        "UNKNOWN": "X"  
    }
    # =======================================
    
    # 1. Line Plot
    sns.lineplot(
        data=df,
        x="Size (B)",
        y="Correlation",
        hue="Dataset",
        style="Dataset", 
        markers=False,
        dashes=True,
        palette=color_map, 
        linewidth=2,
        legend=False
    )

    # 2. Scatter Plot
    ax = sns.scatterplot(
        data=df, 
        x="Size (B)", 
        y="Correlation", 
        hue="Dataset", 
        style="Dataset",
        markers=markers_dict, 
        s=200,                
        alpha=1.0, 
        palette=color_map,   
        edgecolor="white", 
        linewidth=1.5, 
        zorder=10
    )
    
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([0.5, 1.5, 3, 7, 14, 32, 72])
    
    
    plt.xlabel("Model Scale (B)", fontweight='bold', fontsize=14)
    plt.ylabel("Pearson Correlation ($r$)", fontweight='bold', fontsize=14)
    
    plt.axhline(1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)

    plt.ylim(0.90, 1.01) 
    
    plt.grid(True, which="major", ls="--", c='#dddddd', alpha=0.8)
    plt.grid(True, which="minor", ls=":", c='#eeeeee', alpha=0.5)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Knowledge Type", 
               frameon=True, fontsize=12, title_fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "scatter_scaling_law_optimized.png")
    plt.savefig(save_path)
    print(f"Saved optimized scaling plot to {save_path}")
    plt.close()

def plot_dynamics_swarm_optimized(curves_data, output_dir):
    rows = []
    model_sizes = set()
    for model, ds_content in curves_data.items():
        size = extract_model_size(model)
        model_sizes.add(size)
        for ds, inner in ds_content.items():
            raw_curve = inner['state_curve']
            min_val = np.min(raw_curve)
            norm_y = raw_curve / abs(min_val) if abs(min_val) > 1e-9 else raw_curve
            num_layers = len(raw_curve)
            rel_depth = np.linspace(0, 1, num_layers)
            
            for i in range(num_layers):
                rows.append({
                    "Size": size,
                    "Dataset": ds.upper(),
                    "Relative Depth": rel_depth[i],
                    "Effect": norm_y[i]
                })
    df = pd.DataFrame(rows)
    
    cmap_name = "plasma_r"
    target_order = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
    
    g = sns.FacetGrid(df, col="Dataset", hue="Size", palette=cmap_name, 
                      col_order=target_order,
                      height=5, aspect=0.9, xlim=(-0.05, 1.05), ylim=(-1.1, 0.1))
    
    g.map(plt.scatter, "Relative Depth", "Effect", s=35, alpha=0.7, edgecolor='white', linewidth=0.2)
    
    g.set_titles("{col_name}", fontweight='bold', fontsize=14)
    g.set_axis_labels("", "Normalized Indirect Effect")
    g.set(xticks=[0, 0.25, 0.5, 0.75, 1.0])
    
    for ax in g.axes.flat:
        ax.axhline(0, color='#555555', linestyle='-', linewidth=1, alpha=0.5, zorder=0)
        ax.axvspan(0.75, 1.0, color='#eeeeee', alpha=0.4, zorder=0)

    g.fig.suptitle("Comparative Dynamics of Residual Stream Integration", 
                   fontsize=18, fontweight='bold', y=1.05)
    
    g.fig.text(0.42, 0.02, "Relative Depth", ha='center', fontsize=16, fontweight='bold')

    norm = plt.Normalize(df['Size'].min(), df['Size'].max())
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    
    plt.subplots_adjust(top=0.85, right=0.83, bottom=0.15, wspace=0.1)
    
    cbar_ax = g.fig.add_axes([0.86, 0.2, 0.025, 0.6])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Model Scale (B)', fontweight='bold', fontsize=12, labelpad=10)
    
    sorted_sizes = sorted(list(model_sizes))
    ticks_to_show = [s for s in sorted_sizes if s in [0.5, 7.0, 32.0, 72.0]]
    if len(ticks_to_show) < 2:
         ticks_to_show = [min(sorted_sizes), max(sorted_sizes)]

    cbar.set_ticks(ticks_to_show)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(1)
    
    save_path = os.path.join(output_dir, "scatter_dynamics_swarm_optimized.png")
    g.fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved optimized swarm plot to {save_path}")
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="main/causal_trace/results")
    parser.add_argument("--output_dir", type=str, default="main/attn+mlp_vs_state/aggregated/results")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=["qwen2.5_0.5B", "qwen2.5_1.5B", "qwen2.5_3B", 
                                 "qwen2.5_7B", "qwen2.5_14B", "qwen2.5_32B", "qwen2.5_72B"])
    args = parser.parse_args()
    
    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)
    
    stats_data = {}
    curves_data = {}
    
    print(f"Processing {len(args.models)} models...")
    for model_name in args.models:
        model_path = os.path.join(args.base_dir, model_name)
        def get_path(comp):
            candidates = [
                os.path.join(model_path, comp, "final_results_merged.pkl"),
                os.path.join(model_path, comp, "checkpoint.pkl"),
                os.path.join(model_path, comp, "checkpoint_worker_0.pkl")
            ]
            for p in candidates:
                if os.path.exists(p): return p
            return None
        mlp_p, attn_p, state_p = get_path("mlp"), get_path("attn"), get_path("state")
        if not (mlp_p and attn_p and state_p): continue
        mlp, attn, state = load_pkl_data(mlp_p), load_pkl_data(attn_p), load_pkl_data(state_p)
        if not (mlp and attn and state): continue
        stats_data[model_name] = {}
        curves_data[model_name] = {}
        for ds in mlp.keys():
            if ds not in attn or ds not in state: continue
            total = mlp[ds] + attn[ds]
            recon = np.cumsum(total)
            recon = recon - recon[0] + state[ds][0]
            corr = 0
            if len(state[ds]) > 1: corr, _ = pearsonr(state[ds], recon)
            stats_data[model_name][ds] = {"correlation": corr}
            curves_data[model_name][ds] = {"state_curve": state[ds]}
            
    print("Generating Optimized Scatter Plots...")
    if stats_data:
        plot_correlation_scaling_scatter_optimized(stats_data, args.output_dir)
    
    if curves_data:
        plot_dynamics_swarm_optimized(curves_data, args.output_dir)
    else:
        print("No valid data found to plot.")
        
    print("Done!")

if __name__ == "__main__":
    main()