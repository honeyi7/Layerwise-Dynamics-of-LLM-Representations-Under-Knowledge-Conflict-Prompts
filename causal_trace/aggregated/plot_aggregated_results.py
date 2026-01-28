#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


def setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.5, 
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })


def extract_model_size_num(model_name):
    match = re.search(r'(\d+\.?\d*)B', model_name, re.IGNORECASE)
    if match: return float(match.group(1))
    return 0

def get_path_strict(base_dir, model_name, component):
    model_path = os.path.join(base_dir, model_name)
    candidates = [
        os.path.join(model_path, component, "final_results_merged.pkl"),
        os.path.join(model_path, component, "checkpoint.pkl"),
        os.path.join(model_path, component, "checkpoint_worker_0.pkl")
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return None

def normalize_layer_indices(indirect_effects):
    num_layers = len(indirect_effects)
    x_axis = np.linspace(0, 100, num_layers) if num_layers > 0 else np.array([])
    return x_axis, indirect_effects

def calculate_sparsity_metrics(normalized_ie, threshold=0.90):
    positive_ie = normalized_ie[normalized_ie > 0]
    total_layers = len(normalized_ie)
    if len(positive_ie) == 0: return 0, 0.0
    total_effect = np.sum(positive_ie)
    if total_effect < 1e-9: return 0, 0.0
    
    sorted_ie = np.sort(positive_ie)[::-1]
    cumsum_ie = np.cumsum(sorted_ie)
    target_value = total_effect * threshold
    k = np.searchsorted(cumsum_ie, target_value, side='left') + 1
    k = min(k, total_layers)
    ratio = k / total_layers if total_layers > 0 else 0
    return k, ratio

def load_data(base_dir, model_names, target_component):
    trace_data = []
    sparsity_data = []
    print(f"   -> Scanning for '{target_component}'...")
    
    for model_name in model_names:
        pkl_path = get_path_strict(base_dir, model_name, target_component)
        if not pkl_path: continue
        try:
            with open(pkl_path, 'rb') as f: data = pickle.load(f)
            size = extract_model_size_num(model_name)
            for ds_name, ds_content in data.items():
                ie_matrix = np.array(ds_content['indirect_effects'])
                if np.isnan(ie_matrix).any(): ie_matrix = np.nan_to_num(ie_matrix)
                if len(ie_matrix) == 0: continue

                te_array = np.array(ds_content['total_effects'])
                mean_ate = np.mean(te_array)
                mean_ie = np.mean(ie_matrix, axis=0)
                if abs(mean_ate) > 1e-6: normalized_ie = mean_ie / mean_ate
                else: normalized_ie = mean_ie
                
                smoothed_ie = gaussian_filter1d(normalized_ie, sigma=0.8)
                x_rel, _ = normalize_layer_indices(smoothed_ie)
                trace_data.append({
                    "Model": model_name, "Size": size, "Dataset": ds_name.upper(),
                    "Layer_Bin": x_rel, "Effect": smoothed_ie
                })
                
                k_count, k_ratio = calculate_sparsity_metrics(normalized_ie, threshold=0.90)
                sparsity_data.append({
                    "Model": model_name, "Size": size, "Dataset": ds_name.upper(),
                    "Layer Count (K)": k_count, "Layer Ratio (%)": k_ratio * 100.0
                })
        except: pass
    return trace_data, pd.DataFrame(sparsity_data)


def plot_heatmap(trace_data_list, output_dir, component):
    if not trace_data_list: return
    
    available_datasets = set(d['Dataset'] for d in trace_data_list)
    target_order = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
    datasets = [d for d in target_order if d in available_datasets]
    remaining = sorted(list(available_datasets - set(datasets)))
    datasets.extend(remaining)

    common_x = np.linspace(0, 100, 100)
    heatmap_rows = []
    for record in trace_data_list:
        interpolated_y = np.interp(common_x, record['Layer_Bin'], record['Effect'])
        display_label = record['Model'].replace("qwen2.5_", "")
        for i, val in enumerate(interpolated_y):
            heatmap_rows.append({
                "Dataset": record['Dataset'], "Model Label": display_label,
                "Layer Depth (%)": common_x[i], "Effect": val
            })
            
    df = pd.DataFrame(heatmap_rows)
    sorted_models = sorted(df['Model Label'].unique(), key=lambda x: extract_model_size_num(x))
    cmap_dict = {'attn': 'Purples', 'mlp': 'Greens', 'state': 'Reds'}
    current_cmap = cmap_dict.get(component, 'Blues')
    title_dict = {'attn': 'Attention', 'mlp': 'MLP', 'state': 'State'}
    comp_title = title_dict.get(component, component.upper())

    num_plots = len(datasets)
    if num_plots == 0: return
    
    fig, axes = plt.subplots(1, max(3, num_plots), figsize=(20, 6.5), sharey=True)
    if num_plots == 1: axes = [axes]

    middle_idx = (num_plots - 1) // 2
    
    with sns.axes_style("white"):
        for i, ds in enumerate(datasets):
            if i >= len(axes): break
            ax = axes[i]
            subset = df[df['Dataset'] == ds]
            pivot_data = subset.pivot_table(index="Model Label", columns="Layer Depth (%)", values="Effect") 
            pivot_data = pivot_data.reindex(sorted_models)
            
            is_last = (i == len(datasets) - 1)
            sns.heatmap(pivot_data, ax=ax, cmap=current_cmap, cbar=is_last, 
                        vmin=0.0, vmax=0.5,
                        cbar_kws={'label': 'Normalized Impact'} if is_last else {})
            ax.text(0.03, 0.96, ds, transform=ax.transAxes, 
                    fontsize=20, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle="round", fc="white", ec="#dddddd", alpha=0.85))
            
            if i == middle_idx:
                ax.set_xlabel("Relative Depth", fontsize=20, fontweight='bold')
            else:
                ax.set_xlabel("")
                
            if i == 0: 
                ax.set_ylabel("Model Scale", fontsize=15, fontweight='bold')
                plt.setp(ax.get_yticklabels(), rotation=0, fontweight='bold', fontsize=15)
            else: 
                ax.set_ylabel("")
            
            ax.set_xticks([0, 25, 50, 75, 99])
            ax.set_xticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])
            plt.setp(ax.get_xticklabels(), rotation=0, fontweight='bold', fontsize=15)
            
            for _, spine in ax.spines.items():
                spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1)
                
        for j in range(i + 1, len(axes)):
             axes[j].axis('off')

    plt.suptitle(comp_title, fontsize=22, y=0.95, fontweight='bold')
    plt.tight_layout()
    filename = f"heatmap_{component}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    print(f"   -> Saved Heatmap: {filename}")


def plot_combined_ratio_trend(data_map, output_dir):
    components = ['attn', 'mlp', 'state']
    if all(df.empty for df in data_map.values()):
        return

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    
    color_map = {
        "KNOWN": "#006400",     
        "AMBIGUOUS": "#1F77B4", 
        "UNKNOWN": "#D35400"    
    }
    markers_dict = {"KNOWN": "o", "AMBIGUOUS": "s", "UNKNOWN": "X"}
    dataset_order = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
    titles_map = {'attn': 'Attention', 'mlp': 'MLP', 'state': 'State'}

    max_y_val = 0
    all_sizes = set()
    for df in data_map.values():
        if not df.empty:
            max_y_val = max(max_y_val, df["Layer Ratio (%)"].max())
            all_sizes.update(df['Size'].unique())
    unique_sizes = sorted(list(all_sizes))

    for idx, (ax, comp) in enumerate(zip(axes, components)):
        df = data_map[comp]
        ax.set_title(titles_map.get(comp, comp.upper()), fontsize=22, fontweight='bold', pad=15)
        
        if df.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        ax.set_xscale("log")
        ax.set_xticks(unique_sizes)
        
        ax.set_xticklabels([f"{s:g}B" for s in unique_sizes], fontsize=15, fontweight='bold')

        present_datasets = [d for d in dataset_order if d in df['Dataset'].unique()]
        
        for ds in present_datasets:
            subset = df[df['Dataset'] == ds]
            color = color_map.get(ds, "#333333")
            marker = markers_dict.get(ds, "o")
            
            sns.regplot(
                data=subset, x="Size", y="Layer Ratio (%)",
                ax=ax, logx=True, ci=80,
                scatter=False, 
                line_kws={'linewidth': 3, 'alpha': 0.6, 'color': color}
            )
            ax.scatter(
                subset["Size"], subset["Layer Ratio (%)"],
                c=color, marker=marker, s=450,
                edgecolor='white', linewidth=1.5, alpha=0.7,
                zorder=10
            )
            for _, row in subset.iterrows():
                x_pos = row['Size']
                y_pos = row['Layer Ratio (%)']
                count = int(row['Layer Count (K)'])
                txt = ax.text(x_pos, y_pos, f"{count}", 
                              fontsize=13, ha='center', va='center', fontweight='bold', color='white',
                              zorder=11)
                txt.set_path_effects([pe.withStroke(linewidth=2, foreground='black', alpha=0.7)])

        ax.grid(False) 
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.xaxis.grid(True, which='major', linestyle='--', alpha=0.4)
        ax.xaxis.grid(False, which='minor')

        if idx == 0:
            ax.tick_params(axis='y', labelsize=15)
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')

    axes[0].set_ylim(0, max_y_val * 1.35)
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100.0))
    
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    fig.text(0.5, 0.05, "Model Parameters (Log Scale)", ha='center', fontsize=20, fontweight='bold')
    fig.text(0.02, 0.5, "Layer Ratio (%)", va='center', rotation='vertical', fontsize=20, fontweight='bold')
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['KNOWN'], 
               markersize=13, label='KNOWN'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_map['AMBIGUOUS'], 
               markersize=13, label='AMBIGUOUS'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor=color_map['UNKNOWN'], 
               markersize=13, label='UNKNOWN')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', 
                   fontsize=12, frameon=True, framealpha=0.9, edgecolor='#ccc')

    axes[0].text(0.03, 0.96, "Numbers inside points indicate\nabsolute layer count (e.g., '6' layers)", 
            transform=axes[0].transAxes, fontsize=14, verticalalignment='top', 
            bbox=dict(boxstyle="round", fc="white", ec="#dddddd", alpha=0.9, zorder=100))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13, top=0.92, left=0.08, wspace=0.08)
    
    filename = "trend_combined_ratio.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"   -> Saved Combined Trend: {filename}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="main/causal_trace/results")
    parser.add_argument("--output_dir", type=str, default="main/causal_trace/aggregated/results")
    parser.add_argument("--models", nargs='+', 
                        default=["qwen2.5_0.5B", "qwen2.5_1.5B", "qwen2.5_3B", 
                                 "qwen2.5_7B", "qwen2.5_14B", "qwen2.5_32B", "qwen2.5_72B"])
    args = parser.parse_args()
    
    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)
    
    components = ['attn', 'mlp', 'state']
    data_map = {}

    print(f"Starting analysis [Heatmap + Combined Trend]...")
    
    for comp in components:
        print(f"\nProcessing Component: [{comp.upper()}]")
        trace_data, df_sparsity = load_data(args.base_dir, args.models, target_component=comp)
        data_map[comp] = df_sparsity
        
        if trace_data:
            plot_heatmap(trace_data, args.output_dir, component=comp)
        else:
            print(f"   [!] No data found for {comp}.")
            
    print(f"\nGenerating Combined Ratio Trend Plot...")
    plot_combined_ratio_trend(data_map, args.output_dir)
            
    print("\n>>> All tasks completed.")

if __name__ == "__main__":
    main()