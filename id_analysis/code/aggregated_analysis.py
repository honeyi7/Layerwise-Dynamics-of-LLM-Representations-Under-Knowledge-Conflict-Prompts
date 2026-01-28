import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import warnings
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
import matplotlib.font_manager as font_manager


warnings.filterwarnings("ignore")

BASE_DIR = "main/id_analysis/results"
OUTPUT_DIR = "main/id_analysis/aggregated_results" 
MODEL_ORDER = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'], 
    'axes.unicode_minus': False,
    'font.weight': 'bold',          
    'axes.labelweight': 'bold',    
    'axes.titleweight': 'bold',    
    'figure.titleweight': 'bold',   
    'font.size': 14,               
    'axes.labelsize': 14,         
    'axes.titlesize': 14,           
    'xtick.labelsize': 14,         
    'ytick.labelsize': 14,         
    'legend.fontsize': 14,         
    'legend.title_fontsize': 14,    
})

sns.set_context("paper", rc={"font.size":14,"axes.titlesize":14,"axes.labelsize":14})
sns.set_style("ticks") 

# ================= core algorithm =================

def _compute_id_2NN_exact(mus, fraction=0.95):
    N = mus.shape[0]
    if N == 0: return np.nan
    N_eff = int(N * fraction)
    mus = np.maximum(mus, 1e-10) 
    log_mus = np.log(mus)
    log_mus_reduced = np.sort(log_mus)[:N_eff]
    y = -np.log(1 - np.arange(1, N_eff + 1) / N)

    def func(x, m):
        return m * x

    try:
        popt, _ = curve_fit(func, log_mus_reduced, y, p0=[1.0], maxfev=5000)
        return popt[0]
    except:
        return np.nan

def compute_id_from_matrix(X):
    X = np.nan_to_num(X)
    if X.shape[0] < 10: return np.nan 
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='euclidean').fit(X)
    distances, _ = nbrs.kneighbors(X)
    r1, r2 = distances[:, 1], distances[:, 2]
    mask = r1 > 1e-10
    if np.sum(mask) < 5: return np.nan
    mus = r2[mask] / r1[mask]
    return _compute_id_2NN_exact(mus, fraction=0.95)

# ================= data loading =================

def load_and_align_curves(dataset_name, grid_points=200):
    print(f"\n[Status] Processing dataset: {dataset_name}...")
    aligned_data = {} 
    
    for model in MODEL_ORDER:
        curves = {}
        for run_type in ["baseline", "counterfactual"]:
            path = os.path.join(BASE_DIR, f"qwen2.5_{model}", "activations", dataset_name, run_type)
            if not os.path.exists(path): continue
            
            files = glob.glob(os.path.join(path, "*.npy"))
            if not files: continue

            layer_groups = {} 
            for f in files:
                fname = os.path.basename(f)
                match = re.search(r"layer_(\d+)", fname)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in layer_groups: layer_groups[layer_idx] = []
                    layer_groups[layer_idx].append(f)
            
            sorted_layers = sorted(layer_groups.keys())
            if not sorted_layers: continue
            
            y_vals = []
            for l_idx in sorted_layers:
                file_list = layer_groups[l_idx]
                arrays = []
                for f_path in file_list:
                    try: arrays.append(np.load(f_path))
                    except: pass
                
                if not arrays:
                    y_vals.append(np.nan)
                    continue
                
                try:
                    full_batch_data = np.concatenate(arrays, axis=0)
                    id_val = compute_id_from_matrix(full_batch_data)
                    y_vals.append(id_val)
                except:
                    y_vals.append(np.nan)

            curves[run_type] = np.array(y_vals)
        
        if "baseline" in curves and "counterfactual" in curves:
            min_len = min(len(curves["baseline"]), len(curves["counterfactual"]))
            if min_len < 5: continue
            
            raw_delta = curves["counterfactual"][:min_len] - curves["baseline"][:min_len]
            
            if np.isnan(raw_delta).any():
                nans = np.isnan(raw_delta)
                x_inds = lambda z: z.nonzero()[0]
                raw_delta[nans] = np.interp(x_inds(nans), x_inds(~nans), raw_delta[~nans])
            
            raw_delta = np.ascontiguousarray(raw_delta)
            try:
                win_len = min(7, min_len if min_len % 2 != 0 else min_len - 1)
                if win_len > 3: raw_delta = savgol_filter(raw_delta, win_len, 2)
            except: pass

            x_old = np.linspace(0, 1, min_len)
            x_new = np.linspace(0, 1, grid_points)
            f_interp = interp1d(x_old, raw_delta, kind='cubic', fill_value="extrapolate")
            aligned_data[model] = f_interp(x_new)
            
    return aligned_data

# ================= plotting =================

def plot_focused_collapse(data, ds_name):
    start_idx = 100 
    
    x_axis = np.linspace(0.5, 1.0, 100) 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={'width_ratios': [2, 1]})
    

    distinct_colors = [
        "#1F4E79", # 0.5B: Deep Navy Blue
        "#2E75B6", # 1.5B: Medium Blue
        "#4BACC6", # 3B:   Ocean Blue/Teal
        "#806000", # 7B:   Dark Olive/Gold (Neutral)
        "#C55A11", # 14B:  Burnt Orange
        "#C00000", # 32B:  Deep Red
        "#701414"  # 72B:  Very Dark Red (Blackish Red)
    ]
    
    if len(MODEL_ORDER) > len(distinct_colors):
        colors = sns.color_palette("rocket", n_colors=len(MODEL_ORDER))
    else:
        colors = distinct_colors[:len(MODEL_ORDER)]

    min_vals = []
    all_curve_values = []
    
    legend_handles = []
    legend_labels = []

    # --- Panel A ---
    for i, model in enumerate(MODEL_ORDER):
        if model not in data: 
            min_vals.append(np.nan)
            continue
        
        curve_segment = data[model][start_idx:]
        all_curve_values.extend(curve_segment)
        
        focus_region = curve_segment[50:] 
        val = np.min(focus_region) if len(focus_region) > 0 else np.nan
        min_vals.append(val)
        
        lw = 2.0 + (i * 0.3) 
        line, = ax1.plot(x_axis, curve_segment, label=f"{model}", color=colors[i], linewidth=lw, alpha=0.9, zorder=10+i)
        
        legend_handles.append(line)
        legend_labels.append(model)
    
    ax1.set_xlabel("Relative Depth", fontsize=14, fontweight='bold')
    ax1.set_ylabel(r"$\Delta$ Intrinsic Dimension ($ID_{ctf} - ID_{base}$)", fontsize=14, fontweight='bold')
    
    ax1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.setp(ax1.get_xticklabels(), fontsize=14, fontweight='bold')
    plt.setp(ax1.get_yticklabels(), fontsize=14, fontweight='bold')
    
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.axvspan(0.75, 1.0, color='#E0E0E0', alpha=0.5, label="Collapse Region", zorder=0)

    if all_curve_values:
        y_min, y_max = min(all_curve_values), max(all_curve_values)
        y_range = y_max - y_min
        ax1.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

    ax1.grid(True, linestyle=':', alpha=0.5)

    # --- Panel B ---
    valid_indices = [i for i, v in enumerate(min_vals) if not np.isnan(v)]
    valid_models = [MODEL_ORDER[i] for i in valid_indices]
    valid_mins = [min_vals[i] for i in valid_indices]
    
    x_pos = np.arange(len(valid_models))
    
    if len(x_pos) > 0:
        bars = ax2.bar(x_pos, valid_mins, color=[colors[i] for i in valid_indices], edgecolor='black', alpha=0.85, width=0.7)
        
        try:
            z = np.polyfit(x_pos, valid_mins, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(x_pos.min(), x_pos.max(), 300)
            ax2.plot(x_smooth, p(x_smooth), color="#333333", linestyle="--", linewidth=2, alpha=0.8, label="Quadratic Fit")
            y_pred = p(x_pos)
            r2 = r2_score(valid_mins, y_pred)
            print(f"[{ds_name}] Fitting R2: {r2:.3f}") 
        except: pass
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height - (0.15 if height < 0 else -0.15), 
                     f'{height:.2f}',
                     ha='center', va='top' if height < 0 else 'bottom', 
                     fontsize=14, fontweight='bold', color='black') 

        b_min, b_max = min(valid_mins), max(valid_mins)
        b_min = min(b_min, 0)
        b_max = max(b_max, 0)
        b_range = b_max - b_min
        if b_range == 0: b_range = 1
        ax2.set_ylim(b_max + 0.2 * b_range, b_min - 0.2 * b_range) 
    else:
        ax2.invert_yaxis()

    ax2.set_ylabel(r"Max Drop Depth (Min $\Delta$ ID)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Model Scale", fontsize=14, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(valid_models, rotation=45, ha='right')
    
    plt.setp(ax2.get_xticklabels(), fontsize=14, fontweight='bold')
    plt.setp(ax2.get_yticklabels(), fontsize=14, fontweight='bold')
    
    ax2.axhline(0, color='black', linewidth=1)
    
    if len(x_pos) > 0:
        ax2.set_ylim(bottom=b_min - 0.2 * b_range, top=b_max + 0.1 * b_range)
        ax2.invert_yaxis()

    title_font = font_manager.FontProperties(family='serif', weight='bold', size=14)
    
    leg = fig.legend(legend_handles, legend_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.01), 
               ncol=7, 
               fontsize=14,
               frameon=False, 
               title="Model Scale (Increasing Size $\longrightarrow$)", 
               title_fontproperties=title_font
              )
    
    for text in leg.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18) 
    
    save_path = os.path.join(OUTPUT_DIR, f"Focus_Collapse_{ds_name}.png")
    plt.savefig(save_path, dpi=300)
    print(f"  [Plot Saved] {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ds in ["known", "unknown", "ambiguous"]:
        data = load_and_align_curves(ds)
        if data:
            plot_focused_collapse(data, ds)
        else:
            print(f"  [Info] No valid data found for {ds}")

if __name__ == "__main__":
    main()