import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

CKA_RESULT_DIR = "main/cross_model_representation_alignment_via_cka/cka_plots"
TARGET_MODEL = "72B"
MODELS = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
DATASETS = ["known", "ambiguous", "unknown"]

# --- Config for Butterfly Plot (Script 1) ---
OUTPUT_DIR_TOPOLOGY = os.path.join(CKA_RESULT_DIR, "aggregrated_results")
TRANSITION_THRESHOLD = 0.5 

# --- Config for Mapping Plot (Script 2) ---
OUTPUT_DIR_MAPPING = os.path.join(CKA_RESULT_DIR, "aggregrated_results")
RUN_TYPES = ["counterfactual", "baseline"]
# ==================================================================

def load_matrix(model_name, dataset, run_type):
    """
    Unified function to load CKA matrices.
    """
    pattern = f"**/*{model_name}_vs_{TARGET_MODEL}_{dataset}_{run_type}.npy"
    files = glob.glob(os.path.join(CKA_RESULT_DIR, pattern), recursive=True)
    if not files: return None
    return np.load(max(files, key=os.path.getmtime))

# ==================================================================
#                 PART 1: Butterfly / Trade-off Plot
# ==================================================================

def calculate_metrics(mat_corr, mat_cont):
    best_match_corr = np.argmax(mat_corr, axis=0) / mat_corr.shape[0]
    best_match_cont = np.argmax(mat_cont, axis=0) / mat_cont.shape[0]
    
    # Normalize sampling
    common_x = np.linspace(0, 1, 100)
    y_corr = np.interp(common_x, np.linspace(0, 1, len(best_match_corr)), best_match_corr)
    y_cont = np.interp(common_x, np.linspace(0, 1, len(best_match_cont)), best_match_cont)
    
    # --- Metric 1: Surplus ---
    diff = y_cont - y_corr
    positive_diff = np.maximum(0, diff)
    surplus = np.mean(positive_diff)
    
    # --- Metric 2: Delay ---
    def get_transition(y_curve):
        for i, y in enumerate(y_curve):
            if i < len(y_curve)-3 and np.all(y_curve[i:i+3] >= TRANSITION_THRESHOLD):
                return common_x[i]
        return 1.0

    trans_corr = get_transition(y_corr)
    trans_cont = get_transition(y_cont)
    delay = trans_cont - trans_corr
    
    return surplus, delay

def plot_positive_tradeoff():
    print("\n[Part 1] Generating Butterfly Trade-off Plot...")
    os.makedirs(OUTPUT_DIR_TOPOLOGY, exist_ok=True)
    
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    
    data = []
    print("  - Calculating Metrics...")
    
    for dataset in DATASETS:
        for model in MODELS:
            mat_corr = load_matrix(model, dataset, "baseline")
            mat_cont = load_matrix(model, dataset, "counterfactual")
            if mat_corr is None or mat_cont is None: continue
            
            surplus, delay = calculate_metrics(mat_corr, mat_cont)
            
            data.append({
                "Model": model,
                "Dataset": dataset,
                "Positive Surplus": surplus,
                "Alignment Delay": delay
            })
    
    df = pd.DataFrame(data)
    df_avg = df.groupby("Model")[["Positive Surplus", "Alignment Delay"]].mean().reset_index()
    df_avg["Model"] = pd.Categorical(df_avg["Model"], categories=MODELS, ordered=True)
    df_avg = df_avg.sort_values("Model")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    plt.subplots_adjust(wspace=0.08)
    
    y_pos = np.arange(len(MODELS))
    
    # left plot
    bars1 = ax1.barh(y_pos, df_avg["Positive Surplus"], color="#1abc9c", alpha=0.85, height=0.6)
    
    max_val = df_avg["Positive Surplus"].max()
    ax1.set_xlim(0, max_val * 1.3)
    ax1.invert_xaxis()
    
    ax1.set_title("Surplus ($\int \Delta y^{+}$)", fontsize=14, weight='bold', color="#16a085")
    
    for rect in bars1:
        w = rect.get_width()
        label = f"+{w:.3f}" if w > 0.001 else "~0"
        ax1.text(w + (max_val*0.02), rect.get_y() + rect.get_height()/2, label, 
                 ha='right', va='center', fontsize=11, weight='bold')
    
    # right plot
    bars2 = ax2.barh(y_pos, df_avg["Alignment Delay"], color="#c0392b", alpha=0.85, height=0.6)
    ax2.set_xlim(0, 0.35) 
    ax2.set_title("Delay ($\Delta Depth$)", fontsize=14, weight='bold', color="#c0392b")
    
    for rect in bars2:
        w = rect.get_width()
        if abs(w) > 0.005: 
            label = f"+{w:.2f}" if w > 0 else f"{w:.2f}"
            color = 'black' if w > 0 else 'gray'
            ax2.text(w + 0.005, rect.get_y() + rect.get_height()/2, label, 
                     ha='left', va='center', color=color, fontsize=11, weight='bold')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(MODELS, fontsize=12, weight='bold')
    ax2.tick_params(left=False)
    
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='black', linewidth=1, alpha=0.3)
    
    
    

    save_path = os.path.join(OUTPUT_DIR_TOPOLOGY, f"Final_Butterfly_Positive_{TRANSITION_THRESHOLD}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved Final Chart: {save_path}")

# ==================================================================
#                 PART 2: Layer Mapping Plots
# ==================================================================

def normalize_indices(num_layers):
    return np.linspace(0, 1, num_layers)

def plot_paper_figure():
    print("\n[Part 2] Generating Layer Mapping Plots...")
    os.makedirs(OUTPUT_DIR_MAPPING, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    palette = sns.color_palette("flare", len(MODELS))
    
    for run_type in RUN_TYPES:
        print(f"  🎨 Drawing Final Plot for: {run_type.upper()}...")
        
        fig = plt.figure(figsize=(18, 7))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        
        fig.suptitle(f"Layer Function Mapping: {run_type.capitalize()} Setting", 
                     fontsize=22, weight='bold', y=0.98)
        
        lines = []
        
        for i, dataset in enumerate(DATASETS):
            ax = plt.subplot(gs[i])
            
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            
            for idx, model in enumerate(MODELS):
                matrix = load_matrix(model, dataset, run_type)
                if matrix is None: continue
                
                # Calculate the optimal matching layer
                best_match_indices = np.argmax(matrix, axis=0)
                
                x_axis = normalize_indices(len(best_match_indices))
                y_axis = best_match_indices / matrix.shape[0]
                
                line, = ax.plot(x_axis, y_axis, 
                         label=f"{model}", 
                         color=palette[idx], 
                         linewidth=2.5,
                         alpha=0.9) 
                
                if i == 0: 
                    lines.append(line)

            dataset_display = dataset.capitalize()
            ax.set_title(f"{dataset_display}", fontsize=18, weight='bold', pad=10)
            
            ax.set_xlabel(f"Relative Depth (Small Model)", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            if i == 0:
                ax.set_ylabel(f"Matched Relative Depth ({TARGET_MODEL})", fontsize=14)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
        
        dummy_line = plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        
        if lines:
            legend_handles = [dummy_line] + lines
            legend_labels = ["Ideal Linear Scaling"] + MODELS
            
            fig.legend(legend_handles, legend_labels, 
                       loc='lower center', 
                       bbox_to_anchor=(0.5, 0.02), 
                       ncol=len(MODELS) + 1,       
                       frameon=False, 
                       fontsize=16)
        
        plt.subplots_adjust(top=0.88, bottom=0.18, left=0.06, right=0.98, wspace=0.15)
        
        save_name = f"Final_Mapping_{run_type}.png"
        save_path = os.path.join(OUTPUT_DIR_MAPPING, save_name)
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  ✅ Saved: {save_path}")

# ==================================================================
#                       Main Execution
# ==================================================================

if __name__ == "__main__":
    # Execute Task 1
    plot_positive_tradeoff()
    
    # Execute Task 2
    plot_paper_figure()
    
    print("\n✨ All plotting tasks completed successfully.")