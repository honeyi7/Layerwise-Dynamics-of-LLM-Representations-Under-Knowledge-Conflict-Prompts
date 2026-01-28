import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# The directory where you store the probing results
RESULTS_DIR = "main/linear_probe_experiment/results_500epochs_100patience"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "aggregated_analysis")

MODELS = [
    "qwen2.5_0.5B", 
    "qwen2.5_1.5B", 
    "qwen2.5_3B", 
    "qwen2.5_7B", 
    "qwen2.5_14B", 
    "qwen2.5_32B", 
    "qwen2.5_72B"
]


sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans' 

def load_and_normalize_data():
    all_data = []
    
    for model_name in tqdm(MODELS, desc="Loading Data"):
        file_path = os.path.join(RESULTS_DIR, model_name, "metrics_debiased.csv")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Calculate the relative depth (0.0 - 1.0)
        max_layer = df['layer'].max()
        df['relative_depth'] = df['layer'] / max_layer
        df['model'] = model_name
        
        all_data.append(df)
        
    if not all_data:
        raise ValueError("No data loaded!")
        
    return pd.concat(all_data, ignore_index=True)

def plot_aggregated_trends(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 14,
        'figure.titleweight': 'bold'
    })
    
    palette = sns.color_palette("flare", n_colors=len(MODELS))
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # ================= left: Knowledge Boundary =================
    sns.lineplot(
        data=df, 
        x="relative_depth", 
        y="acc_baseline", 
        hue="model", 
        palette=palette,
        linewidth=2.5,
        alpha=0.85,
        ax=axes[0],
        legend=False  
    )
    
    axes[0].set_xlabel("Relative Depth", fontweight='bold', fontsize=14)
    axes[0].set_ylabel("Probe Accuracy", fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    for tick in axes[0].get_xticklabels():
        tick.set_fontweight('bold')
    for tick in axes[0].get_yticklabels():
        tick.set_fontweight('bold')

    # ================= right: Suppression Rate =================
    sns.lineplot(
        data=df, 
        x="relative_depth", 
        y="suppression_rate", 
        hue="model", 
        palette=palette,
        linewidth=2.5,
        alpha=0.85,
        ax=axes[1]
    )
    
    axes[1].set_xlabel("Relative Depth", fontweight='bold', fontsize=14)
    axes[1].set_ylabel("Suppression Rate", fontweight='bold', fontsize=14)
    
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    
    for tick in axes[1].get_xticklabels():
        tick.set_fontweight('bold')
    for tick in axes[1].get_yticklabels():
        tick.set_fontweight('bold')

    # ================= Legend settings =================
    handles, labels = axes[1].get_legend_handles_labels()
    
    leg = axes[1].legend(
        handles=handles,
        labels=labels,
        title="Model Size", 
        loc='lower right', 
        bbox_to_anchor=(1, 0), 
        framealpha=0.4,       
        edgecolor='gray',     
        fancybox=True         
    )
    
    plt.setp(leg.get_texts(), fontweight='bold')
    plt.setp(leg.get_title(), fontweight='bold')
    leg._legend_box.align = "left"

    # ================= save and clear =================
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "aggregated_combined_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Combined plot saved to {save_path}")

def analyze_key_metrics(df):
    """
    calculate some key metrcs and save as a table
    """
    summary = []
    
    for model in MODELS:
        m_df = df[df['model'] == model]
        if m_df.empty: continue
        
        avg_acc = m_df[m_df['relative_depth'] > 0.2]['acc_baseline'].mean()
        
        max_suppression = m_df['suppression_rate'].max()
        peak_loc = m_df.loc[m_df['suppression_rate'].idxmax(), 'relative_depth']
        
        final_suppression = m_df.iloc[-1]['suppression_rate']
        
        awakening_score = max_suppression - final_suppression
        
        summary.append({
            "Model": model,
            "Avg_Boundary_Acc": round(avg_acc, 3),
            "Max_Suppression": round(max_suppression, 3),
            "Peak_Location": round(peak_loc, 2),
            "Final_Suppression": round(final_suppression, 3),
            "Awakening_Score": round(awakening_score, 3)
        })
    
    sum_df = pd.DataFrame(summary)
    sum_path = os.path.join(OUTPUT_DIR, "trend_summary.csv")
    sum_df.to_csv(sum_path, index=False)
    print("Summary table saved.")
    print(sum_df)

if __name__ == "__main__":
    try:
        data = load_and_normalize_data()
        plot_aggregated_trends(data)
        analyze_key_metrics(data)
    except Exception as e:
        print(f"An error occurred: {e}")