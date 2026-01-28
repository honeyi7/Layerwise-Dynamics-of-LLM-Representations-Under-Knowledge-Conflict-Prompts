import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "main/check_logits/results"

MODELS = [
    ("0.5B", "qwen2.5_0.5B"),
    ("1.5B", "qwen2.5_1.5B"),
    ("3B",   "qwen2.5_3B"),
    ("7B",   "qwen2.5_7B"),
    ("14B",  "qwen2.5_14B"),
    ("32B",  "qwen2.5_32B"),
    ("72B",  "qwen2.5_72B"),
]

CATEGORIES = ["known", "ambiguous", "unknown"]
OUTPUT_FILE = "main/check_logits/aggregated_result/refined_scaling_analysis_bold.png"

PALETTE = {
    "known": "#2ca25f",      
    "ambiguous": "#1f78b4",   
    "unknown": "#d95f0e"    
}

MARKERS = {
    "known": "o", 
    "unknown": "X", 
    "ambiguous": "s"
}
# ===========================================

def load_data():
    records = []
    print(f"Loading data from {BASE_DIR}...")
    for model_label, folder_name in MODELS:
        pkl_path = os.path.join(BASE_DIR, folder_name, "baseline_stats.pkl")
        if not os.path.exists(pkl_path):
            continue
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            for cat in CATEGORIES:
                if cat in data:
                    diffs = data[cat]['diffs']
                    for val in diffs:
                        records.append({
                            "Model Scale": model_label,
                            "Category": cat,
                            "Logit Difference": val
                        })
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
    return pd.DataFrame(records)

def plot_analysis(df):

    sns.set_theme(style="white", context="paper", font_scale=1.8)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # ================= left plot: mean evolution =================
    sns.lineplot(
        data=df,
        x="Model Scale",
        y="Logit Difference",
        hue="Category",
        style="Category",
        markers=MARKERS,
        dashes=False,
        palette=PALETTE,
        ax=axes[0],
        linewidth=4,     
        markersize=16,     
        err_style="band"
    )
    
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.6)
    axes[0].xaxis.grid(False)
    
    axes[0].set_ylabel("Mean Value of Logit Differences", fontsize=22, fontweight='bold', labelpad=15)
    axes[0].set_xlabel("")


    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(
        h, l, 
        title="Knowledge Category", 
        loc='lower right', 
        frameon=True, 
        fancybox=True, 
        markerscale=1.0,   
        fontsize=18,      
        title_fontsize=18  
    )

    # ================= right plot: distribution robustness =================
    sns.boxplot(
        data=df,
        x="Model Scale",
        y="Logit Difference",
        hue="Category",
        palette=PALETTE,
        ax=axes[1],
        showfliers=False,
        width=0.7,
        linewidth=2.5     
    )
    
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.6)
    axes[1].xaxis.grid(False)

    axes[1].set_ylabel("Distribution of Logit Differences", fontsize=22, fontweight='bold', labelpad=15)
    axes[1].set_xlabel("")
    
    if axes[1].get_legend():
        axes[1].get_legend().remove()


    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
        
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')


    fig.supxlabel("Model Scale (Qwen2.5-Instruct Series)", fontsize=24, fontweight='bold', y=0.02)

    plt.tight_layout()

    plt.subplots_adjust(bottom=0.15, wspace=0.18)
    
    save_path = os.path.abspath(OUTPUT_FILE)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

def main():
    df = load_data()
    if not df.empty:
        plot_analysis(df)
    else:
        print("No data found.")

if __name__ == "__main__":
    main()