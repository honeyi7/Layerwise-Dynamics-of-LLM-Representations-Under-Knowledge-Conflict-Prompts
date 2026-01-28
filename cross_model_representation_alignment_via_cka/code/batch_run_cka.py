import os
import sys
import subprocess

RESULTS_ROOT = "main/id_analysis/results"
OUTPUT_DIR = "main/cross_model_representation_alignment_via_cka/cka_plots"
CKA_SCRIPT = "cka_analysis_main.py"

DATASET_ROOT_BASE = "main/knowledge_probe/result"

MODELS = [
    ("qwen2.5_0.5B", "0.5B"),
    ("qwen2.5_1.5B", "1.5B"),
    ("qwen2.5_3B",   "3B"),
    ("qwen2.5_7B",   "7B"),
    ("qwen2.5_14B",  "14B"),
    ("qwen2.5_32B",  "32B"),
    ("qwen2.5_72B",  "72B"),
]

DATASETS = ["known", "unknown", "ambiguous"]
RUN_TYPES = ["counterfactual", "baseline"]

# =======================================

def get_jsonl_path(model_name_folder, dataset_type):
    """
    Returns the absolute path of the corresponding jsonl file based on the model folder name and dataset type.
    !!! You need to modify this function according to your actual file storage location !!!
    """
    # Assume your jsonl is in the results directory under the corresponding model folder?
    # Or in a separate dataset directory?
    # For example: D:\ICL\data\qwen2.5_7B\known.jsonl

    # Assume the jsonl is in the results directory (if not, modify base_path)
    # For example, many times the jsonl is passed via args for running the script
    # You need to find the jsonl path used when running id_analysis

    # Assume the path structure is as follows: D:\ICL\datasets\{model_name}\{dataset_type}.jsonl
    # Note: model_name_folder might be "qwen2.5_0.5B"

    path = os.path.join(DATASET_ROOT_BASE, model_name_folder, f"{dataset_type}_bucket.jsonl")
    
    if not os.path.exists(path):
        print(f"❌ Warning: JSONL not found at {path}")
        return None
    return path

def main():
    target_folder, target_name = MODELS[-1] # 72B
    source_models = MODELS[:-1]             # Others

    for src_folder, src_name in source_models:
        for ds in DATASETS:
            # Get the JSONL file path
            jsonl_a = get_jsonl_path(src_folder, ds)
            jsonl_b = get_jsonl_path(target_folder, ds)
            
            if not jsonl_a or not jsonl_b:
                print(f"Skipping {src_name} vs {target_name} ({ds}) due to missing JSONL.")
                continue

            for rt in RUN_TYPES:
                print(f"\nComparing {src_name} vs {target_name} | {ds} | {rt}")
                
                cmd = [
                    sys.executable, CKA_SCRIPT,
                    "--model_a_root", os.path.join(RESULTS_ROOT, src_folder),
                    "--model_a_name", src_name,
                    "--jsonl_a", jsonl_a,
                    
                    "--model_b_root", os.path.join(RESULTS_ROOT, target_folder),
                    "--model_b_name", target_name,
                    "--jsonl_b", jsonl_b,
                    
                    "--dataset", ds,
                    "--run_type", rt,
                    "--output_dir", os.path.join(OUTPUT_DIR, f"{src_name}_vs_{target_name}")
                ]
                
                subprocess.run(cmd)

if __name__ == "__main__":
    main()