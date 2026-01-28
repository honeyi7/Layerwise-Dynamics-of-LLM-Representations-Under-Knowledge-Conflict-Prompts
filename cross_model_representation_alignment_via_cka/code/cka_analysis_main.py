#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
import torch
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==========================================
# Index alignment based on JSONL content
# ==========================================

def get_alignment_indices(path_a, path_b):
    """
    Read two JSONL files, find the common samples (based on prompt content),
    and return two sets of indices: (indices_a, indices_b).
    """

    print(f"🔍 Aligning datasets based on content...")
    print(f"  - File A: {path_a}")
    print(f"  - File B: {path_b}")

    def load_keys(path):
        keys = {} # content -> index
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    # Construct a unique identifier: Prompt + Subject
                    prompt = item['requested_rewrite']['prompt']
                    subject = item['requested_rewrite']['subject']
                    unique_key = prompt.format(subject)
                    keys[unique_key] = idx
                except:
                    continue
        return keys

    map_a = load_keys(path_a)
    map_b = load_keys(path_b)
    
    common_keys = set(map_a.keys()) & set(map_b.keys())
    
    if len(common_keys) == 0:
        raise ValueError("No common samples found between the two datasets.")

    # Export indices in a fixed order
    sorted_keys = sorted(list(common_keys))
    
    indices_a = [map_a[k] for k in sorted_keys]
    indices_b = [map_b[k] for k in sorted_keys]
    
    return indices_a, indices_b

# ==========================================
# CKA computation core (GPU Accelerated)
# ==========================================

class CKA_Calculator:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        L_X_c = self.centering(L_X)
        L_Y_c = self.centering(L_Y)
        return torch.sum(L_X_c * L_Y_c)

    def linear_CKA(self, X, Y):
        # Ensure the input is converted to Tensor and moved to GPU
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float().to(self.device)
        if isinstance(Y, np.ndarray): Y = torch.from_numpy(Y).float().to(self.device)
        
        # Flatten excess dimensions, keeping the Batch dimension
        if X.dim() > 2: X = X.view(X.shape[0], -1)
        if Y.dim() > 2: Y = Y.view(Y.shape[0], -1)

        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        
        return (hsic / (var1 * var2)).item()

# ==========================================
# data loader
# ==========================================

def load_full_layer_data(base_dir, layer_idx):
    single_file = os.path.join(base_dir, f"layer_{layer_idx:03d}_single.npy")
    if os.path.exists(single_file):
        return np.load(single_file)
    
    worker_files = glob.glob(os.path.join(base_dir, f"layer_{layer_idx:03d}_worker_*.npy"))
    if worker_files:
        worker_files.sort(key=lambda x: int(x.split('_worker_')[-1].replace('.npy', '')))
        arrays = [np.load(f) for f in worker_files]
        return np.concatenate(arrays, axis=0)
    
    return None

def load_layer_data_filtered(base_dir, layer_idx, valid_indices):
    full_data = load_full_layer_data(base_dir, layer_idx)
    if full_data is None: return None
    return full_data[valid_indices]

def get_valid_layers(base_dir):
    files = glob.glob(os.path.join(base_dir, "layer_*.npy"))
    if not files: return []
    layers = set()
    for f in files:
        try:
            layers.add(int(os.path.basename(f).split('_')[1]))
        except: continue
    return sorted(list(layers))



def run_cka_comparison(args):
    # 1. Determine the directory
    dir_a = os.path.join(args.model_a_root, "activations", args.dataset, args.run_type)
    dir_b = os.path.join(args.model_b_root, "activations", args.dataset, args.run_type)
    
    # 2. Get aligned indices (based on the full JSONL dataset)
    if not os.path.exists(args.jsonl_a) or not os.path.exists(args.jsonl_b):
        print(f"❌ Error: JSONL files not found.")
        return

    candidate_indices_a, candidate_indices_b = get_alignment_indices(args.jsonl_a, args.jsonl_b)
    print(f"📋 Candidate common samples from JSONL: {len(candidate_indices_a)}")

    # 3. Check the actual length of the .npy file (to prevent out-of-bounds errors)
    layers_a = get_valid_layers(dir_a)
    layers_b = get_valid_layers(dir_b)
    
    if not layers_a or not layers_b:
        print("❌ Error: No activation layers found.")
        return

    # Preload the first layer of data to check the actual length
    print("📏 Checking actual .npy data size...")
    sample_a = load_full_layer_data(dir_a, layers_a[0])
    sample_b = load_full_layer_data(dir_b, layers_b[0])
    
    if sample_a is None or sample_b is None:
        print("❌ Error: Failed to load sample layer data.")
        return

    max_len_a = sample_a.shape[0]
    max_len_b = sample_b.shape[0]
    print(f"  - Model A extracted samples: {max_len_a}")
    print(f"  - Model B extracted samples: {max_len_b}")
    
    del sample_a
    del sample_b

    # 4. Filter indices: must satisfy idx < max_len
    final_indices_a = []
    final_indices_b = []
    
    for ia, ib in zip(candidate_indices_a, candidate_indices_b):
        if ia < max_len_a and ib < max_len_b:
            final_indices_a.append(ia)
            final_indices_b.append(ib)
            
    print(f"✅ Final aligned samples after truncation check: {len(final_indices_a)} (Dropped {len(candidate_indices_a) - len(final_indices_a)})")
    
    if len(final_indices_a) < 5:
        print("⚠️ Warning: Too few samples (<5) to calculate CKA. Skipping.")
        return

    # 5. CKA computation
    cka = CKA_Calculator()
    print(f"🚀 Computing CKA on {cka.device}...")
    
    heatmap_matrix = np.zeros((len(layers_b), len(layers_a)))
    
    for j, idx_b in enumerate(tqdm(layers_b, desc="Rows (Model B)")):
        data_b = load_layer_data_filtered(dir_b, idx_b, final_indices_b)
        if data_b is None: continue
        
        for i, idx_a in enumerate(layers_a):
            data_a = load_layer_data_filtered(dir_a, idx_a, final_indices_a)
            if data_a is None: continue
            
            val = cka.linear_CKA(data_a, data_b)
            heatmap_matrix[j, i] = val
            
    # 6. plot
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_matrix, cmap="magma", xticklabels=5, yticklabels=5, vmin=0.0, vmax=1.0)
    
    ax.set_xlabel(f"{args.model_a_name} Layers")
    ax.set_ylabel(f"{args.model_b_name} Layers")
    ax.set_title(f"Aligned CKA: {args.dataset} ({args.run_type})\n(N={len(final_indices_a)})")
    
    xticks = np.arange(0, len(layers_a), 5)
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels([layers_a[i] for i in xticks], rotation=0)
    
    yticks = np.arange(0, len(layers_b), 5)
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels([layers_b[i] for i in yticks], rotation=0)
    
    ax.invert_yaxis()
    
    save_path = os.path.join(args.output_dir, f"CKA_Aligned_{args.model_a_name}_vs_{args.model_b_name}_{args.dataset}_{args.run_type}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    np.save(save_path.replace('.png', '.npy'), heatmap_matrix)
    print(f"Saved plot and data: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_root", type=str, required=True)
    parser.add_argument("--model_a_name", type=str, required=True)
    parser.add_argument("--jsonl_a", type=str, required=True)
    
    parser.add_argument("--model_b_root", type=str, required=True)
    parser.add_argument("--model_b_name", type=str, required=True)
    parser.add_argument("--jsonl_b", type=str, required=True)
    
    parser.add_argument("--dataset", type=str, default="known")
    parser.add_argument("--run_type", type=str, default="counterfactual")
    parser.add_argument("--output_dir", type=str, default="./cka_results_aligned")
    
    args = parser.parse_args()
    run_cka_comparison(args)