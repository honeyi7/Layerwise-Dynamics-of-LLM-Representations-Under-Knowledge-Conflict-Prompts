#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit

# ==========================================
# ID computation tools
# ==========================================

def _compute_id_2NN_exact(mus, fraction=0.95, algorithm="base"):
    N = mus.shape[0]
    if N == 0: return np.nan
    
    N_eff = int(N * fraction)
    log_mus = np.log(mus)
    log_mus_reduced = np.sort(log_mus)[:N_eff]

    if algorithm == "ml":
        intrinsic_dim = (N - 1) / np.sum(log_mus)
    elif algorithm == "base":
        # Fit y = m * x
        y = -np.log(1 - np.arange(1, N_eff + 1) / N)

        def func(x, m):
            return m * x

        try:
            intrinsic_dim, _ = curve_fit(func, log_mus_reduced, y)
            return intrinsic_dim[0]
        except:
            return np.nan
    else:
        raise ValueError("Please select a valid algorithm type")
    return intrinsic_dim

def compute_id_layer(X):
    N = X.shape[0]
    if N < 5: return 0.0
    
    # Use brute force to search for nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='euclidean').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    mask = r1 > 1e-10
    if np.sum(mask) < 5: return 0.0
    
    mus = r2[mask] / r1[mask]
    
    return _compute_id_2NN_exact(mus, fraction=0.95)


def load_dataset(path, max_samples=None):
    dataset = []
    if not os.path.exists(path): return dataset
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and max_samples > 0 and i >= max_samples: break
            try: dataset.append(json.loads(line))
            except: continue
    return dataset

# ==========================================
# Step 1: extract activations
# ==========================================

def extract_step(args):
    print(f"Loading Model: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    datasets = {
        "known": args.know_data_path,
        "unknown": args.unknown_data_path, 
        "ambiguous": args.ambiguous_data_path
    }
    
    for ds_name, ds_path in datasets.items():
        if not ds_path or not os.path.exists(ds_path):
            print(f"Dataset skipped: {ds_name}")
            continue
        
        data = load_dataset(ds_path, args.max_samples)
        print(f"Processing {ds_name}: {len(data)} samples")

        buffers = {"baseline": {}, "counterfactual": {}}
        
        for i in tqdm(range(0, len(data), args.batch_size), desc=f"{ds_name}"):
            batch = data[i : i + args.batch_size]
            
            subjects = [s['requested_rewrite']['subject'] for s in batch]
            templates = [s['requested_rewrite']['prompt'] for s in batch]
            targets = [s['requested_rewrite']['target_new']['str'] for s in batch]
            
            prompts_baseline = [t.format(s) for t, s in zip(templates, subjects)]
            prompts_counterfactual = [
                f"Context: {p} {tgt}. Based on the context, {p}" 
                for p, tgt in zip(prompts_baseline, targets)
            ]
            
            for run_type, prompts in [("baseline", prompts_baseline), ("counterfactual", prompts_counterfactual)]:
                inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                for layer_idx, layer_tensor in enumerate(outputs.hidden_states):
                    batch_indices = torch.arange(len(batch), device=device)
                    vectors = layer_tensor[batch_indices, last_token_idx, :]
                    
                    if layer_idx not in buffers[run_type]: buffers[run_type][layer_idx] = []
                    buffers[run_type][layer_idx].append(vectors.float().cpu().numpy())
        
        for run_type, layers_data in buffers.items():
            save_dir = os.path.join(args.output_dir, "activations", ds_name, run_type)
            os.makedirs(save_dir, exist_ok=True)
            for l_idx, d_list in layers_data.items():
                final_arr = np.concatenate(d_list, axis=0)
                np.save(os.path.join(save_dir, f"layer_{l_idx:03d}_single.npy"), final_arr)
    
    print("Extraction finished.")

# ==========================================
# Step 2: analysis and plotting
# ==========================================

def analysis_step(args):
    print("Starting Analysis Phase (Rigorous Curve Fit)...")
    datasets = ["known", "unknown", "ambiguous"]
    run_types = ["baseline", "counterfactual"]
    
    sns.set_theme(style="whitegrid")
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    
    for ds_name in datasets:
        plt.figure(figsize=(10, 6))
        has_data = False
        
        for run_type in run_types:
            base_dir = os.path.join(args.output_dir, "activations", ds_name, run_type)
            files = glob.glob(os.path.join(base_dir, "layer_*_single.npy"))
            
            if not files: continue
            has_data = True
            
            layer_indices = sorted(list(set([int(os.path.basename(f).split('_')[1]) for f in files])))
            id_curve = []
            
            for layer_idx in tqdm(layer_indices, desc=f"{ds_name}-{run_type}"):
                fpath = os.path.join(base_dir, f"layer_{layer_idx:03d}_single.npy")
                X = np.load(fpath)
                id_curve.append(compute_id_layer(X))
            
            if run_type == "baseline":
                plt.plot(layer_indices, id_curve, label="baseline", color="blue", linestyle="--", alpha=0.7)
            else:
                plt.plot(layer_indices, id_curve, label="counterfactual", color="red", linewidth=2.5)
        
        if has_data:
            plt.title(f"ID Dynamics ({args.model_alias}): {ds_name.upper()}", fontsize=16)
            plt.xlabel("Layer Index")
            plt.ylabel("Intrinsic Dimension")
            plt.legend()
            plt.ylim(bottom=0)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(args.output_dir, "plots", f"ID_{ds_name}.png"), dpi=300)
            plt.close()

    print("Analysis Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True, choices=["extract", "analyze"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_alias", type=str, default="Model") 
    parser.add_argument("--know_data_path", type=str)
    parser.add_argument("--unknown_data_path", type=str)
    parser.add_argument("--ambiguous_data_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=600)
    
    args = parser.parse_args()
    
    if args.step == "extract":
        extract_step(args)
    elif args.step == "analyze":
        analysis_step(args)

if __name__ == "__main__":
    main()