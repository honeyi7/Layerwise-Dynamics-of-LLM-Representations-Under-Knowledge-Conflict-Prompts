#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
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
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors

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
    if N < 100: return 0.0 
    
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='euclidean').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    mask = r1 > 1e-10
    if np.sum(mask) < 5: return 0.0
    
    mus = r2[mask] / r1[mask]
    
    return _compute_id_2NN_exact(mus, fraction=0.95)

# ==========================================
# data load and processing
# ==========================================

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
# step1：Extract activations
# ==========================================

def extract_step(args):
    print(f"[Worker {args.worker_id}] Loading Model: {args.model_path}")
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
        if not ds_path: continue
        
        full_data = load_dataset(ds_path, args.max_samples_per_dataset)
        total_len = len(full_data)
        chunk_size = int(np.ceil(total_len / args.num_workers))
        start_idx = args.worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_len)
        my_data = full_data[start_idx:end_idx]
        
        if not my_data: continue
        print(f"[Worker {args.worker_id}] Processing {ds_name}: {len(my_data)} samples")

        buffers = {"baseline": {}, "counterfactual": {}}
        
        for i in tqdm(range(0, len(my_data), args.batch_size), desc=f"{ds_name}"):
            batch = my_data[i : i + args.batch_size]
            
            subjects = [s['requested_rewrite']['subject'] for s in batch]
            prompt_templates = [s['requested_rewrite']['prompt'] for s in batch]
            targets = [s['requested_rewrite']['target_new']['str'] for s in batch]
            
            prompts_baseline = [t.format(s) for t, s in zip(prompt_templates, subjects)]
            prompts_counterfactual = [
                f"Context: {p} {tgt}. Based on the context, {p}" 
                for p, tgt in zip(prompts_baseline, targets)
            ]
            
            for run_type, prompts in [("baseline", prompts_baseline), ("counterfactual", prompts_counterfactual)]:
                inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                for layer_idx, layer_tensor in enumerate(outputs.hidden_states):
                    batch_indices = torch.arange(len(batch), device=device)
                    vectors = layer_tensor[batch_indices, last_token_indices, :]
                    
                    if layer_idx not in buffers[run_type]:
                        buffers[run_type][layer_idx] = []
                    buffers[run_type][layer_idx].append(vectors.float().cpu().numpy())
        
        for run_type, layers_data in buffers.items():
            save_dir = os.path.join(args.output_dir, "activations", ds_name, run_type)
            os.makedirs(save_dir, exist_ok=True)
            for layer_idx, data_list in layers_data.items():
                if not data_list: continue
                final_arr = np.concatenate(data_list, axis=0)
                fname = f"layer_{layer_idx:03d}_worker_{args.worker_id}.npy"
                np.save(os.path.join(save_dir, fname), final_arr)
        
        del buffers
        torch.cuda.empty_cache()

# ==========================================
# step2：analysis and plotting
# ==========================================

def analysis_step(args):
    print("Starting Analysis Phase (Using Exact TwoNN Logic)...")
    datasets = ["known", "unknown", "ambiguous"]
    run_types = ["baseline", "counterfactual"]
    
    sns.set_theme(style="whitegrid")
    final_results = {}
    
    for ds_name in datasets:
        final_results[ds_name] = {}
        for run_type in run_types:
            base_dir = os.path.join(args.output_dir, "activations", ds_name, run_type)
            if not os.path.exists(base_dir): continue
            
            files = glob.glob(os.path.join(base_dir, "layer_*_worker_*.npy"))
            if not files: continue
            
            layer_indices = sorted(list(set([int(os.path.basename(f).split('_')[1]) for f in files])))
            id_curve = []
            
            for layer_idx in tqdm(layer_indices, desc=f"Calc ID {ds_name}-{run_type}"):
                layer_files = glob.glob(os.path.join(base_dir, f"layer_{layer_idx:03d}_worker_*.npy"))
                arrays = [np.load(f) for f in layer_files]
                if not arrays: 
                    id_curve.append(np.nan)
                    continue
                
                X = np.concatenate(arrays, axis=0)
                try:
                    val_id = compute_id_layer(X)
                except Exception as e:
                    print(f"Error at L{layer_idx}: {e}")
                    val_id = np.nan
                id_curve.append(val_id)
                
            final_results[ds_name][run_type] = (layer_indices, id_curve)
            
    # plot
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    
    for ds_name in datasets:
        if ds_name not in final_results or not final_results[ds_name]: continue
        
        plt.figure(figsize=(10, 6)) # Keep the same sizing as Single GPU
        
        if "baseline" in final_results[ds_name]:
            layers, ids = final_results[ds_name]["baseline"]
            plt.plot(layers, ids, label="baseline", color="blue", linestyle="--", alpha=0.7)
            
        if "counterfactual" in final_results[ds_name]:
            layers, ids = final_results[ds_name]["counterfactual"]
            plt.plot(layers, ids, label="counterfactual", color="red", linewidth=2.5)

        model_name_display = "Qwen2.5-32B" # Or retrieve from args, here it's hardcoded to match the large model
        plt.title(f"ID Dynamics ({model_name_display}): {ds_name.upper()}", fontsize=16)
        
        plt.xlabel("Layer Index")
        plt.ylabel("Intrinsic Dimension")
        plt.legend()
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(args.output_dir, "plots", f"ID_{ds_name}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved: {save_path}")
        plt.close()

    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True, choices=["extract", "analyze"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--know_data_path", type=str)
    parser.add_argument("--unknown_data_path", type=str)
    parser.add_argument("--ambiguous_data_path", type=str)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_samples_per_dataset", type=int, default=600)
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.step == "extract":
        extract_step(args)
    elif args.step == "analyze":
        analysis_step(args)

if __name__ == "__main__":
    main()