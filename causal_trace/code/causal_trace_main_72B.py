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
import logging
import pickle
import traceback
import gc
import functools
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer

import nethook

# ==========================================
# Configuration and tools
# ==========================================

def setup_logging(output_dir, worker_id=None):
    suffix = f"_worker_{worker_id}" if worker_id is not None else "_master"
    log_file = os.path.join(output_dir, f'causal_trace{suffix}.log')
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - [Worker {worker_id if worker_id is not None else "M"}] - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def get_actual_model(model):
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        return model.module
    return model

# ==========================================
# Hook function
# ==========================================

def capture_hook_func(output, layer_name, captured_dict, indices, batch_idxs):
    """
    Capture the Hidden State of a specific token.
    Dynamically move indices based on the actual device of the hidden state (h).
    Move the captured data to CPU to save GPU memory.
    """
    if isinstance(output, tuple): h = output[0]
    else: h = output
    
    # Get the device of the current layer output (could be cuda:0, cuda:1, etc.)
    current_device = h.device
    
    # Move the indices to the same device
    local_indices = indices.to(current_device)
    local_batch_idxs = batch_idxs.to(current_device)
    
    # The 72B model has many layers, it is recommended to detach and move the data to CPU to avoid OOM
    captured_dict[layer_name] = h[local_batch_idxs, local_indices, :].detach().cpu()
    
    return output

def edit_hook_func(output, layer_name, captured_dict, indices, batch_idxs):
    """
    Intervene on the Hidden State of a specific token.
    Move the patch data stored on the CPU back to the GPU of the current layer.
    """

    if isinstance(output, tuple): h = output[0]
    else: h = output
    
    if layer_name not in captured_dict: return output

    current_device = h.device
    local_indices = indices.to(current_device)
    local_batch_idxs = batch_idxs.to(current_device)
    
    # Retrieve from the dictionary (possibly on CPU), move to the current GPU, and convert dtype (BF16/FP16)
    src_h = captured_dict[layer_name].to(current_device).type(h.dtype)
    
    # Perform replacement
    h[local_batch_idxs, local_indices, :] = src_h
    
    if isinstance(output, tuple): return (h,) + output[1:]
    else: return h

# ==========================================
# core causal tracing function
# ==========================================

def perform_true_batch_causal_tracing(model, tokenizer, batch_samples, device, layer_names, intervention_kind):
    actual_model = get_actual_model(model)
    batch_size = len(batch_samples)
    
    # 1. construct Prompt
    subjects = [s['requested_rewrite']['subject'] for s in batch_samples]
    prompt_templates = [s['requested_rewrite']['prompt'] for s in batch_samples]
    
    baseline_prompts = [t.format(s) for t, s in zip(prompt_templates, subjects)]
    
    cf_targets_str = [s['requested_rewrite']['target_new']['str'] for s in batch_samples]
    counterfactual_prompts = [f"Context: {p} {cf_tgt}. Based on the context, {p}" for p, cf_tgt in zip(baseline_prompts, cf_targets_str)]
    
    # 2. Tokenize
    tokenizer.padding_side = 'right' 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # put inputs in main device (usually cuda:0)
    baseline_inputs = tokenizer(baseline_prompts, padding=True, return_tensors="pt").to(device)
    cf_inputs = tokenizer(counterfactual_prompts, padding=True, return_tensors="pt").to(device)
    
    # 3. Indices & Targets (original in device)
    intervention_indices = baseline_inputs['attention_mask'].sum(dim=1) - 1
    cf_extract_indices = cf_inputs['attention_mask'].sum(dim=1) - 1

    true_targets = [s['requested_rewrite']['target_true']['str'] for s in batch_samples]
    true_target_ids = []
    cf_target_ids = []
    
    for i in range(batch_size):
        tt_ids = tokenizer.encode(" " + true_targets[i], add_special_tokens=False)
        ct_ids = tokenizer.encode(" " + cf_targets_str[i], add_special_tokens=False)
        true_target_ids.append(tt_ids[0] if tt_ids else 0)
        cf_target_ids.append(ct_ids[0] if ct_ids else 0)
        
    batch_indices = torch.arange(batch_size, device=device)
    true_target_ids = torch.tensor(true_target_ids, device=device)
    cf_target_ids = torch.tensor(cf_target_ids, device=device)

    model.eval()
    
    # Helper function: Automatically handle device alignment when calculating logits
    def get_probs_diff(logits, b_idxs, i_idxs, t_ids, c_ids):
        # logits may be on cuda:1, while indices are on cuda:0
        target_device = logits.device
        
        # Move all indices to the device where logits are located
        _b = b_idxs.to(target_device)
        _i = i_idxs.to(target_device)
        _t = t_ids.to(target_device)
        _c = c_ids.to(target_device)
        
        last_token_logits = logits[_b, _i, :]
        probs_true = last_token_logits[_b, _t]
        probs_cf = last_token_logits[_b, _c]
        
        return (probs_true - probs_cf)

    with torch.no_grad():
        # --- 1. baseline Run ---
        baseline_outputs = model(**baseline_inputs)
        # Calculate the logit difference under the baseline state (result is kept on the device where logits are located)
        Y_baseline_gpu = get_probs_diff(
            baseline_outputs.logits, batch_indices, intervention_indices, true_target_ids, cf_target_ids
        )

        # --- 2. Counterfactual Run (Capture) ---
        captured_states = {}
        # Hook callback
        capture_cb = functools.partial(capture_hook_func, captured_dict=captured_states, indices=cf_extract_indices, batch_idxs=batch_indices)

        with nethook.TraceDict(actual_model, layers=layer_names, edit_output=lambda o, l: capture_cb(o, l)):
             cf_outputs_full = model(**cf_inputs)

        # Calculate the logit difference under the ctf state
        Y_counterfactual_gpu = get_probs_diff(
            cf_outputs_full.logits, batch_indices, cf_extract_indices, true_target_ids, cf_target_ids
        )

        # --- 3. Intervention Loop ---
        total_layers = len(layer_names)
        # Place the result matrix on the CPU to avoid GPU memory fragmentation, or on the main device
        # For safety, move to CPU immediately after calculation
        indirect_effects_list = []
        
        # Move the baseline value to CPU for subsequent calculations
        base_baseline = Y_baseline_gpu.float().cpu()
        
        for layer_idx, layer_name in enumerate(layer_names):
            edit_cb = functools.partial(edit_hook_func, layer_name=layer_name, captured_dict=captured_states, indices=intervention_indices, batch_idxs=batch_indices)
            
            with nethook.Trace(actual_model, layer=layer_name, edit_output=lambda o: edit_cb(o)):
                int_outputs = model(**baseline_inputs)
            
            Y_intervention_gpu = get_probs_diff(
                int_outputs.logits, batch_indices, intervention_indices, true_target_ids, cf_target_ids
            )
            
            # Calculate the difference and move to CPU
            # Ensure both tensors are on the same device before performing subtraction (Y_intervention_gpu and Y_baseline_gpu are usually on the last GPU)
            effect = (Y_intervention_gpu - Y_baseline_gpu.to(Y_intervention_gpu.device)).float().cpu()
            indirect_effects_list.append(effect)
            
    # Organize the results
    # indirect_effects_list is [Layer1_Tensor(Batch), Layer2_Tensor(Batch), ...]
    # It needs to be converted to (Batch, Layers)
    indirect_effects_cpu = torch.stack(indirect_effects_list, dim=1).numpy() # Shape: [B, L]
    Y_baseline_cpu = Y_baseline_gpu.float().cpu().numpy()
    Y_counterfactual_cpu = Y_counterfactual_gpu.float().cpu().numpy()

    results = []
    for b in range(batch_size):
        total_effect = Y_counterfactual_cpu[b] - Y_baseline_cpu[b]
        results.append({
            "total_effect": float(total_effect),
            "indirect_effects": indirect_effects_cpu[b].tolist(),
            "Y_baseline": float(Y_baseline_cpu[b]),
            "Y_counterfactual": float(Y_counterfactual_cpu[b])
        })
        
    return results


def load_dataset(path, max_samples=None):
    dataset = []
    if not os.path.exists(path): return dataset
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and max_samples > 0 and i >= max_samples: break
            try: dataset.append(json.loads(line))
            except: continue
    return dataset

def get_checkpoint_path(output_dir, worker_id):
    return os.path.join(output_dir, f"checkpoint_worker_{worker_id}.pkl")

def save_checkpoint(results, output_dir, worker_id):
    path = get_checkpoint_path(output_dir, worker_id)
    try:
        with open(path, 'wb') as f: pickle.dump(results, f)
    except Exception as e:
        print(f"Save failed: {e}")



def merge_and_plot(output_dir, intervention_kind):
    logger = logging.getLogger(__name__)
    logger.info("Starting Merge and Plot phase...")
    
    # Find the checkpoint of all workers
    pattern = os.path.join(output_dir, "checkpoint_worker_*.pkl")
    files = glob.glob(pattern)
    logger.info(f"Found files: {files}")
    
    if not files:
        logger.error("No checkpoint files found to merge!")
        return

    merged_results = {}
    
    for fpath in files:
        logger.info(f"Loading {fpath}...")
        try:
            with open(fpath, 'rb') as f:
                chunk_res = pickle.load(f)
                for ds_name, ds_data in chunk_res.items():
                    if ds_name not in merged_results:
                        merged_results[ds_name] = {"total_effects": [], "indirect_effects": []}
                    merged_results[ds_name]["total_effects"].extend(ds_data["total_effects"])
                    merged_results[ds_name]["indirect_effects"].extend(ds_data["indirect_effects"])
        except Exception as e:
            logger.error(f"Error loading {fpath}: {e}")

    final_pkl_path = os.path.join(output_dir, "final_results_merged.pkl")
    with open(final_pkl_path, 'wb') as f:
        pickle.dump(merged_results, f)
    logger.info(f"Merged results saved to {final_pkl_path}")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))
    fig.suptitle(f"Causal Tracing Analysis - Intervention: {intervention_kind.upper()}", fontsize=20, fontweight='bold', y=0.95)
    
    has_results = False
    stats_results = {}
    num_layers = 0

    for name, data in merged_results.items():
        if not data["total_effects"]: continue
        
        indirect_effects_np = np.array(data["indirect_effects"])
        if np.isnan(indirect_effects_np).any(): 
            indirect_effects_np = np.nan_to_num(indirect_effects_np)
        
        num_samples = indirect_effects_np.shape[0]
        if num_samples == 0: continue
        
        has_results = True
        num_layers = indirect_effects_np.shape[1]
        aie_per_layer = np.mean(indirect_effects_np, axis=0)
        ate = np.mean(data["total_effects"])
        
        ax1.plot(range(num_layers), aie_per_layer, marker='o', markersize=4, label=f"{name} (ATE={ate:.2f})")
        
        if abs(ate) > 1e-9:
            norm_aie = aie_per_layer / ate
            ax2.plot(range(num_layers), norm_aie, marker='s', markersize=4, label=f"{name}")
        
        stats_results[name] = {
            "num_samples": int(num_samples),
            "average_total_effect": float(ate),
            "aie_per_layer": aie_per_layer.tolist()
        }

    if has_results:
        ax1.set_title("Raw Average Indirect Effect (AIE)", fontsize=16, fontweight='bold')
        ax1.set_ylabel("AIE (Logit Diff)", fontsize=12)
        ax1.legend(loc='best')
        ax1.set_xlim(-0.5, num_layers - 0.5)

        ax2.set_title("Normalized AIE", fontsize=16, fontweight='bold')
        ax2.set_ylabel("Normalized AIE", fontsize=12)
        ax2.axhline(y=1.0, color='r', linestyle=':', alpha=0.5)
        ax2.legend(loc='best')
        ax2.set_xlim(-0.5, num_layers - 0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f"causal_tracing_{intervention_kind}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
        
        with open(os.path.join(output_dir, f"stats_{intervention_kind}.json"), 'w') as f:
            json.dump(stats_results, f, indent=2)
    else:
        logger.warning("No valid results to plot.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="run", choices=["run", "merge_and_plot"], help="Step to execute")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--know_data_path", type=str)
    parser.add_argument("--unknown_data_path", type=str)
    parser.add_argument("--ambiguous_data_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--intervention_kind", type=str, default="attn")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_samples_per_dataset", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=50)
    
    parser.add_argument("--worker_id", type=int, default=0, help="ID of current worker")
    parser.add_argument("--num_workers", type=int, default=1, help="Total number of workers")
    
    args = parser.parse_args()
    
    if args.step == "merge_and_plot":
        setup_logging(args.output_dir)
        merge_and_plot(args.output_dir, args.intervention_kind)
        return

    # --- Run Step ---
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, args.worker_id)
    
    logger.info(f"Worker {args.worker_id}/{args.num_workers} starting. Intervention: {args.intervention_kind}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        logger.error(traceback.format_exc())
        return

    actual_model = get_actual_model(model)
    num_layers = actual_model.config.num_hidden_layers
    
    if args.intervention_kind == 'state': layer_names = [f"model.layers.{i}" for i in range(num_layers)]
    elif args.intervention_kind == 'mlp': layer_names = [f"model.layers.{i}.mlp" for i in range(num_layers)]
    elif args.intervention_kind == 'attn': layer_names = [f"model.layers.{i}.self_attn" for i in range(num_layers)]
    else:
        logger.error(f"Unknown intervention kind: {args.intervention_kind}")
        return

    datasets_config = {
        "known": args.know_data_path,
        "unknown": args.unknown_data_path,
        "ambiguous": args.ambiguous_data_path
    }
    
    results = {k: {"total_effects": [], "indirect_effects": []} for k in datasets_config.keys()}
    
    for dataset_name, dpath in datasets_config.items():
        if not dpath or not os.path.exists(dpath):
            logger.warning(f"Dataset path invalid or not provided: {dpath}")
            continue
            
        full_data = load_dataset(dpath, args.max_samples_per_dataset)
        if not full_data: 
            logger.warning(f"Dataset {dataset_name} is empty.")
            continue
        
        # Sharding logic
        total_len = len(full_data)
        chunk_size = int(np.ceil(total_len / args.num_workers))
        start_idx = args.worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_len)
        
        my_data = full_data[start_idx:end_idx]
        logger.info(f"Dataset [{dataset_name}]: Total {total_len}, Worker {args.worker_id} processing {len(my_data)} samples ({start_idx}-{end_idx})")
        
        if not my_data: continue

        for i in tqdm(range(0, len(my_data), args.batch_size), desc=f"Worker {args.worker_id} - {dataset_name}"):
            batch_samples = my_data[i:min(i + args.batch_size, len(my_data))]
            try:
                batch_results = perform_true_batch_causal_tracing(
                    model, tokenizer, batch_samples, device, layer_names, args.intervention_kind
                )
                for res in batch_results:
                    results[dataset_name]["total_effects"].append(res["total_effect"])
                    results[dataset_name]["indirect_effects"].append(res["indirect_effects"])
            except Exception as e:
                logger.error(f"Batch Error: {e}")
                logger.error(traceback.format_exc())
                
            if i > 0 and i % args.save_interval == 0: 
                save_checkpoint(results, args.output_dir, args.worker_id)
                gc.collect()
                torch.cuda.empty_cache()

    save_checkpoint(results, args.output_dir, args.worker_id)
    logger.info(f"Worker {args.worker_id} Finished.")

if __name__ == "__main__":
    main()