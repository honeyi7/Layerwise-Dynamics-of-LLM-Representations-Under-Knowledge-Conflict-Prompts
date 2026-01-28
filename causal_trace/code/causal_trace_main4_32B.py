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
import functools
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import nethook


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
# Hook 
# ==========================================

def capture_hook_func(output, layer_name, captured_dict, indices, batch_idxs):
    if isinstance(output, tuple): h = output[0]
    else: h = output
    captured_dict[layer_name] = h[batch_idxs, indices, :].detach()
    return output

def edit_hook_func(output, layer_name, captured_dict, indices, batch_idxs):
    if isinstance(output, tuple): h = output[0]
    else: h = output
    if layer_name not in captured_dict: return output

    # Tensor already on GPU, directly assign
    src_h = captured_dict[layer_name].type(h.dtype)
    h[batch_idxs, indices, :] = src_h
    
    if isinstance(output, tuple): return (h,) + output[1:]
    else: return h


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
    
    baseline_inputs = tokenizer(baseline_prompts, padding=True, return_tensors="pt").to(device)
    cf_inputs = tokenizer(counterfactual_prompts, padding=True, return_tensors="pt").to(device)
    
    # 3. Indices
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
    
    with torch.no_grad():
        # --- 1. baseline Run ---
        baseline_outputs = model(**baseline_inputs)
        last_token_logits = baseline_outputs.logits[batch_indices, intervention_indices, :]
        probs_true = last_token_logits[batch_indices, true_target_ids]
        probs_cf = last_token_logits[batch_indices, cf_target_ids]
        Y_baseline_gpu = (probs_true - probs_cf)

        # --- 2. Counterfactual Run (Capture) ---
        captured_states = {}
        capture_cb = functools.partial(capture_hook_func, captured_dict=captured_states, indices=cf_extract_indices, batch_idxs=batch_indices)

        with nethook.TraceDict(actual_model, layers=layer_names, edit_output=lambda o, l: capture_cb(o, l)):
             cf_outputs_full = model(**cf_inputs)

        cf_logits_final = cf_outputs_full.logits[batch_indices, cf_extract_indices, :]
        y_cf_true = cf_logits_final[batch_indices, true_target_ids]
        y_cf_new = cf_logits_final[batch_indices, cf_target_ids]
        Y_counterfactual_gpu = (y_cf_true - y_cf_new)

        # --- 3. Intervention Loop ---
        total_layers = len(layer_names)
        indirect_effects_gpu = torch.zeros((batch_size, total_layers), device=device, dtype=torch.float32)
        
        for layer_idx, layer_name in enumerate(layer_names):
            edit_cb = functools.partial(edit_hook_func, layer_name=layer_name, captured_dict=captured_states, indices=intervention_indices, batch_idxs=batch_indices)
            
            with nethook.Trace(actual_model, layer=layer_name, edit_output=lambda o: edit_cb(o)):
                int_outputs = model(**baseline_inputs)
            
            int_logits = int_outputs.logits[batch_indices, intervention_indices, :]
            y_int_true = int_logits[batch_indices, true_target_ids]
            y_int_new = int_logits[batch_indices, cf_target_ids]
            Y_intervention_gpu = (y_int_true - y_int_new)
            
            indirect_effects_gpu[:, layer_idx] = Y_intervention_gpu - Y_baseline_gpu
            
    results = []
    ie_cpu = indirect_effects_gpu.float().cpu().numpy()
    yc_cpu = Y_baseline_gpu.float().cpu().numpy()
    ycf_cpu = Y_counterfactual_gpu.float().cpu().numpy()

    for b in range(batch_size):
        total_effect = ycf_cpu[b] - yc_cpu[b]
        results.append({
            "total_effect": float(total_effect),
            "indirect_effects": ie_cpu[b].tolist(),
            "Y_baseline": float(yc_cpu[b]),
            "Y_counterfactual": float(ycf_cpu[b])
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

def save_checkpoint(results, output_dir, worker_id):
    path = os.path.join(output_dir, f"checkpoint_worker_{worker_id}.pkl")
    try:
        with open(path, 'wb') as f: pickle.dump(results, f)
    except Exception as e:
        print(f"Save failed: {e}")


def merge_and_plot(output_dir, intervention_kind):
    logger = logging.getLogger(__name__)
    logger.info("Starting Merge and Plot phase...")
    
    pattern = os.path.join(output_dir, "checkpoint_worker_*.pkl")
    files = glob.glob(pattern)
    if not files:
        logger.error("No checkpoint files found!")
        return

    merged_results = {}
    for fpath in files:
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
    with open(final_pkl_path, 'wb') as f: pickle.dump(merged_results, f)
    
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))
    fig.suptitle(f"Causal Tracing - {intervention_kind.upper()}", fontsize=20, fontweight='bold', y=0.95)
    
    has_results = False
    stats_results = {}
    num_layers = 0

    for name, data in merged_results.items():
        if not data["total_effects"]: continue
        has_results = True
        
        indirect_effects_np = np.array(data["indirect_effects"])
        if np.isnan(indirect_effects_np).any(): indirect_effects_np = np.nan_to_num(indirect_effects_np)
        
        num_samples = indirect_effects_np.shape[0]
        if num_samples == 0: continue
        
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
        ax1.set_title("Raw AIE", fontsize=16); ax1.legend()
        ax1.set_xlim(-0.5, num_layers - 0.5)
        ax2.set_title("Normalized AIE", fontsize=16); ax2.legend(); ax2.axhline(y=1.0, color='r', linestyle=':')
        ax2.set_xlim(-0.5, num_layers - 0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"causal_tracing_{intervention_kind}.png"), dpi=300)
        with open(os.path.join(output_dir, f"stats_{intervention_kind}.json"), 'w') as f: json.dump(stats_results, f, indent=2)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="run", choices=["run", "merge_and_plot"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--know_data_path", type=str)
    parser.add_argument("--unknown_data_path", type=str)
    parser.add_argument("--ambiguous_data_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--intervention_kind", type=str, default="attn")
    parser.add_argument("--batch_size", type=int, default=10) 
    parser.add_argument("--max_samples_per_dataset", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    
    if args.step == "merge_and_plot":
        setup_logging(args.output_dir)
        merge_and_plot(args.output_dir, args.intervention_kind)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, args.worker_id)
    logger.info(f"Worker {args.worker_id}/{args.num_workers} starting. 32B Model Mode.")
    
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
        logger.error(f"Model load failed: {e}"); return

    actual_model = get_actual_model(model)
    num_layers = actual_model.config.num_hidden_layers
    
    if args.intervention_kind == 'state': layer_names = [f"model.layers.{i}" for i in range(num_layers)]
    elif args.intervention_kind == 'mlp': layer_names = [f"model.layers.{i}.mlp" for i in range(num_layers)]
    elif args.intervention_kind == 'attn': layer_names = [f"model.layers.{i}.self_attn" for i in range(num_layers)]
    
    datasets_config = {"known": args.know_data_path, "unknown": args.unknown_data_path, "ambiguous": args.ambiguous_data_path}
    results = {k: {"total_effects": [], "indirect_effects": []} for k in datasets_config.keys()}
    
    for dataset_name, dpath in datasets_config.items():
        full_data = load_dataset(dpath, args.max_samples_per_dataset)
        if not full_data: continue
        
        # Sharding logic
        total_len = len(full_data)
        chunk_size = int(np.ceil(total_len / args.num_workers))
        start_idx = args.worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_len)
        my_data = full_data[start_idx:end_idx]
        
        logger.info(f"[{dataset_name}] Worker {args.worker_id} processing {len(my_data)} ({start_idx}-{end_idx})")
        if not my_data: continue

        for i in tqdm(range(0, len(my_data), args.batch_size), desc=f"Worker {args.worker_id}"):
            batch_samples = my_data[i:min(i + args.batch_size, len(my_data))]
            try:
                batch_results = perform_true_batch_causal_tracing(model, tokenizer, batch_samples, device, layer_names, args.intervention_kind)
                for res in batch_results:
                    results[dataset_name]["total_effects"].append(res["total_effect"])
                    results[dataset_name]["indirect_effects"].append(res["indirect_effects"])
            except Exception as e:
                logger.error(f"Error: {e}"); traceback.print_exc()
            if i > 0 and i % args.save_interval == 0: save_checkpoint(results, args.output_dir, args.worker_id)

    save_checkpoint(results, args.output_dir, args.worker_id)
    logger.info(f"Worker {args.worker_id} Finished.")

if __name__ == "__main__":
    main()