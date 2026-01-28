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
from transformers import AutoModelForCausalLM, AutoTokenizer

import nethook

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, f'causal_trace.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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

def capture_hook_func(output, layer_name, captured_dict, indices, batch_idxs, *args):
    """Capture the Hidden State during the Counterfactual (context-enhanced) runtime"""
    if isinstance(output, tuple): h = output[0]
    else: h = output
    
    # Capture the specified location
    captured_dict[layer_name] = h[batch_idxs, indices, :].detach().clone()
    return output

def edit_hook_func(output, layer_name, captured_dict, indices, batch_idxs, logger, *args):
    """Inject the state with contextual information into the baseline run"""
    if isinstance(output, tuple): h = output[0]
    else: h = output
    
    if layer_name not in captured_dict: return output

    src_h = captured_dict[layer_name].to(h.device).type(h.dtype)
    
    h[batch_idxs, indices, :] = src_h
    
    if isinstance(output, tuple): return (h,) + output[1:]
    else: return h

def perform_true_batch_causal_tracing(model, tokenizer, batch_samples, device, layer_names, intervention_kind, logger):
    actual_model = get_actual_model(model)
    batch_size = len(batch_samples)
    
    # 1. construct Prompt
    subjects = [s['requested_rewrite']['subject'] for s in batch_samples]
    prompt_templates = [s['requested_rewrite']['prompt'] for s in batch_samples]
    
    # baseline Prompt = raw Prompt
    baseline_prompts = [t.format(s) for t, s in zip(prompt_templates, subjects)]
    
    # Counterfactual Prompt =  (ICL)
    cf_targets_str = [s['requested_rewrite']['target_new']['str'] for s in batch_samples]

    counterfactual_prompts = [f"Context: {p} {cf_tgt}. Based on the context, {p}" for p, cf_tgt in zip(baseline_prompts, cf_targets_str)]
    
    # 2. Tokenize
    tokenizer.padding_side = 'right' 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    baseline_inputs = tokenizer(baseline_prompts, padding=True, return_tensors="pt").to(device)
    cf_inputs = tokenizer(counterfactual_prompts, padding=True, return_tensors="pt").to(device)
    
    # 3. compute intervention indices (last Token)
    intervention_indices = baseline_inputs['attention_mask'].sum(dim=1) - 1
    cf_extract_indices = cf_inputs['attention_mask'].sum(dim=1) - 1

    # prepare Target IDs
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
        # =========================================
        # 1. baseline Run
        # =========================================
        baseline_outputs = model(**baseline_inputs)
        
        last_token_logits = baseline_outputs.logits[batch_indices, intervention_indices, :]
        probs_true = last_token_logits[batch_indices, true_target_ids]
        probs_cf = last_token_logits[batch_indices, cf_target_ids]
        Y_baseline = (probs_true - probs_cf).float().cpu().numpy()

        # =========================================
        # 2. Counterfactual Run
        # =========================================
        captured_states = {}
        
        capture_cb = functools.partial(
            capture_hook_func, 
            captured_dict=captured_states, 
            indices=cf_extract_indices, 
            batch_idxs=batch_indices
        )

        with nethook.TraceDict(actual_model, layers=layer_names, edit_output=lambda o, l: capture_cb(o, l)):
             cf_outputs_full = model(**cf_inputs)

        cf_logits_final = cf_outputs_full.logits[batch_indices, cf_extract_indices, :]
        y_cf_true = cf_logits_final[batch_indices, true_target_ids]
        y_cf_new = cf_logits_final[batch_indices, cf_target_ids]
        Y_counterfactual = (y_cf_true - y_cf_new).float().cpu().numpy()

        # =========================================
        # 3. Intervention Run
        # =========================================
        total_layers = len(layer_names)
        indirect_effects = np.zeros((batch_size, total_layers)) 
        
        for layer_idx, layer_name in enumerate(layer_names):
            
            edit_cb = functools.partial(
                edit_hook_func,
                layer_name=layer_name,
                captured_dict=captured_states,
                indices=intervention_indices,
                batch_idxs=batch_indices,
                logger=logger if layer_idx == 0 else None
            )
            
            with nethook.Trace(actual_model, layer=layer_name, edit_output=lambda o: edit_cb(o)):
                int_outputs = model(**baseline_inputs)
            
            int_logits = int_outputs.logits[batch_indices, intervention_indices, :]
            y_int_true = int_logits[batch_indices, true_target_ids]
            y_int_new = int_logits[batch_indices, cf_target_ids]
            Y_intervention = (y_int_true - y_int_new).float().cpu().numpy()
            
            indirect_effects[:, layer_idx] = Y_intervention - Y_baseline
            
    results = []
    for b in range(batch_size):
        total_effect = Y_counterfactual[b] - Y_baseline[b]
        ie = indirect_effects[b]
        
        results.append({
            "total_effect": float(total_effect),
            "indirect_effects": ie.tolist(),
            "Y_baseline": float(Y_baseline[b]),
            "Y_counterfactual": float(Y_counterfactual[b])
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

def save_checkpoint(results, output_dir):
    try:
        with open(os.path.join(output_dir, "checkpoint.pkl"), 'wb') as f: pickle.dump(results, f)
    except: pass

def load_checkpoint(output_dir):
    path = os.path.join(output_dir, "checkpoint.pkl")
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f: return pickle.load(f)
        except: return None
    return None

def analyze_and_plot_results(results, intervention_kind, output_dir):
    logger = logging.getLogger(__name__)
    logger.info("\n--- Generating Plots ---")
    
    sns.set_theme(style="whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))
    
    fig.suptitle(f"Causal Tracing Analysis - Intervention: {intervention_kind.upper()}", fontsize=20, fontweight='bold', y=0.95)
    
    has_results = False
    stats_results = {}
    num_layers = 0

    for name, data in results.items():
        if not data["total_effects"]: continue
        has_results = True
        
        indirect_effects_np = np.array(data["indirect_effects"])
        if np.isnan(indirect_effects_np).any(): indirect_effects_np = np.nan_to_num(indirect_effects_np)
        
        num_samples = indirect_effects_np.shape[0]
        if num_samples == 0: continue
        
        num_layers = indirect_effects_np.shape[1]
        aie_per_layer = np.mean(indirect_effects_np, axis=0)
        ate = np.mean(data["total_effects"])
        
        logger.info(f"Dataset: {name}, Samples: {num_samples}, ATE: {ate:.4f}")
        
        # Raw AIE
        ax1.plot(range(num_layers), aie_per_layer, marker='o', markersize=4, label=f"{name} (ATE={ate:.2f})")
        
        # Normalized AIE
        if abs(ate) > 1e-9:
            norm_aie = aie_per_layer / ate
            ax2.plot(range(num_layers), norm_aie, marker='s', markersize=4, label=f"{name}")
        
        stats_results[name] = {
            "num_samples": int(num_samples),
            "average_total_effect": float(ate),
            "aie_per_layer": aie_per_layer.tolist()
        }

    if has_results:
        # --- Raw AIE ---
        ax1.set_title("Raw Average Indirect Effect (AIE)", fontsize=16, fontweight='bold')
        ax1.set_ylabel("AIE (Logit Difference Change)\n(Intervention - baseline)", fontsize=12)
        ax1.set_xlabel("Layer Index", fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.legend(title="Dataset", fontsize=10, loc='best')
        ax1.set_xticks(range(0, num_layers))
        ax1.set_xlim(-0.5, num_layers - 0.5)

        # --- Normalized AIE ---
        ax2.set_title("Normalized AIE (Proportion of Total Effect Restored)", fontsize=16, fontweight='bold')
        ax2.set_ylabel("Normalized AIE\n(AIE / ATE)", fontsize=12)
        ax2.set_xlabel("Layer Index", fontsize=12)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        ax2.legend(title="Dataset", fontsize=10, loc='best')
        ax2.axhline(y=1.0, color='r', linestyle=':', alpha=0.5, label='Total Effect Baseline')
        ax2.axhline(y=0.0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xticks(range(0, num_layers))
        ax2.set_xlim(-0.5, num_layers - 0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(output_dir, f"causal_tracing_{intervention_kind}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
        
        with open(os.path.join(output_dir, f"stats_{intervention_kind}.json"), 'w') as f:
            json.dump(stats_results, f, indent=2)
            
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--know_data_path", type=str, required=True)
    parser.add_argument("--unknown_data_path", type=str, required=True)
    parser.add_argument("--ambiguous_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples_per_dataset", type=int, default=-1)
    parser.add_argument("--intervention_kind", type=str, default="attn", choices=["state", "mlp", "attn"])
    parser.add_argument("--batch_size", type=int, default=10) 
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--debug_samples", type=int, default=0)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    logger.info(f"Running ICL-BASED Causal Tracing (Intervention: {args.intervention_kind})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    model_kwargs["device_map"] = "auto" if args.use_multi_gpu else {"": 0}

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Model load failed: {e}"); return

    actual_model = get_actual_model(model)
    num_layers = actual_model.config.num_hidden_layers
    
    if args.intervention_kind == 'state': layer_names = [f"model.layers.{i}" for i in range(num_layers)]
    elif args.intervention_kind == 'mlp': layer_names = [f"model.layers.{i}.mlp" for i in range(num_layers)]
    elif args.intervention_kind == 'attn': layer_names = [f"model.layers.{i}.self_attn" for i in range(num_layers)]
    
    datasets = {
        "known": load_dataset(args.know_data_path),
        "unknown": load_dataset(args.unknown_data_path),
        "ambiguous": load_dataset(args.ambiguous_data_path)
    }
    
    results = {k: {"total_effects": [], "indirect_effects": []} for k in datasets.keys()}
    
    for dataset_name, data in datasets.items():
        if not data: continue
        logger.info(f"Processing {dataset_name}")
        limit = len(data) if args.debug_samples == 0 else args.debug_samples
        
        for i in tqdm(range(0, limit, args.batch_size), desc=dataset_name):
            batch_samples = data[i:min(i + args.batch_size, limit)]
            try:
                batch_results = perform_true_batch_causal_tracing(
                    model, tokenizer, batch_samples, device, layer_names, args.intervention_kind, logger
                )
                for res in batch_results:
                    results[dataset_name]["total_effects"].append(res["total_effect"])
                    results[dataset_name]["indirect_effects"].append(res["indirect_effects"])
            except Exception as e:
                logger.error(f"Batch Error: {e}")
                traceback.print_exc()
                
            if i > 0 and i % args.save_interval == 0: save_checkpoint(results, args.output_dir)

    save_checkpoint(results, args.output_dir)
    analyze_and_plot_results(results, args.intervention_kind, args.output_dir)
    logger.info("Done.")

if __name__ == "__main__":
    main()