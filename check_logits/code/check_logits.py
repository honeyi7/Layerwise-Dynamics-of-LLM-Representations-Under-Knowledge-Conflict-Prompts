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

from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'baseline_check.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def load_dataset(path, max_samples=None):
    dataset = []
    if not os.path.exists(path): 
        return dataset
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and max_samples > 0 and i >= max_samples: break
            try: dataset.append(json.loads(line))
            except: continue
    return dataset

def get_batch_logits_diff(model, tokenizer, batch, device):
    """
    Logit Difference: Logit(True Target) - Logit(New Target)
    """
    subjects = [s['requested_rewrite']['subject'] for s in batch]
    prompt_templates = [s['requested_rewrite']['prompt'] for s in batch]
    prompts = [t.format(s) for t, s in zip(prompt_templates, subjects)]
    
    true_targets = [s['requested_rewrite']['target_true']['str'] for s in batch]
    new_targets = [s['requested_rewrite']['target_new']['str'] for s in batch]

    # 2. Tokenize
    # Set padding to right to easily retrieve the last token
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    
    # Get the position of the last token in the prompt
    last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
    
    # 3. prepare Target IDs
    true_ids = []
    new_ids = []
    for i in range(len(batch)):
        # Note: Add a space prefix, this is usually necessary
        t_enc = tokenizer.encode(" " + true_targets[i], add_special_tokens=False)
        n_enc = tokenizer.encode(" " + new_targets[i], add_special_tokens=False)
        true_ids.append(t_enc[0] if t_enc else 0)
        new_ids.append(n_enc[0] if n_enc else 0)
        
    batch_indices = torch.arange(len(batch), device=device)
    true_ids = torch.tensor(true_ids, device=device)
    new_ids = torch.tensor(new_ids, device=device)
    
    # 4. Forward Pass
    with torch.no_grad():
        outputs = model(**inputs)
        # [Batch, Seq, Vocab] -> [Batch, Vocab]
        final_logits = outputs.logits[batch_indices, last_token_indices, :]
        
        logits_true = final_logits[batch_indices, true_ids]
        logits_new = final_logits[batch_indices, new_ids]
        
        # compute differences
        diffs = (logits_true - logits_new).float().cpu().numpy()
        
        # Additionally calculate the Softmax probabilities of the True Target as auxiliary reference
        probs = torch.softmax(final_logits, dim=-1)
        probs_true = probs[batch_indices, true_ids].float().cpu().numpy()
        
    return diffs, probs_true

def plot_distributions(results, output_dir):

    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 5.5))
    
    legend_args = dict(
        loc='upper right',      
        fontsize='x-small',      
        framealpha=1,            
        borderaxespad=0,        
        borderpad=0.4,         
        handlelength=1.0,        
        handletextpad=0.4,       
        labelspacing=0.3         
    )

    # === 1. Logit Difference distribution ===
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        if len(data['diffs']) > 0:
            sns.kdeplot(
                data['diffs'], 
                label=f"{name} (Mean: {np.mean(data['diffs']):.2f})", 
                fill=True, 
                alpha=0.3
            )
    
    plt.title("Logit Difference Distribution")
    plt.xlabel("Logit Difference")
    plt.ylabel("Density")
    
    plt.legend(**legend_args)
    
    # === 2. Probability distribution ===
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        if len(data['probs']) > 0:
            sns.kdeplot(
                data['probs'], 
                label=f"{name} (Mean: {np.mean(data['probs']):.2f})", 
                fill=True, 
                alpha=0.3
            )
            
    plt.title("True Target Probability Distribution")
    plt.xlabel("Probability P(True)")
    plt.ylabel("Density")
    plt.xlim(-0.1, 1.1)
    
    plt.legend(**legend_args)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "baseline_logits_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--know_data_path", type=str, required=True)
    parser.add_argument("--unknown_data_path", type=str, required=True)
    parser.add_argument("--ambiguous_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--intervention_kind", type=str, default="attn") 
    parser.add_argument("--save_interval", type=int, default=100)
    
    parser.add_argument("--max_samples_per_dataset", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=50)
    
    args = parser.parse_args()
    
    logger = setup_logging(args.output_dir)
    logger.info(f"Checking Baseline Logits for {args.model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # 0.5B is relatively small, usually doesn't require multi_gpu device_map, just use to(device)
        # For larger models, this will be handled automaticall
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return

    datasets = {
        "known": load_dataset(args.know_data_path, args.max_samples_per_dataset),
        "unknown": load_dataset(args.unknown_data_path, args.max_samples_per_dataset),
        "ambiguous": load_dataset(args.ambiguous_data_path, args.max_samples_per_dataset)
    }
    
    final_results = {}

    for name, data in datasets.items():
        if not data: continue
        logger.info(f"Processing {name} dataset ({len(data)} samples)...")
        
        all_diffs = []
        all_probs = []
        
        for i in tqdm(range(0, len(data), args.batch_size)):
            batch = data[i : i + args.batch_size]
            diffs, probs = get_batch_logits_diff(model, tokenizer, batch, device)
            all_diffs.extend(diffs)
            all_probs.extend(probs)
            
        final_results[name] = {
            "diffs": np.array(all_diffs),
            "probs": np.array(all_probs)
        }
        
        mean_diff = np.mean(all_diffs)
        std_diff = np.std(all_diffs)
        mean_prob = np.mean(all_probs)
        
        logger.info(f"Results for {name.upper()}:")
        logger.info(f"  Mean Logit Diff (True - New): {mean_diff:.4f} ± {std_diff:.4f}")
        logger.info(f"  Mean Probability (True):      {mean_prob:.4f}")

    plot_distributions(final_results, args.output_dir)
    
    with open(os.path.join(args.output_dir, "baseline_stats.pkl"), 'wb') as f:
        pickle.dump(final_results, f)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()