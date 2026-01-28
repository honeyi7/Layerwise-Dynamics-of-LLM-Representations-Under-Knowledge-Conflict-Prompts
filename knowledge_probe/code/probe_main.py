#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for probing model knowledge boundaries (vLLM version - Fast Inference)

Advantages:
1. Native Tensor Parallel: Automatically splits model weights evenly across 8 GPUs
2. Memory Management: PagedAttention significantly reduces OOM risks.
3. Speed: Throughput is greatly improved.
"""


import os
import re
import json
import argparse
import collections
import time
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def normalize_answer(s: str) -> str:
    """normalize answer"""
    s = str(s).lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s

def compute_f1(prediction: str, ground_truth: str) -> float:
    """compute F1"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def format_prompt_with_subject(prompt: str, subject: str) -> str:
    if "{}" in prompt:
        return prompt.replace("{}", subject)
    elif "{" in prompt:
        try:
            return prompt.format(subject=subject)
        except (KeyError, ValueError):
            return prompt
    else:
        return prompt

def main():
    parser = argparse.ArgumentParser(description="vLLM multi-GPU detection for probing the knowledge boundaries of large models")
    parser.add_argument("--model_path", type=str, required=True, default="")
    parser.add_argument("--input_file", type=str, required=True, default="")
    parser.add_argument("--output_dir", type=str, required=True, default="")
    parser.add_argument("--f1_threshold", type=float, default=0.5)
    parser.add_argument("--save_interval", type=int, default=2000, help="After processing how many entries should the results be saved?")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="How many GPUs should be used for tensor parallelism?")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Prepare System Prompt and Tokenizer
    # Although vLLM can handle raw text, we still need the tokenizer to apply the chat template
    print(f"loading Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    system_prompt = (
        "You are a helpful, concise and efficient AI assistant. "
        "When answering any question, provide only the core, final answer with no more than five words, "
        "omitting your thought process, reasoning steps, or unnecessary background information."
    )

    # 2. Initialize the vLLM engine
    # core: tensor_parallel_size=8 will automatically split the model across 8 GPUs
    print(f"Initializing vLLM engine (TP={args.tensor_parallel_size})...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.95, # Allow vLLM to use 95% of the GPU memory
        dtype="bfloat16"
    )

    # 3. Greedy Search
    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None
    )

    # 4. data preparation
    all_items = []
    pending_prompts = [] # Store the complete prompt string after the template has been applied
    prompt_indices = []  # Record which item the prompt belongs to

    
    with open(args.input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            try:
                item = json.loads(line)
                subject = item["requested_rewrite"]["subject"]
                
                # Build all prompt variations for this entry, loading only the core keywords and paraphrased rewrites
                raw_prompts = [item["requested_rewrite"]["prompt"].format(subject)]
                # You can also include generation_prompts, but be aware that matching ground_truth will be challenging. It is recommended to only use paraphrase.
                for key in ["paraphrase_prompts"]: 
                    for p in item.get(key, []):
                        raw_prompts.append(format_prompt_with_subject(p, subject))
                
                # apply Chat Template
                for raw_p in raw_prompts:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": raw_p}
                    ]
                    # tokenize=False makes it return the processed string instead of the id
                    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    pending_prompts.append(full_prompt)
                    prompt_indices.append(i)
                
                all_items.append(item)
                
            except Exception as e:
                continue

    print(f"Data preparation completed: {len(all_items)} entries, with a total of {len(pending_prompts)} prompts")

    # 5. Execute inference (process in chunks for intermediate saving)
    print("start vLLM inference...")
    start_time = time.time()
    
    # Initialize result container
    results_by_item = [[] for _ in range(len(all_items))]
    
    # If the data volume is huge, for safety, we process it in chunks before sending it to vLLM
    # (Although vLLM can handle everything at once, we do this to print progress and save results)
    chunk_size = args.save_interval * 5 
    total_prompts = len(pending_prompts)
    
    combined_data = list(zip(pending_prompts, prompt_indices))
    
    known_count = 0
    unknown_count = 0
    ambiguous_count = 0
    
    files = {
        "known": open(os.path.join(args.output_dir, "known_bucket.jsonl"), 'w', encoding='utf-8'),
        "unknown": open(os.path.join(args.output_dir, "unknown_bucket.jsonl"), 'w', encoding='utf-8'),
        "ambiguous": open(os.path.join(args.output_dir, "ambiguous_bucket.jsonl"), 'w', encoding='utf-8'),
        "details_known": open(os.path.join(args.output_dir, "known_bucket_details.jsonl"), 'w', encoding='utf-8'),
        "details_unknown": open(os.path.join(args.output_dir, "unknown_bucket_details.jsonl"), 'w', encoding='utf-8'),
        "details_ambiguous": open(os.path.join(args.output_dir, "ambiguous_bucket_details.jsonl"), 'w', encoding='utf-8')
    }

    # Record the item indices that have been processed and written to the file
    processed_item_indices = set()
    
    for i in range(0, total_prompts, chunk_size):
        chunk = combined_data[i : min(i + chunk_size, total_prompts)]
        chunk_prompts = [x[0] for x in chunk]
        chunk_indices = [x[1] for x in chunk]
        
        # vLLM core call: Perform inference for this batch with a single line of code
        outputs = llm.generate(chunk_prompts, sampling_params)
        
        # collect results
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            item_idx = chunk_indices[j]
            
            # compute F1
            ground_truth = all_items[item_idx]["requested_rewrite"]["target_true"]["str"]
            f1 = compute_f1(generated_text, ground_truth)
            
            results_by_item[item_idx].append({
                "prompt": chunk_prompts[j], # Note that this is the complete prompt with the template. If you need the pure prompt, you may need to store it earlier.
                "generated_answer": generated_text,
                "f1_score": f1
            })

    # 6. Write to file (Iterate through all items and check if all prompts are completed)
    # Note: Since we ran the prompts in a shuffled order, we need to ensure that all probes for an item have returned before writing it
    # Simplified logic: After the above loop finishes, all results_by_item should be filled

    print("Inference completed, writing results...")
    for i, item in enumerate(tqdm(all_items, desc="Saving")):
        probe_details = results_by_item[i]
        if not probe_details:
            continue
            
        correctness_flags = [res["f1_score"] >= args.f1_threshold for res in probe_details]
        
        detailed_log = {
            "case_id": item.get("case_id", f"item_{i}"),
            "subject": item["requested_rewrite"]["subject"],
            "ground_truth": item["requested_rewrite"]["target_true"]["str"],
            "probe_details": probe_details
        }
        
        if all(correctness_flags):
            cat = "known"
            known_count += 1
        elif not any(correctness_flags):
            cat = "unknown"
            unknown_count += 1
        else:
            cat = "ambiguous"
            ambiguous_count += 1
            
        files[cat].write(json.dumps(item, ensure_ascii=False) + '\n')
        files[f"details_{cat}"].write(json.dumps(detailed_log, ensure_ascii=False) + '\n')

    for f in files.values():
        f.close()

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"Total time spent: {elapsed:.2f} 秒")
    print(f"Average throughput: {total_prompts / elapsed:.2f} prompts/sec")
    print(f"Known: {known_count}, Unknown: {unknown_count}, Ambiguous: {ambiguous_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()