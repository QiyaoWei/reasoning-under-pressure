#!/usr/bin/env python3
"""
Convert diamonds and function correctness datasets to VERL parquet format for GRPO training.
"""

import os
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
from utils.prompts import SYSTEM_PROMPT_DIAMONDS, SYSTEM_PROMPT_FUNCTION_CORRECTNESS

def categorize_measurement_pattern(measurements):
    """Categorize measurement patterns for function correctness dataset rebalancing."""
    if not measurements:
        return "empty"
    
    true_count = sum(measurements)
    total_count = len(measurements)
    true_ratio = true_count / total_count
    
    if true_ratio == 1.0:
        return "all_true"
    elif true_ratio == 0.0:
        return "all_false"
    else: # 0 < true_ratio < 1
        return "mixed"

def rebalance_function_correctness_dataset(dataset, target_samples_per_category=1000, seed=42):
    """Rebalance function correctness dataset to have uniform distribution of measurement patterns."""
    
    # Categorize all samples
    categories = defaultdict(list)
    for i, sample in enumerate(dataset):
        pattern = categorize_measurement_pattern(sample['measurements'])
        categories[pattern].append(i)
    
    # Log original distribution
    print("Original measurement pattern distribution:")
    for pattern, indices in categories.items():
        print(f"  {pattern}: {len(indices)} samples")
    
    # Determine target size per category
    min_samples = min(len(indices) for indices in categories.values())
    target_per_category = min(target_samples_per_category, min_samples)
    
    print(f"Rebalancing to {target_per_category} samples per category")
    
    # Sample uniformly from each category
    balanced_indices = []
    for pattern, indices in categories.items():
        np.random.seed(seed)
        selected_indices = np.random.choice(indices, target_per_category, replace=False)
        balanced_indices.extend(selected_indices)
    
    # Create balanced dataset
    balanced_dataset = dataset.select(balanced_indices)
    
    # Log final distribution
    final_categories = defaultdict(list)
    for i, sample in enumerate(balanced_dataset):
        pattern = categorize_measurement_pattern(sample['measurements'])
        final_categories[pattern].append(i)
    print("Balanced measurement pattern distribution:")
    for pattern, indices in final_categories.items():
        print(f"  {pattern}: {len(indices)} samples")
    
    return balanced_dataset

def convert_to_verl_format(dataset, dataset_split="train", system_prompt=SYSTEM_PROMPT_DIAMONDS, dataset_name="diamonds-seed0"):
    """Convert diamonds dataset to VERL format."""
    
    data_rows = []
    
    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {dataset_split}")):
        
        # Build the prompt in chat format
        # System prompt
        prompt = [{'role': 'system', 'content': system_prompt}]

        # User prompt
        if dataset_name.startswith("diamonds"):
            user_content = f"Now predict the measurements of the sensors for the following python snippet. Remember to think step by step through the code execution, then output your reasoning in <reasoning>...</reasoning> tags and the predicted measurements enclosed in <measurements>...</measurements> tags.\n{example['text']}\n"
        else: # function correctness
            user_content = f"Now predict the results of the test cases for the following Python function. Remember to think step by step through the code execution, then output your reasoning in <reasoning>...</reasoning> tags and the predicted results enclosed in <measurements>...</measurements> tags.\n{example['text']}\n"
        prompt.append({'role': 'user', 'content': user_content})
        
        # Convert measurements to string format for ground truth
        measurements = example.get('measurements', [])
        # Convert [True, False, True] to "[true, false, true]"
        measurements_str = str(measurements).lower().replace("'", "")
        
        # Create row in VERL format
        row = {
            'data_source': dataset_name,  # Custom data source identifier
            'prompt': prompt,
            'ability': 'reasoning',  # Task type
            'reward_model': {
                'ground_truth': measurements_str,  # Store as string
                'style': 'rule'  # Rule-based reward
            },
            'extra_info': {
                'index': idx,
                'is_correct': example.get('is_correct', False),  # latent variable
                'is_clean': example.get('is_clean', False),
                'difficulty': example.get('difficulty', -1), # default -1 for function correctness
                'measurements_list': measurements,  # Keep original list for reference
                'original_text': example['text']
            }
        }
        
        data_rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_rows)
    
    return df


def main():
    """Main conversion function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert dataset to VERL parquet format for GRPO training")
    parser.add_argument("--dataset-name", type=str, default="diamonds-seed0", 
                       choices=[f"diamonds-seed{i}" for i in range(8)] + ["function_correctness"],
                       help="Dataset to convert")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1], 
                       help="Split ratio for train/val/test")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for shuffling")
    parser.add_argument("--limit", type=lambda x: None if x.lower() == 'none' else int(x), default=None,
                        help="Limit total dataset size before splitting (default: None for all).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for parquet files (default: ~/data/{dataset-name})")
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = sum(args.split_ratio)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")
    
    # Create output directory
    if args.output_dir is None:
        # Use relative path for Docker compatibility - saves to current working directory
        output_dir = os.path.join("data", args.dataset_name)
    else:
        output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {args.dataset_name} dataset...")
    
    # Load the full train split
    dataset = load_dataset(
        f"redwoodresearch/{args.dataset_name}",
        trust_remote_code=True,
        split="train"
    )
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Rebalance function correctness dataset
    if args.dataset_name == "function_correctness":
        print("Rebalancing function correctness dataset for uniform measurement pattern distribution...")
        dataset = rebalance_function_correctness_dataset(dataset, target_samples_per_category=int(len(dataset)/4), seed=args.seed)

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=args.seed)
    
    # Apply limit if specified
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited dataset size: {len(dataset)}")
    
    # Split into train/val/test
    total_size = len(dataset)
    train_size = int(total_size * args.split_ratio[0])
    val_size = int(total_size * args.split_ratio[1])
    test_size = total_size - train_size - val_size
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
    print(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Specify the system prompt for the dataset
    if args.dataset_name.startswith("diamonds"):
        system_prompt = SYSTEM_PROMPT_DIAMONDS
    elif args.dataset_name == "function_correctness":
        system_prompt = SYSTEM_PROMPT_FUNCTION_CORRECTNESS
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    
    # Convert to VERL format
    print("\nConverting to VERL format...")
    train_df = convert_to_verl_format(train_dataset, "train", system_prompt, args.dataset_name)
    val_df = convert_to_verl_format(val_dataset, "validation", system_prompt, args.dataset_name)
    test_df = convert_to_verl_format(test_dataset, "test", system_prompt, args.dataset_name)
    
    # Save to parquet files
    print("\nSaving to parquet files...")
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path)
    print(f"Saved {len(train_df)} training samples to {train_path}")
    
    val_df.to_parquet(val_path)
    print(f"Saved {len(val_df)} validation samples to {val_path}")
    
    test_df.to_parquet(test_path)
    print(f"Saved {len(test_df)} test samples to {test_path}")
    
    print("\nDataset conversion complete!")
    
    # Verify files
    print("\nVerifying saved files...")
    for path in [train_path, val_path, test_path]:
        df_check = pd.read_parquet(path)
        print(f"  {os.path.basename(path)}: {len(df_check)} rows, columns: {df_check.columns.tolist()}")


if __name__ == "__main__":
    main()