#!/usr/bin/env python3
"""
Inference script for VERL-trained models on DIAMONDS and function correctness datasets.
Adapted from my_implementation.py to work with VERL checkpoint structure.
Uses transformers instead of unsloth for compatibility with VERL checkpoints.

This script supports both diamonds-seed0 to diamonds-seed7 and function_correctness datasets.
"""

import os
import re
import json
import glob
import torch
import argparse
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_VERBOSITY'] = 'error'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from utils.extract_measurements import extract_measurements

# Constants
MAX_LEN_PROMPT = 2048
# MAX_LEN_GENERATION will be set dynamically based on dataset


def load_test_dataset(dataset_name: str = "diamonds-seed0", 
                     dataset_set: str = "test",
                     test_size: int = None,
                     seed: int = 42,
                     data_dir: str = None) -> List[Dict]:
    """Load test dataset with support for multiple datasets.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'diamonds-seed0', 'function_correctness')
        dataset_set: Which split to load ('train', 'val', 'test')
        test_size: Limit number of test samples
        seed: Random seed
        data_dir: Directory containing parquet files (default: data/{dataset_name})
    
    Returns:
        List of dictionaries, each containing the sample data
    """
    try:
        # Load dataset from parquet file
        if data_dir is None:
            data_dir = os.path.join("data", dataset_name)
        
        parquet_file = os.path.join(data_dir, f"{dataset_set}.parquet")
        
        if os.path.exists(parquet_file):
            logger.info(f"Loading {dataset_name} {dataset_set} dataset from parquet file: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            
            # Extract the data we need from VERL format
            data_rows = []
            for _, row in df.iterrows():
                data_rows.append({
                    'index': row['extra_info']['index'],
                    'prompt': row['prompt'],
                    'measurements': row['extra_info']['measurements_list'],
                    'difficulty': row['extra_info']['difficulty'],
                    'is_correct': row['extra_info']['is_correct'],
                    'text': row['extra_info']['original_text']
                })
            
            # Apply size limit if specified
            if test_size:
                data_rows = data_rows[:min(test_size, len(data_rows))]
            
            logger.info(f"Loaded {dataset_name} {dataset_set} dataset with {len(data_rows)} samples from parquet")
            return data_rows
        
        else:
            raise FileNotFoundError(f"Parquet file not found at {parquet_file}. Please run prepare_dataset.py first to create the parquet files.")
    
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        raise



def load_checkpoint_verl(checkpoint_path: str, device: str = "cuda") -> Tuple[object, object]:
    """Load model and tokenizer from VERL checkpoint."""
    try:
        logger.info(f"Loading VERL checkpoint from {checkpoint_path}")
        
        # Check if it's a HuggingFace model ID (format: namespace/repo_name)
        if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
            # Split to check if it's a valid HF repo format
            parts = checkpoint_path.split("/")
            if len(parts) == 2:
                # Valid HuggingFace format
                logger.info(f"Loading from HuggingFace Hub: {checkpoint_path}")
                model_path = checkpoint_path
            else:
                # Invalid format for HuggingFace
                logger.error(f"Invalid HuggingFace repository format. Expected 'namespace/repo_name', got '{checkpoint_path}'")
                logger.info("If your model is in a subdirectory on HuggingFace, it should be at the repository root.")
                raise ValueError(f"Invalid HuggingFace repository path: {checkpoint_path}")
        elif os.path.exists(checkpoint_path):
            # Local path handling
            # Check if HuggingFace format exists
            hf_path = os.path.join(checkpoint_path, "actor", "huggingface")
            actor_path = os.path.join(checkpoint_path, "actor")
            
            # Check if this is an FSDP checkpoint that needs conversion
            # Check for actual model files in HF format
            hf_has_models = False
            if os.path.exists(hf_path):
                hf_model_files = glob.glob(os.path.join(hf_path, "*.safetensors")) + \
                                 glob.glob(os.path.join(hf_path, "*.bin")) + \
                                 glob.glob(os.path.join(hf_path, "*.pt"))
                hf_has_models = len(hf_model_files) > 0
            
            if os.path.exists(actor_path) and not hf_has_models:
                # Check for FSDP checkpoint files
                fsdp_config_path = os.path.join(actor_path, "fsdp_config.json")
                fsdp_model_files = glob.glob(os.path.join(actor_path, "model_world_size_*_rank_*.pt"))
                
                if os.path.exists(fsdp_config_path) and fsdp_model_files:
                    logger.info("Detected FSDP checkpoint format. Converting to HuggingFace format...")
                    logger.info("This may take a few moments for the first run...")
                    
                    # Run the conversion
                    import subprocess
                    conversion_cmd = [
                        "python", "-m", "verl.model_merger", "merge",
                        "--backend", "fsdp",
                        "--local_dir", actor_path,
                        "--target_dir", hf_path
                    ]
                    
                    try:
                        result = subprocess.run(conversion_cmd, capture_output=True, text=True, check=True)
                        logger.info("Successfully converted FSDP checkpoint to HuggingFace format")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to convert FSDP checkpoint: {e}")
                        logger.error(f"Command output: {e.stderr}")
                        raise
            
            # Now check again for HuggingFace format
            if os.path.exists(hf_path):
                logger.info(f"Loading from local HuggingFace format at: {hf_path}")
                model_path = hf_path
            else:
                # Try loading from actor directory directly
                if os.path.exists(actor_path):
                    logger.info(f"Loading from local actor directory: {actor_path}")
                    model_path = actor_path
                else:
                    # Try as direct path
                    logger.info(f"Loading from local direct path: {checkpoint_path}")
                    model_path = checkpoint_path
        else:
            # Not a valid HF repo and doesn't exist locally
            logger.info(f"Attempting to load as path: {checkpoint_path}")
            model_path = checkpoint_path
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


def evaluate_single_sample(model, tokenizer, sample, temperature: float = 0.1, dataset_name: str = "diamonds-seed0") -> Dict:
    """Evaluate a single sample and return predictions."""
    prompt = tokenizer.apply_chat_template(
        sample['prompt'], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=MAX_LEN_PROMPT, 
        truncation=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_LEN_GENERATION,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    predicted = extract_measurements(response, dataset_name=dataset_name)
    ground_truth = sample['measurements']
    
    result = {
        'text': sample['text'],
        'response': response,
        'predicted': predicted,
        'ground_truth': ground_truth,
        'n_measurements': len(ground_truth) if ground_truth else 0,
        'difficulty': sample.get('difficulty', -1),
        'latent_variable': sample.get('is_correct', None),
        'has_measurements': '<measurements>' in response and '</measurements>' in response,
        'correct_format': predicted is not None,
        'wrong_nb_measurements': len(predicted) != len(ground_truth) if predicted is not None and ground_truth else False,
    }
    
    if predicted is not None and ground_truth:
        result['is_correct'] = predicted == ground_truth
        
        # Handle length mismatches by comparing minimum matching lengths
        if len(predicted) != len(ground_truth):
            min_length = min(len(predicted), len(ground_truth))
            predicted_truncated = predicted[:min_length]
            ground_truth_truncated = ground_truth[:min_length]
            result['partial_correct'] = sum(p == g for p, g in zip(predicted_truncated, ground_truth_truncated))
        else:
            result['partial_correct'] = sum(p == g for p, g in zip(predicted, ground_truth))
    else:
        result['is_correct'] = False
        result['partial_correct'] = 0
    
    return result


def evaluate_checkpoint(checkpoint_path: str, 
                       eval_dataset: List[Dict],
                       num_samples: int = None,
                       temperature: float = 0.1,
                       save_predictions: bool = True,
                       dataset_name: str = "diamonds-seed0",
                       dataset_set: str = "test",
                       output_dir: str = None) -> Dict:
    """Evaluate a single checkpoint on test dataset."""
    
    # Load model
    model, tokenizer = load_checkpoint_verl(checkpoint_path)
    
    # Sample subset if requested
    if num_samples and num_samples < len(eval_dataset):
        indices = np.random.choice(len(eval_dataset), num_samples, replace=False)
        eval_dataset = eval_dataset.select(indices)
    else:
        eval_dataset = eval_dataset
    
    # Setup output files
    checkpoint_name = os.path.basename(checkpoint_path)
    
    if save_predictions:
        if output_dir:
            predictions_dir = os.path.join(output_dir, "predictions")
        else:
            predictions_dir = os.path.join(os.path.dirname(checkpoint_path), "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
    
    # Evaluation metrics
    results = []
    metrics_by_difficulty = defaultdict(lambda: {
        'total': 0, 'n_measurements': 0, 'correct': 0, 'partial': 0, 'format_correct': 0, 'wrong_nb_measurements': 0
    })
    
    # Evaluate samples
    logger.info(f"Evaluating {len(eval_dataset)} samples...")
    
    for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
        sample = eval_dataset[idx]
        result = evaluate_single_sample(model, tokenizer, sample, temperature, dataset_name)
        results.append(result)
        
        # Update metrics
        difficulty = result['difficulty']
        metrics_by_difficulty[difficulty]['total'] += 1
        metrics_by_difficulty[difficulty]['n_measurements'] += result['n_measurements']
        metrics_by_difficulty['overall']['total'] += 1
        metrics_by_difficulty['overall']['n_measurements'] += result['n_measurements']
        
        if result['is_correct']:
            metrics_by_difficulty[difficulty]['correct'] += 1
            metrics_by_difficulty['overall']['correct'] += 1
        
        metrics_by_difficulty[difficulty]['partial'] += result['partial_correct']
        metrics_by_difficulty['overall']['partial'] += result['partial_correct']
        
        if result['correct_format']:
            metrics_by_difficulty[difficulty]['format_correct'] += 1
            metrics_by_difficulty['overall']['format_correct'] += 1
        
        if result['wrong_nb_measurements']:
            metrics_by_difficulty[difficulty]['wrong_nb_measurements'] += 1
            metrics_by_difficulty['overall']['wrong_nb_measurements'] += 1
    
    # Compute final metrics
    metrics = {
        'checkpoint': checkpoint_name,
        'num_samples': len(eval_dataset),
        'temperature': temperature,
        'timestamp': datetime.now().isoformat(),
        'dataset_name': dataset_name,
        'dataset_set': dataset_set,
    }
    
    for difficulty, stats in metrics_by_difficulty.items():
        total = stats['total']
        if total > 0:
            metrics[f'accuracy_{difficulty}'] = stats['correct'] / total
            metrics[f'partial_accuracy_{difficulty}'] = stats['partial'] / stats['n_measurements']
            metrics[f'format_accuracy_{difficulty}'] = stats['format_correct'] / total
            metrics[f'n_samples_{difficulty}'] = total
            metrics[f'n_measurements_{difficulty}'] = stats['n_measurements']
            metrics[f'n_format_correct_{difficulty}'] = stats['format_correct']
            metrics[f'n_wrong_nb_measurements_{difficulty}'] = stats['wrong_nb_measurements']
    
    # Save predictions if requested
    if save_predictions:
        predictions_file = os.path.join(predictions_dir, f"{checkpoint_name}_predictions.json")
        with open(predictions_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'predictions': results
            }, f, indent=2)
        logger.info(f"Saved predictions to {predictions_file}")
        
        # Save raw outputs for analysis
        raw_outputs_file = os.path.join(predictions_dir, f"{checkpoint_name}_raw_outputs.json")
        raw_outputs = [{
            'sample_idx': i,
            'text': eval_dataset[i]['text'],
            'response': result['response'],
            'predicted': result['predicted'],
            'ground_truth': result['ground_truth'],
            'latent_variable': result.get('latent_variable', None)
        } for i, result in enumerate(results)]
        with open(raw_outputs_file, 'w') as f:
            json.dump(raw_outputs, f, indent=2)
        logger.info(f"Saved raw outputs to {raw_outputs_file}")
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()
    
    return metrics


def evaluate_multiple_checkpoints(checkpoints_dir: str,
                                eval_dataset: List[Dict],
                                checkpoint_pattern: str = "global_step_*",
                                num_samples: int = None,
                                temperature: float = 0.1,
                                save_results: bool = True,
                                dataset_name: str = "diamonds-seed0",
                                dataset_set: str = "test",
                                output_dir: str = None) -> List[Dict]:
    """Evaluate multiple checkpoints and compare results."""
    
    # Find all checkpoints
    checkpoint_paths = sorted(
        [p for p in Path(checkpoints_dir).glob(checkpoint_pattern) if p.is_dir()],
        key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0
    )
    
    if not checkpoint_paths:
        logger.error(f"No checkpoints found in {checkpoints_dir} matching pattern {checkpoint_pattern}")
        return []
    
    logger.info(f"Found {len(checkpoint_paths)} checkpoints to evaluate")
    
    # Evaluate each checkpoint
    all_metrics = []
    for checkpoint_path in checkpoint_paths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {checkpoint_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            metrics = evaluate_checkpoint(
                str(checkpoint_path),
                eval_dataset,
                num_samples=num_samples,
                temperature=temperature,
                save_predictions=save_results,
                output_dir=output_dir,
                dataset_name=dataset_name,
                dataset_set=dataset_set
            )
            all_metrics.append(metrics)
            
            # Log key metrics
            logger.info(f"Overall Accuracy: {metrics.get('accuracy_overall', 0):.3f}")
            logger.info(f"Partial Accuracy: {metrics.get('partial_accuracy_overall', 0):.3f}")
            logger.info(f"Format Accuracy: {metrics.get('format_accuracy_overall', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {checkpoint_path.name}: {e}")
            continue
    
    # Save comparison results
    if save_results and all_metrics:
        base_dir = output_dir or checkpoints_dir
        results_file = os.path.join(base_dir, "checkpoint_comparison.json")
        with open(results_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"\nSaved comparison results to {results_file}")
        
        # Create CSV for easy analysis
        df = pd.DataFrame(all_metrics)
        csv_file = os.path.join(base_dir, "checkpoint_comparison.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV results to {csv_file}")
        
        # Print summary
        print_summary(all_metrics)
    
    return all_metrics


def print_summary(all_metrics: List[Dict]):
    """Print summary of evaluation results."""
    if not all_metrics:
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    
    # Find best checkpoint by overall accuracy
    best_checkpoint = max(all_metrics, key=lambda x: x.get('accuracy_overall', 0))
    logger.info(f"\nBest Checkpoint: {best_checkpoint['checkpoint']}")
    logger.info(f"Overall Accuracy: {best_checkpoint.get('accuracy_overall', 0):.3f}")
    logger.info(f"Partial Accuracy: {best_checkpoint.get('partial_accuracy_overall', 0):.3f}")
    
    # Print progress over checkpoints
    logger.info("\nProgress Over Training:")
    logger.info(f"{'Checkpoint':<20} {'Overall Acc':<15} {'Partial Acc':<15} {'Format Acc':<15}")
    logger.info("-" * 65)
    
    for metrics in all_metrics[-10:]:  # Show last 10 checkpoints
        checkpoint = metrics['checkpoint']
        overall_acc = metrics.get('accuracy_overall', 0)
        partial_acc = metrics.get('partial_accuracy_overall', 0)
        format_acc = metrics.get('format_accuracy_overall', 0)
        logger.info(f"{checkpoint:<20} {overall_acc:<15.3f} {partial_acc:<15.3f} {format_acc:<15.3f}")


def main():
    parser = argparse.ArgumentParser(description="Inference script for VERL checkpoints on DIAMONDS and function correctness datasets")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to single checkpoint (e.g., checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_40)")
    parser.add_argument("--checkpoints-dir", type=str,
                       help="Directory containing multiple checkpoints to evaluate")
    parser.add_argument("--pattern", type=str, default="global_step_*",
                       help="Pattern to match checkpoints (default: global_step_*)")
    parser.add_argument("--dataset-name", type=str, default="diamonds-seed0", 
                       choices=[f"diamonds-seed{i}" for i in range(8)] + ["function_correctness"],
                       help="Dataset to evaluate")
    parser.add_argument("--dataset-set", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of test samples to evaluate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for generation (default: 0.1)")
    parser.add_argument("--save-predictions", action="store_true", default=True,
                       help="Save individual predictions")
    parser.add_argument("--test-size", type=int, default=None,
                       help="Limit test dataset size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Directory to save all evaluation results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Directory containing parquet files (default: data/{dataset-name})")

    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set MAX_LEN_GENERATION depending on the dataset
    global MAX_LEN_GENERATION
    if args.dataset_name.startswith("diamonds-seed"):
        MAX_LEN_GENERATION = 1024
    elif args.dataset_name == "function_correctness":
        MAX_LEN_GENERATION = 2048
    
    # Load eval dataset
    logger.info(f"Loading {args.dataset_name} {args.dataset_set} dataset...")
    eval_dataset = load_test_dataset(dataset_name=args.dataset_name, dataset_set=args.dataset_set, test_size=args.test_size, seed=args.seed, data_dir=args.data_dir)
    
    # Evaluate checkpoints
    if args.checkpoint:
        # Evaluate single checkpoint
        # Skip local path check - let HuggingFace handle remote models
        metrics = evaluate_checkpoint(
            args.checkpoint,
            eval_dataset,
            num_samples=args.num_samples,
            temperature=args.temperature,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_set=args.dataset_set
        )
        
        logger.info("\nFinal Results:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                logger.info(f"{key}: {value:.3f}")
            else:
                logger.info(f"{key}: {value}")
    
    elif args.checkpoints_dir:
        # Evaluate multiple checkpoints
        all_metrics = evaluate_multiple_checkpoints(
            args.checkpoints_dir,
            eval_dataset,
            checkpoint_pattern=args.pattern,
            num_samples=args.num_samples,
            temperature=args.temperature,
            save_results=args.save_predictions,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_set=args.dataset_set
        )
    
    else:
        logger.error("Please provide either --checkpoint or --checkpoints-dir")
        return
    
    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()