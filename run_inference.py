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
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def calculate_confidence_interval(successes: int, total: int, confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """Calculate confidence interval for a proportion using Wilson Score interval."""
    if total == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate z-score for given confidence level
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)
    
    # Wilson Score interval formula
    n = total
    p_hat = successes / n
    
    # Wilson Score interval components
    denominator = 1 + (z_score**2 / n)
    
    # Center of the interval
    center = (p_hat + (z_score**2 / (2 * n))) / denominator
    
    # Margin of error
    margin = (z_score / denominator) * np.sqrt((p_hat * (1 - p_hat) / n) + (z_score**2 / (4 * n**2)))
    
    # Calculate bounds
    lower_bound = max(0.0, center - margin)
    upper_bound = min(1.0, center + margin)
    
    return lower_bound, upper_bound, margin

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
from utils.kl_utils import (
    compute_token_level_kl,
    extract_token_log_probs,
)

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


def load_checkpoint_verl(checkpoint_path: str, device: str = "cuda", use_flash_attention: bool = True) -> Tuple[object, object]:
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
        
        # Try to use flash attention if available
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            "trust_remote_code": True,
        }

        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2 for faster inference")
            except Exception as e:
                logger.info("Flash Attention 2 not available, using default attention")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
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
    ground_truth = list(ground_truth) # convert array to list
    
    # Calculate token counts
    total_word_count = len(response.split())
    total_token_count = len(tokenizer.encode(response, add_special_tokens=False))
    
    # Extract text before measurements
    if '<measurements>' in response:
        text_before_measurements = response.split('<measurements>')[0]
    else:
        text_before_measurements = response
    
    before_measurements_word_count = len(text_before_measurements.split())
    before_measurements_token_count = len(tokenizer.encode(text_before_measurements, add_special_tokens=False))
    
    result = {
        'text': sample['text'],
        'response': response,
        'predicted': predicted,
        'ground_truth': ground_truth,
        'n_measurements': len(ground_truth) if ground_truth is not None else 0,
        'difficulty': sample.get('difficulty', -1),
        'latent_variable': sample.get('is_correct', None),
        'has_measurements': '<measurements>' in response and '</measurements>' in response,
        'correct_format': predicted is not None,
        'wrong_nb_measurements': len(predicted) != len(ground_truth) if predicted is not None and ground_truth is not None else False,
        'total_word_count': total_word_count,
        'total_token_count': total_token_count,
        'before_measurements_word_count': before_measurements_word_count,
        'before_measurements_token_count': before_measurements_token_count,
    }
    
    if predicted is not None and ground_truth is not None:
        result['is_correct'] = predicted == ground_truth
        
        # Handle length mismatches by comparing minimum matching lengths
        if len(predicted) != len(ground_truth):
            min_length = min(len(predicted), len(ground_truth))
            predicted_truncated = predicted[:min_length]
            ground_truth_truncated = ground_truth[:min_length]
            result['partial_correct'] = sum(p == g for p, g in zip(predicted_truncated, ground_truth_truncated))
            result['proportion_correct'] = result['partial_correct'] / len(ground_truth)
        else:
            result['partial_correct'] = sum(p == g for p, g in zip(predicted, ground_truth))
            result['proportion_correct'] = result['partial_correct'] / len(ground_truth)
    else:
        result['is_correct'] = False
        result['partial_correct'] = 0
        result['proportion_correct'] = 0

    return result


def evaluate_batch(model, tokenizer, samples, temperature: float = 0.1, batch_size: int = 8, dataset_name: str = "diamonds-seed0", ref_model=None, kl_penalty: str = "kl") -> List[Dict]:
    """Evaluate a batch of samples and return predictions."""
    results = []

    # Prepare all prompts
    prompts = []
    for sample in samples:
        prompt = tokenizer.apply_chat_template(
            sample['prompt'],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    # Tokenize batch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=MAX_LEN_PROMPT,
        truncation=True,
        padding=True
    ).to(model.device)

    # Generate for batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_LEN_GENERATION,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Optional KL computation using a reference model
    kl_mean_per_seq = None
    kl_sum_per_seq = None
    kl_max_per_seq = None
    if ref_model is not None:
        with torch.no_grad():
            # Match other_inference_with_kl.py: use all-ones attention mask when scoring sequences
            ones_attention_mask = torch.ones_like(outputs)

            # Forward both models on the full sequences
            model_output = model(
                input_ids=outputs,
                attention_mask=ones_attention_mask,
                return_dict=True,
            )
            ref_output = ref_model(
                input_ids=outputs,
                attention_mask=ones_attention_mask,
                return_dict=True,
            )

            # Align logits and tokens for next-token prediction
            generated_tokens = outputs[:, 1:]  # [batch, seq_len-1]
            current_logits = model_output.logits[:, :-1, :]  # [batch, seq_len-1, vocab]
            ref_logits = ref_output.logits[:, :-1, :]  # [batch, seq_len-1, vocab]

            # Extract log probs for actual tokens
            logprobs_current = extract_token_log_probs(current_logits, generated_tokens)
            logprobs_ref = extract_token_log_probs(ref_logits, generated_tokens)

            # Build response mask per sample: 1 for generated tokens, 0 for prompt tokens
            # inputs['attention_mask'] has left padding; its sum per row is prompt length
            prompt_lengths = inputs['attention_mask'].sum(dim=1)  # [batch]
            seq_len_m1 = generated_tokens.shape[1]
            arange_idx = torch.arange(seq_len_m1, device=outputs.device).unsqueeze(0).expand(outputs.size(0), -1)
            response_mask = (arange_idx >= (prompt_lengths.unsqueeze(1) - 1)).float()

            # Token-level KL, then aggregate per sequence
            kl_tokens = compute_token_level_kl(logprobs_current, logprobs_ref, kl_penalty=kl_penalty)
            kl_tokens = kl_tokens * response_mask

            num_tokens = response_mask.sum(dim=-1).clamp(min=1)
            kl_sum_per_seq = kl_tokens.sum(dim=-1)
            kl_mean_per_seq = kl_sum_per_seq / num_tokens
            kl_max_per_seq = kl_tokens.max(dim=-1).values

    # Process each output
    for i, (sample, output) in enumerate(zip(samples, outputs)):
        response = tokenizer.decode(
            output[inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        predicted = extract_measurements(response, dataset_name=dataset_name)
        ground_truth = sample['measurements']
        ground_truth = list(ground_truth) # convert array to list

        # Calculate token counts
        total_word_count = len(response.split())
        total_token_count = len(tokenizer.encode(response, add_special_tokens=False))
        
        # Extract text before measurements
        if '<measurements>' in response:
            text_before_measurements = response.split('<measurements>')[0]
        else:
            text_before_measurements = response
        
        before_measurements_word_count = len(text_before_measurements.split())
        before_measurements_token_count = len(tokenizer.encode(text_before_measurements, add_special_tokens=False))

        result = {
            'text': sample['text'],
            'response': response,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'n_measurements': len(ground_truth) if ground_truth is not None else 0,
            'difficulty': sample.get('difficulty', -1),
            'latent_variable': sample.get('is_correct', None),
            'has_measurements': '<measurements>' in response and '</measurements>' in response,
            'correct_format': predicted is not None,
            'wrong_nb_measurements': len(predicted) != len(ground_truth) if predicted is not None and ground_truth is not None else False,
            'total_word_count': total_word_count,
            'total_token_count': total_token_count,
            'before_measurements_word_count': before_measurements_word_count,
            'before_measurements_token_count': before_measurements_token_count,
        }

        # Attach per-sample KL metrics if available
        if kl_mean_per_seq is not None:
            result['kl_mean'] = float(kl_mean_per_seq[i].item())
            result['kl_sum'] = float(kl_sum_per_seq[i].item())
            result['kl_max'] = float(kl_max_per_seq[i].item())
            result['kl_penalty_type'] = kl_penalty

        if predicted is not None and ground_truth is not None:
            result['is_correct'] = predicted == ground_truth
            
            # Handle length mismatches by comparing minimum matching lengths (function correctness)
            if len(predicted) != len(ground_truth):
                min_length = min(len(predicted), len(ground_truth))
                predicted_truncated = predicted[:min_length]
                ground_truth_truncated = ground_truth[:min_length]
                result['partial_correct'] = sum(p == g for p, g in zip(predicted_truncated, ground_truth_truncated))
                result['proportion_correct'] = result['partial_correct'] / len(ground_truth)
            else:
                result['partial_correct'] = sum(p == g for p, g in zip(predicted, ground_truth))
                result['proportion_correct'] = result['partial_correct'] / len(ground_truth)
        else:
            result['is_correct'] = False
            result['partial_correct'] = 0
            result['proportion_correct'] = 0
        results.append(result)

    return results


def evaluate_checkpoint(checkpoint_path: str, 
                       eval_dataset: List[Dict],
                       num_samples: int = None,
                       temperature: float = 0.1,
                       save_predictions: bool = True,
                       dataset_name: str = "diamonds-seed0",
                       dataset_set: str = "test",
                       output_dir: str = None,
                       batch_size: int = 8,
                       use_flash_attention: bool = True,
                       ref_model_path: Optional[str] = None,
                       kl_penalty: str = "kl") -> Dict:
    """Evaluate a single checkpoint on test dataset."""
    
    # Load model
    model, tokenizer = load_checkpoint_verl(checkpoint_path, use_flash_attention=use_flash_attention)

    # Load reference model
    ref_model = None
    if ref_model_path:
        logger.info(f"Loading reference model from {ref_model_path}")
        ref_model, _ = load_checkpoint_verl(ref_model_path, use_flash_attention=use_flash_attention)
    
    # Sample subset if requested
    if num_samples and num_samples < len(eval_dataset):
        indices = np.random.choice(len(eval_dataset), num_samples, replace=False)
        eval_dataset = [eval_dataset[i] for i in indices]
    
    # Setup output files
    checkpoint_name = os.path.basename(checkpoint_path)
    
    if save_predictions:
        if output_dir:
            # Ensure the base output directory exists first
            os.makedirs(output_dir, exist_ok=True)
            predictions_dir = os.path.join(output_dir, "predictions")
        else:
            predictions_dir = os.path.join(os.path.dirname(checkpoint_path), "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
    
    # Evaluation metrics
    results = []
    metrics_by_difficulty = defaultdict(lambda: {
        'total': 0, 'n_measurements': 0, 'correct': 0, 'partial': 0, 'proportion_correct': 0, 'format_correct': 0, 'wrong_nb_measurements': 0,
    'total_word_count': 0, 'total_token_count': 0, 'before_measurements_word_count': 0, 'before_measurements_token_count': 0,
        'kl_mean_sum': 0.0,
    })
    
    # Evaluate samples
    logger.info(f"Evaluating {len(eval_dataset)} samples with batch size {batch_size}...")

    # Process in batches
    for batch_start in tqdm(range(0, len(eval_dataset), batch_size), desc="Evaluating batches"):
        batch_end = min(batch_start + batch_size, len(eval_dataset))
        batch_samples = [eval_dataset[i] for i in range(batch_start, batch_end)]

        # Evaluate batch
        batch_results = evaluate_batch(
            model,
            tokenizer,
            batch_samples,
            temperature,
            batch_size,
            dataset_name,
            ref_model=ref_model,
            kl_penalty=kl_penalty,
        )
        results.extend(batch_results)

        # Update metrics for batch
        for result in batch_results:
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
            metrics_by_difficulty[difficulty]['proportion_correct'] += result['proportion_correct']
            metrics_by_difficulty['overall']['proportion_correct'] += result['proportion_correct']

            if result['correct_format']:
                metrics_by_difficulty[difficulty]['format_correct'] += 1
                metrics_by_difficulty['overall']['format_correct'] += 1

            if result['wrong_nb_measurements']:
                metrics_by_difficulty[difficulty]['wrong_nb_measurements'] += 1
                metrics_by_difficulty['overall']['wrong_nb_measurements'] += 1
            
            # Accumulate token counts
            metrics_by_difficulty[difficulty]['total_word_count'] += result['total_word_count']
            metrics_by_difficulty[difficulty]['total_token_count'] += result['total_token_count']
            metrics_by_difficulty[difficulty]['before_measurements_word_count'] += result['before_measurements_word_count']
            metrics_by_difficulty[difficulty]['before_measurements_token_count'] += result['before_measurements_token_count']
            
            metrics_by_difficulty['overall']['total_word_count'] += result['total_word_count']
            metrics_by_difficulty['overall']['total_token_count'] += result['total_token_count']
            metrics_by_difficulty['overall']['before_measurements_word_count'] += result['before_measurements_word_count']
            metrics_by_difficulty['overall']['before_measurements_token_count'] += result['before_measurements_token_count']

            # Accumulate KL if present
            if 'kl_mean' in result:
                metrics_by_difficulty[difficulty]['kl_mean_sum'] += result['kl_mean']
                metrics_by_difficulty['overall']['kl_mean_sum'] += result['kl_mean']
    
    # Compute final metrics
    metrics = {
        'checkpoint': checkpoint_name,
        'num_samples': len(eval_dataset),
        'temperature': temperature,
        'timestamp': datetime.now().isoformat(),
        'dataset_name': dataset_name,
        'dataset_set': dataset_set,
        'has_kl': ref_model is not None,
    }
    
    for difficulty, stats in metrics_by_difficulty.items():
        total = stats['total']
        if total > 0:
            # Calculate basic metrics
            measurement_wise_acc = stats['proportion_correct'] / total
            all_correct_acc = stats['correct'] / total
            format_acc = stats['format_correct'] / total
            partial_acc = stats['partial'] / stats['n_measurements']
            
            # Calculate confidence intervals
            # For measurement-wise accuracy, we need to calculate it differently since it's an average of proportions
            # We'll use the total number of measurements as the denominator
            measurement_wise_lower, measurement_wise_upper, measurement_wise_margin = calculate_confidence_interval(
                stats['proportion_correct'], total
            )
            
            all_correct_lower, all_correct_upper, all_correct_margin = calculate_confidence_interval(
                stats['correct'], total
            )
            
            format_lower, format_upper, format_margin = calculate_confidence_interval(
                stats['format_correct'], total
            )
            
            partial_lower, partial_upper, partial_margin = calculate_confidence_interval(
                stats['partial'], stats['n_measurements']
            )
            
            # Store basic metrics with confidence intervals as lists [lower, upper, margin]
            metrics[f'measurement-wise_accuracy_{difficulty}'] = {
                'value': measurement_wise_acc,
                'confidence_interval': [measurement_wise_lower, measurement_wise_upper, measurement_wise_margin]
            }
            metrics[f'all_correct_accuracy_{difficulty}'] = {
                'value': all_correct_acc,
                'confidence_interval': [all_correct_lower, all_correct_upper, all_correct_margin]
            }
            metrics[f'format_accuracy_{difficulty}'] = {
                'value': format_acc,
                'confidence_interval': [format_lower, format_upper, format_margin]
            }
            metrics[f'partial_accuracy_{difficulty}'] = {
                'value': partial_acc,
                'confidence_interval': [partial_lower, partial_upper, partial_margin]
            }
            
            # Store sample counts
            metrics[f'n_samples_{difficulty}'] = total
            metrics[f'n_measurements_{difficulty}'] = stats['n_measurements']
            metrics[f'n_format_correct_{difficulty}'] = stats['format_correct']
            metrics[f'n_wrong_nb_measurements_{difficulty}'] = stats['wrong_nb_measurements']
            
            # Store token count metrics
            metrics[f'avg_total_word_count_{difficulty}'] = stats['total_word_count'] / total
            metrics[f'avg_total_token_count_{difficulty}'] = stats['total_token_count'] / total
            metrics[f'avg_before_measurements_word_count_{difficulty}'] = stats['before_measurements_word_count'] / total
            metrics[f'avg_before_measurements_token_count_{difficulty}'] = stats['before_measurements_token_count'] / total

            # Add average KL metrics if available
            if metrics.get('has_kl') and stats.get('kl_mean_sum', 0) != 0:
                metrics[f'kl_mean_{difficulty}'] = stats['kl_mean_sum'] / total
    
    # Save predictions if requested
    if save_predictions:
        predictions_file = os.path.join(predictions_dir, f"{checkpoint_name}_predictions.json")
        with open(predictions_file, 'w') as f:
            json.dump(convert_numpy_types({
                'metrics': metrics,
                'predictions': results
            }), f, indent=2)
        logger.info(f"Saved predictions to {predictions_file}")
        
        # Save raw outputs for analysis
        raw_outputs_file = os.path.join(predictions_dir, f"{checkpoint_name}_raw_outputs.json")
        raw_outputs = [{
            'sample_idx': i,
            'text': eval_dataset[i]['text'],
            'response': result['response'],
            'predicted': result['predicted'],
            'ground_truth': result['ground_truth'],
            'latent_variable': result.get('latent_variable', None),
            'kl_mean': result.get('kl_mean', None),
            'kl_sum': result.get('kl_sum', None),
            'kl_max': result.get('kl_max', None),
            'kl_penalty_type': result.get('kl_penalty_type', None),
        } for i, result in enumerate(results)]
        with open(raw_outputs_file, 'w') as f:
            json.dump(convert_numpy_types(raw_outputs), f, indent=2)
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
                                output_dir: str = None,
                                batch_size: int = 8,
                                use_flash_attention: bool = True,
                                keep_model_loaded: bool = False,
                                ref_model_path: Optional[str] = None,
                                kl_penalty: str = "kl") -> List[Dict]:
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
    loaded_model = None
    loaded_tokenizer = None

    for i, checkpoint_path in enumerate(checkpoint_paths):
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {checkpoint_path.name} ({i+1}/{len(checkpoint_paths)})")
        logger.info(f"{'='*60}")
        
        try:
            if keep_model_loaded and i == 0:
                # Load first model and keep it for comparison
                loaded_model, loaded_tokenizer = load_checkpoint_verl(
                    str(checkpoint_path),
                    use_flash_attention=use_flash_attention
                )

            metrics = evaluate_checkpoint(
                str(checkpoint_path),
                eval_dataset,
                num_samples=num_samples,
                temperature=temperature,
                save_predictions=save_results,
                output_dir=output_dir,
                dataset_name=dataset_name,
                dataset_set=dataset_set,
                batch_size=batch_size,
                use_flash_attention=use_flash_attention,
                ref_model_path=ref_model_path,
                kl_penalty=kl_penalty,
            )
            all_metrics.append(metrics)
            
            # Log key metrics with confidence intervals
            measurement_wise_data = metrics.get('measurement-wise_accuracy_overall', {'value': 0, 'confidence_interval': [0, 0, 0]})
            measurement_wise_acc = measurement_wise_data['value']
            measurement_wise_ci = measurement_wise_data['confidence_interval']
            measurement_wise_margin = measurement_wise_ci[2]
            logger.info(f"Measurement-wise Accuracy: {measurement_wise_acc:.3f} ± {measurement_wise_margin:.3f} (95% CI: [{measurement_wise_ci[0]:.3f}, {measurement_wise_ci[1]:.3f}])")
            
            all_correct_data = metrics.get('all_correct_accuracy_overall', {'value': 0, 'confidence_interval': [0, 0, 0]})
            all_correct_acc = all_correct_data['value']
            all_correct_ci = all_correct_data['confidence_interval']
            all_correct_margin = all_correct_ci[2]
            logger.info(f"All Correct Accuracy: {all_correct_acc:.3f} ± {all_correct_margin:.3f} (95% CI: [{all_correct_ci[0]:.3f}, {all_correct_ci[1]:.3f}])")
            
            format_data = metrics.get('format_accuracy_overall', {'value': 0, 'confidence_interval': [0, 0, 0]})
            format_acc = format_data['value']
            format_ci = format_data['confidence_interval']
            format_margin = format_ci[2]
            logger.info(f"Format Accuracy: {format_acc:.3f} ± {format_margin:.3f} (95% CI: [{format_ci[0]:.3f}, {format_ci[1]:.3f}])")
            
            # Log token count information
            avg_total_words = metrics.get('avg_total_word_count_overall', 0)
            avg_total_tokens = metrics.get('avg_total_token_count_overall', 0)
            avg_before_measurements_words = metrics.get('avg_before_measurements_word_count_overall', 0)
            avg_before_measurements_tokens = metrics.get('avg_before_measurements_token_count_overall', 0)
            logger.info(f"Avg Total Response: {avg_total_words:.1f} words, {avg_total_tokens:.1f} tokens")
            logger.info(f"Avg Before Measurements: {avg_before_measurements_words:.1f} words, {avg_before_measurements_tokens:.1f} tokens")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {checkpoint_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Clean up if we kept model loaded
    if loaded_model is not None:
        del loaded_model
        torch.cuda.empty_cache()
    
    # Save comparison results
    if save_results and all_metrics:
        base_dir = output_dir or checkpoints_dir
        results_file = os.path.join(base_dir, "checkpoint_comparison.json")
        with open(results_file, 'w') as f:
            json.dump(convert_numpy_types(all_metrics), f, indent=2)
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
    best_checkpoint = max(all_metrics, key=lambda x: x.get('measurement-wise_accuracy_overall', {'value': 0})['value'])
    logger.info(f"\nBest Checkpoint: {best_checkpoint['checkpoint']}")
    
    measurement_wise_data = best_checkpoint.get('measurement-wise_accuracy_overall', {'value': 0, 'confidence_interval': [0, 0, 0]})
    measurement_wise_acc = measurement_wise_data['value']
    measurement_wise_ci = measurement_wise_data['confidence_interval']
    measurement_wise_margin = measurement_wise_ci[2]
    logger.info(f"Measurement-wise Accuracy: {measurement_wise_acc:.3f} ± {measurement_wise_margin:.3f} (95% CI: [{measurement_wise_ci[0]:.3f}, {measurement_wise_ci[1]:.3f}])")
    
    all_correct_data = best_checkpoint.get('all_correct_accuracy_overall', {'value': 0, 'confidence_interval': [0, 0, 0]})
    all_correct_acc = all_correct_data['value']
    all_correct_ci = all_correct_data['confidence_interval']
    all_correct_margin = all_correct_ci[2]
    logger.info(f"All Correct Accuracy: {all_correct_acc:.3f} ± {all_correct_margin:.3f} (95% CI: [{all_correct_ci[0]:.3f}, {all_correct_ci[1]:.3f}])")
    
    format_data = best_checkpoint.get('format_accuracy_overall', {'value': 0, 'confidence_interval': [0, 0, 0]})
    format_acc = format_data['value']
    format_ci = format_data['confidence_interval']
    format_margin = format_ci[2]
    logger.info(f"Format Accuracy: {format_acc:.3f} ± {format_margin:.3f} (95% CI: [{format_ci[0]:.3f}, {format_ci[1]:.3f}])")
    
    # Log token count information for best checkpoint
    avg_total_words = best_checkpoint.get('avg_total_word_count_overall', 0)
    avg_total_tokens = best_checkpoint.get('avg_total_token_count_overall', 0)
    avg_before_measurements_words = best_checkpoint.get('avg_before_measurements_word_count_overall', 0)
    avg_before_measurements_tokens = best_checkpoint.get('avg_before_measurements_token_count_overall', 0)
    logger.info(f"Avg Total Response: {avg_total_words:.1f} words, {avg_total_tokens:.1f} tokens")
    logger.info(f"Avg Before Measurements: {avg_before_measurements_words:.1f} words, {avg_before_measurements_tokens:.1f} tokens")
    
    # Print progress over checkpoints
    logger.info("\nProgress Over Training:")
    logger.info(f"{'Checkpoint':<20} {'Measurement-wise Acc (±95% CI)':<35} {'All Correct Acc (±95% CI)':<35} {'Format Acc (±95% CI)':<35}")
    logger.info("-" * 125)
    
    for metrics in all_metrics[-10:]:  # Show last 10 checkpoints
        checkpoint = metrics['checkpoint']
        measurement_wise_data = metrics.get('measurement-wise_accuracy_overall', {'value': 0, 'confidence_interval': [0, 0, 0]})
        measurement_wise_acc = measurement_wise_data['value']
        measurement_wise_margin = measurement_wise_data['confidence_interval'][2]
        all_correct_data = metrics.get('all_correct_accuracy_overall', {'value': 0, 'confidence_interval': [0, 0, 0]})
        all_correct_acc = all_correct_data['value']
        all_correct_margin = all_correct_data['confidence_interval'][2]
        format_data = metrics.get('format_accuracy_overall', {'value': 0, 'confidence_interval': [0, 0, 0]})
        format_acc = format_data['value']
        format_margin = format_data['confidence_interval'][2]
        
        measurement_wise_str = f"{measurement_wise_acc:.3f}±{measurement_wise_margin:.3f}"
        all_correct_str = f"{all_correct_acc:.3f}±{all_correct_margin:.3f}"
        format_str = f"{format_acc:.3f}±{format_margin:.3f}"
        
        logger.info(f"{checkpoint:<20} {measurement_wise_str:<35} {all_correct_str:<35} {format_str:<35}")


def main():
    parser = argparse.ArgumentParser(description="Inference script for VERL checkpoints on DIAMONDS and function correctness datasets")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to single checkpoint (e.g., checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_40)")
    parser.add_argument("--ref-model", type=str, default=None,
                       help="Optional path to reference model for KL calculation (e.g., base model or initial checkpoint)")
    parser.add_argument("--checkpoints-dir", type=str,
                       help="Directory containing multiple checkpoints to evaluate")
    parser.add_argument("--pattern", type=str, default="global_step_*",
                       help="Pattern to match checkpoints (default: global_step_*)")
    parser.add_argument("--dataset-name", type=str, default="diamonds-seed0", 
                       choices=[f"diamonds-seed{i}" for i in range(8)] + ["function_correctness"],
                       help="Dataset to evaluate")
    parser.add_argument("--dataset-set", type=str, default="val",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of test samples to evaluate (default: None for all)")
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
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference (default: 8)")
    parser.add_argument("--no-flash-attention", action="store_true",
                       help="Disable flash attention even if available")
    parser.add_argument("--keep-model-loaded", action="store_true",
                       help="Keep model loaded between checkpoints (experimental)")
    parser.add_argument("--kl-penalty", type=str, default="kl",
                       choices=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"],
                       help="Type of KL penalty to use when computing KL (default: kl)")

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
            dataset_set=args.dataset_set,
            batch_size=args.batch_size,
            use_flash_attention=not args.no_flash_attention,
            ref_model_path=args.ref_model,
            kl_penalty=args.kl_penalty,
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
            dataset_set=args.dataset_set,
            batch_size=args.batch_size,
            use_flash_attention=not args.no_flash_attention,
            keep_model_loaded=args.keep_model_loaded,
            ref_model_path=args.ref_model,
            kl_penalty=args.kl_penalty,
        )
    
    else:
        logger.error("Please provide either --checkpoint or --checkpoints-dir")
        return
    
    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()