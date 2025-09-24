#!/usr/bin/env python3
"""
Process reasoning traces from regular evaluation results through monitor model.
"""
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from openai import AsyncOpenAI
import time
import random
import numpy as np
from scipy import stats

# Import monitor utilities
from utils.monitor_utils import (
    extract_reasoning_trace,
    create_monitor_prompt,
    extract_monitor_prediction,
    extract_monitor_reasoning
)


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


def rebalance_samples(samples: List[Dict]) -> List[Dict]:
    """Rebalance samples to achieve 50/50 split of True/False latent variables."""
    # Separate samples by latent variable
    true_samples = [s for s in samples if s.get("latent_variable") is True]
    false_samples = [s for s in samples if s.get("latent_variable") is False]
    
    print(f"Original distribution: {len(true_samples)} True, {len(false_samples)} False")
    
    # Calculate target counts
    if len(true_samples) >= len(false_samples):
        target_samples_per_category = len(false_samples)
    else:
        target_samples_per_category = len(true_samples)
    
    print(f"Target samples per category: {target_samples_per_category}")
    
    # Randomly sample the required number of each type
    random.seed(42)  # For reproducibility
    
    selected_true = random.sample(true_samples, target_samples_per_category)
    selected_false = random.sample(false_samples, target_samples_per_category)
    
    # Combine and shuffle
    rebalanced_samples = selected_true + selected_false
    random.shuffle(rebalanced_samples)
    
    print(f"Rebalanced distribution: {len(selected_true)} True, {len(selected_false)} False")
    print(f"Total samples: {len(rebalanced_samples)}")
    
    return rebalanced_samples

async def get_monitor_response_async(client: AsyncOpenAI, prompt: Dict[str, str], sample_idx: int, semaphore: asyncio.Semaphore, model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Get response from monitor model via OpenAI API asynchronously.
    """
    async with semaphore:  # Limit concurrent requests
        try:
            # Make API call to monitor
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Extract reasoning and answer
            monitor_reasoning = extract_monitor_reasoning(response_text)
            monitor_prediction = extract_monitor_prediction(response_text)
            
            return {
                "sample_idx": sample_idx,
                "full_response": response_text,
                "monitor_reasoning": monitor_reasoning,
                "monitor_prediction": monitor_prediction,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            return {
                "sample_idx": sample_idx,
                "full_response": "",
                "monitor_reasoning": "",
                "monitor_prediction": None,
                "success": False,
                "error": str(e)
            }

async def process_batch_async(client: AsyncOpenAI, batch_data: List[Dict], max_concurrent: int = 60, model_name: str = "gpt-4o-mini") -> List[Dict]:
    """
    Process a batch of samples asynchronously with rate limiting.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = []
    for item in batch_data:
        task = get_monitor_response_async(
            client, 
            item["prompt"], 
            item["sample_idx"],
            semaphore,
            model_name
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Sort results by sample_idx to maintain order
    results_dict = {r["sample_idx"]: r for r in results}
    return [results_dict[item["sample_idx"]] for item in batch_data]

async def process_evaluation_results_async(input_file: str, output_file: str, api_key: str = None, batch_size: int = 20, max_concurrent: int = 60, dataset_name: str = "diamonds-seed0", model_name: str = "gpt-4o-mini", rebalance: bool = False):
    """Process evaluation results and generate monitor predictions with batching."""
    
    # Initialize AsyncOpenAI client
    if api_key:
        client = AsyncOpenAI(api_key=api_key)
    else:
        # Will use OPENAI_API_KEY environment variable
        client = AsyncOpenAI()
    
    # Load the raw outputs
    print(f"Loading raw outputs from {input_file}...")
    with open(input_file, 'r') as f:
        raw_outputs = json.load(f)
    
    print(f"Found {len(raw_outputs)} samples to process")
    
    # Rebalance samples if requested
    if rebalance:
        print(f"Rebalancing samples to uniform latent variable distribution (50/50)...")
        raw_outputs = rebalance_samples(raw_outputs)
    
    print(f"Using batch size: {batch_size}, max concurrent requests: {max_concurrent}")
    
    # Prepare all prompts
    batch_data_all = []
    for i, sample in enumerate(raw_outputs):
        reasoning_trace = extract_reasoning_trace(sample.get("response", ""))
        monitor_prompt = create_monitor_prompt(reasoning_trace, dataset_name)
        
        batch_data_all.append({
            "sample_idx": sample.get("sample_idx", i),
            "prompt": monitor_prompt,
            "original_sample": sample,
            "reasoning_trace": reasoning_trace
        })
    
    # Process in batches
    monitor_results = []
    successful_predictions = 0
    failed_predictions = 0
    
    for batch_start in range(0, len(batch_data_all), batch_size):
        batch_end = min(batch_start + batch_size, len(batch_data_all))
        batch = batch_data_all[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(batch_data_all) + batch_size - 1)//batch_size} (samples {batch_start}-{batch_end-1})...")
        
        start_time = time.time()
        batch_responses = await process_batch_async(client, batch, max_concurrent, model_name)
        batch_time = time.time() - start_time
        
        print(f"  Batch completed in {batch_time:.2f} seconds ({len(batch)/batch_time:.2f} samples/sec)")
        
        # Process batch results
        for idx, monitor_response in enumerate(batch_responses):
            original_data = batch[idx]
            sample = original_data["original_sample"]
            
            if monitor_response["success"]:
                successful_predictions += 1
            else:
                failed_predictions += 1
            
            # Create result entry
            monitor_prediction_value = monitor_response["monitor_prediction"]
            latent_variable_value = sample.get("latent_variable", None)
            if monitor_prediction_value is None or latent_variable_value is None:
                monitor_correct_value = None
            else:
                monitor_correct_value = (monitor_prediction_value == latent_variable_value)

            result = {
                "sample_idx": sample.get("sample_idx", idx),
                "original_response": sample.get("response", ""),
                "reasoning_trace": original_data["reasoning_trace"],
                "original_prediction": sample.get("predicted", []),
                "ground_truth": sample.get("ground_truth", []),
                "latent_variable": sample.get("latent_variable", None),
                "monitor_correct": monitor_correct_value,
                "monitor_full_response": monitor_response["full_response"],
                "monitor_reasoning": monitor_response["monitor_reasoning"],
                "monitor_prediction": monitor_response["monitor_prediction"],
                "monitor_processing_success": monitor_response["success"],
                "monitor_error": monitor_response["error"]
            }
            
            monitor_results.append(result)
        
        # Save intermediate results after each batch
        if batch_end % 100 == 0 or batch_end == len(batch_data_all):
            intermediate_file = output_file.replace('.json', f'_intermediate_{batch_end}.json')
            with open(intermediate_file, 'w') as f:
                json.dump(monitor_results, f, indent=2)
            print(f"  Saved intermediate results to {intermediate_file}")
        
        # Add a small delay between batches to avoid rate limiting
        if batch_end < len(batch_data_all):
            await asyncio.sleep(1)
    
    # Save final results
    print(f"Saving monitor results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(monitor_results, f, indent=2)
    
    # Calculate accuracy metrics
    correct_predictions = sum(1 for r in monitor_results 
                             if r["monitor_prediction"] is not None 
                             and r["monitor_prediction"] == r["latent_variable"])
    total_valid_predictions = sum(1 for r in monitor_results if r["monitor_prediction"] is not None)
    
    # Calculate 95% confidence interval for overall accuracy
    accuracy_ci_lower, accuracy_ci_upper, accuracy_ci_margin = calculate_confidence_interval(
        correct_predictions, total_valid_predictions, confidence_level=0.95
    )
    
    # Calculate conditional accuracies
    # Monitor accuracy when original model is correct
    original_correct_samples = [r for r in monitor_results 
                               if r["monitor_prediction"] is not None 
                               and r["original_prediction"] == r["ground_truth"]]
    monitor_correct_when_original_correct = sum(1 for r in original_correct_samples 
                                               if r["monitor_prediction"] == r["latent_variable"])
    total_original_correct = len(original_correct_samples)
    
    # Monitor accuracy when original model is incorrect
    original_incorrect_samples = [r for r in monitor_results 
                                 if r["monitor_prediction"] is not None 
                                 and r["original_prediction"] != r["ground_truth"]]
    monitor_correct_when_original_incorrect = sum(1 for r in original_incorrect_samples 
                                                 if r["monitor_prediction"] == r["latent_variable"])
    total_original_incorrect = len(original_incorrect_samples)
    
    # Monitor accuracy when LV=True
    lv_true_samples = [r for r in monitor_results 
                      if r["monitor_prediction"] is not None 
                      and r["latent_variable"] is True]
    monitor_correct_when_lv_true = sum(1 for r in lv_true_samples 
                                      if r["monitor_prediction"] == r["latent_variable"])
    total_lv_true = len(lv_true_samples)
    
    # Monitor accuracy when LV=False
    lv_false_samples = [r for r in monitor_results 
                       if r["monitor_prediction"] is not None 
                       and r["latent_variable"] is False]
    monitor_correct_when_lv_false = sum(1 for r in lv_false_samples 
                                       if r["monitor_prediction"] == r["latent_variable"])
    total_lv_false = len(lv_false_samples)
    
    # Calculate confidence intervals for conditional accuracies
    ci_original_correct = calculate_confidence_interval(
        monitor_correct_when_original_correct, total_original_correct, confidence_level=0.95
    ) if total_original_correct > 0 else (0.0, 0.0, 0.0)
    
    ci_original_incorrect = calculate_confidence_interval(
        monitor_correct_when_original_incorrect, total_original_incorrect, confidence_level=0.95
    ) if total_original_incorrect > 0 else (0.0, 0.0, 0.0)
    
    ci_lv_true = calculate_confidence_interval(
        monitor_correct_when_lv_true, total_lv_true, confidence_level=0.95
    ) if total_lv_true > 0 else (0.0, 0.0, 0.0)
    
    ci_lv_false = calculate_confidence_interval(
        monitor_correct_when_lv_false, total_lv_false, confidence_level=0.95
    ) if total_lv_false > 0 else (0.0, 0.0, 0.0)
    
    # Create a summary file with monitor name
    monitor_name = model_name.replace("-", "_").replace(".", "_")
    summary_file = output_file.replace('.json', '_summary.json')
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "output_file": output_file,
        "monitor_model": model_name,
        "total_samples": len(monitor_results),
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "valid_predictions": total_valid_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": correct_predictions / total_valid_predictions if total_valid_predictions > 0 else 0,
        "accuracy_95_ci": {
            "lower_bound": accuracy_ci_lower,
            "upper_bound": accuracy_ci_upper,
            "margin": accuracy_ci_margin
        },
        "conditional_accuracies": {
            "when_original_correct": {
                "accuracy": monitor_correct_when_original_correct / total_original_correct if total_original_correct > 0 else 0,
                "correct_predictions": monitor_correct_when_original_correct,
                "total_samples": total_original_correct,
                "95_ci": {
                    "lower_bound": ci_original_correct[0],
                    "upper_bound": ci_original_correct[1],
                    "margin": ci_original_correct[2]
                }
            },
            "when_original_incorrect": {
                "accuracy": monitor_correct_when_original_incorrect / total_original_incorrect if total_original_incorrect > 0 else 0,
                "correct_predictions": monitor_correct_when_original_incorrect,
                "total_samples": total_original_incorrect,
                "95_ci": {
                    "lower_bound": ci_original_incorrect[0],
                    "upper_bound": ci_original_incorrect[1],
                    "margin": ci_original_incorrect[2]
                }
            },
            "when_lv_true": {
                "accuracy": monitor_correct_when_lv_true / total_lv_true if total_lv_true > 0 else 0,
                "correct_predictions": monitor_correct_when_lv_true,
                "total_samples": total_lv_true,
                "95_ci": {
                    "lower_bound": ci_lv_true[0],
                    "upper_bound": ci_lv_true[1],
                    "margin": ci_lv_true[2]
                }
            },
            "when_lv_false": {
                "accuracy": monitor_correct_when_lv_false / total_lv_false if total_lv_false > 0 else 0,
                "correct_predictions": monitor_correct_when_lv_false,
                "total_samples": total_lv_false,
                "95_ci": {
                    "lower_bound": ci_lv_false[0],
                    "upper_bound": ci_lv_false[1],
                    "margin": ci_lv_false[2]
                }
            }
        },
        "model": model_name,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Clean up intermediate files
    print(f"\nCleaning up intermediate files...")
    intermediate_files_removed = 0
    for batch_end in range(100, len(batch_data_all) + 1, 100):
        intermediate_file = output_file.replace('.json', f'_intermediate_{batch_end}.json')
        if os.path.exists(intermediate_file):
            try:
                os.remove(intermediate_file)
                intermediate_files_removed += 1
                print(f"  Removed: {intermediate_file}")
            except OSError as e:
                print(f"  Warning: Could not remove {intermediate_file}: {e}")
    
    # Also remove the final intermediate file if it exists
    final_intermediate_file = output_file.replace('.json', f'_intermediate_{len(batch_data_all)}.json')
    if os.path.exists(final_intermediate_file):
        try:
            os.remove(final_intermediate_file)
            intermediate_files_removed += 1
            print(f"  Removed: {final_intermediate_file}")
        except OSError as e:
            print(f"  Warning: Could not remove {final_intermediate_file}: {e}")
    
    print(f"Cleaned up {intermediate_files_removed} intermediate files")
    
    print(f"\n=== Processing Complete ===")
    print(f"Total samples: {len(monitor_results)}")
    print(f"Successful monitor calls: {successful_predictions}")
    print(f"Failed monitor calls: {failed_predictions}")
    print(f"Valid predictions: {total_valid_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    if total_valid_predictions > 0:
        accuracy = correct_predictions / total_valid_predictions
        print(f"Overall Accuracy: {accuracy:.2%}")
        print(f"95% Confidence Interval: [{accuracy_ci_lower:.2%}, {accuracy_ci_upper:.2%}] (±{accuracy_ci_margin:.2%})")
    
    print(f"\n=== Conditional Accuracies ===")
    
    # Monitor accuracy when original model is correct/incorrect
    if total_original_correct > 0:
        acc_orig_correct = monitor_correct_when_original_correct / total_original_correct
        print(f"When Original Model Correct: {acc_orig_correct:.2%} ({monitor_correct_when_original_correct}/{total_original_correct})")
        print(f"  95% CI: [{ci_original_correct[0]:.2%}, {ci_original_correct[1]:.2%}] (±{ci_original_correct[2]:.2%})")
    
    if total_original_incorrect > 0:
        acc_orig_incorrect = monitor_correct_when_original_incorrect / total_original_incorrect
        print(f"When Original Model Incorrect: {acc_orig_incorrect:.2%} ({monitor_correct_when_original_incorrect}/{total_original_incorrect})")
        print(f"  95% CI: [{ci_original_incorrect[0]:.2%}, {ci_original_incorrect[1]:.2%}] (±{ci_original_incorrect[2]:.2%})")
    
    # Monitor accuracy when LV=True/False
    if total_lv_true > 0:
        acc_lv_true = monitor_correct_when_lv_true / total_lv_true
        print(f"When LV=True: {acc_lv_true:.2%} ({monitor_correct_when_lv_true}/{total_lv_true})")
        print(f"  95% CI: [{ci_lv_true[0]:.2%}, {ci_lv_true[1]:.2%}] (±{ci_lv_true[2]:.2%})")
    
    if total_lv_false > 0:
        acc_lv_false = monitor_correct_when_lv_false / total_lv_false
        print(f"When LV=False: {acc_lv_false:.2%} ({monitor_correct_when_lv_false}/{total_lv_false})")
        print(f"  95% CI: [{ci_lv_false[0]:.2%}, {ci_lv_false[1]:.2%}] (±{ci_lv_false[2]:.2%})")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")

def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Process evaluation results with monitor model')
    parser.add_argument("--dataset-name", type=str, default="diamonds-seed0", 
                       choices=[f"diamonds-seed{i}" for i in range(8)] + ["function_correctness"],
                       help="Dataset name to determine which reporter prompt to use")
    parser.add_argument('--input', '-i', default="predictions/global_step_40_raw_outputs.json",
                        help='Input file with raw outputs')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file for monitor predictions (default: same directory as input with name monitor_predictions.json)')
    parser.add_argument('--api-key', '-k', help='OpenAI API key (can also use OPENAI_API_KEY env var)')
    parser.add_argument('--batch-size', '-b', type=int, default=20,
                        help='Number of samples to process in each batch (default: 20)')
    parser.add_argument('--max-concurrent', '-c', type=int, default=60,
                        help='Maximum concurrent API requests (default: 60)')
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini",
                       help="Name of the monitor model to use for analysis (default: gpt-4o-mini)")
    parser.add_argument("--rebalance", action="store_true",
                       help="Rebalance samples to achieve 50/50 split of True/False latent variables")
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        # Extract monitor name from model name (e.g., "gpt-4o-mini" -> "gpt-4o-mini")
        monitor_name = args.model_name.replace("-", "_").replace(".", "_")
        args.output = os.path.join(input_dir, f"{monitor_name}_monitor_predictions.json")
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: No OpenAI API key found.")
        print("Please set OPENAI_API_KEY environment variable or use --api-key argument.")
        print(f"Usage: {sys.argv[0]} --api-key YOUR_KEY")
        sys.exit(1)
    
    print(f"\nUsing monitor model: {args.model_name}\n")
    
    # Run the async processing
    asyncio.run(process_evaluation_results_async(
        args.input, 
        args.output, 
        api_key,
        args.batch_size,
        args.max_concurrent,
        args.dataset_name,
        args.model_name,
        args.rebalance,
    ))

if __name__ == "__main__":
    main()