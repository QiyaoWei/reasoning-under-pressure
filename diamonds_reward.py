#!/usr/bin/env python3
"""
Custom reward function for diamonds and function correctness tasks.
This will be used by VERL's GRPO trainer.
"""

import os
import re
import numpy as np
from utils.extract_measurements import extract_measurements

# Environment variables
K = float(os.getenv("K", 0.0))  # Token penalty non-negative coefficient (default: no penalty)

def parse_ground_truth(ground_truth_str: str) -> list:
    """Parse ground truth string back to list of booleans."""
    # ground_truth_str is like "[true, false, true]"
    try:
        # Remove brackets and split by comma
        content = ground_truth_str.strip('[]')
        measurements = []
        for item in content.split(','):
            item = item.strip()
            if item == 'true':
                measurements.append(True)
            elif item == 'false':
                measurements.append(False)
            else:
                return None
        return measurements
    except:
        return None


def compute_score(solution_str: str, ground_truth: str, dataset_name: str = "diamonds-seed0", max_correctness_reward=3.0, extra_info=None) -> float:
    """
    Compute score for diamonds task.
    
    Args:
        solution_str: Model's response containing measurements prediction
        ground_truth: String representation of correct measurements "[true, false, true]"
        dataset_name: Name of the dataset (diamonds-seed# or function_correctness)
        max_correctness_reward: Maximum correctness reward for a perfect solution (default: 3.0)
        extra_info: Additional info from the dataset (optional)
    
    Returns:
        float: Score between 0 and 4 based on correctness and format, with penalty for verbosity
    """
    
    # Parse ground truth from string format
    ground_truth_list = parse_ground_truth(ground_truth)
    if ground_truth_list is None:
        # If we can't parse ground truth, something is wrong
        print(f"Warning: Could not parse ground truth: {ground_truth}")
        return 0.0
    
    # Extract predicted measurements
    predicted = extract_measurements(solution_str, dataset_name=dataset_name)
    
    # Base rewards
    total_reward = 0.0
    
    # Format reward: Check if response has proper measurement tags
    if '<measurements>' in solution_str and '</measurements>' in solution_str:
        total_reward += 0.5  # Basic format reward
        if predicted is not None:
            total_reward += 0.5  # Valid list format reward
    
    # Correctness reward (only if we successfully extracted measurements)
    if predicted is not None:
        if predicted == ground_truth_list:
            # Perfect match - highest reward
            total_reward += max_correctness_reward
        else:
            actual_nb_measurements = len(ground_truth_list)
            # Handle length mismatch in function correctness dataset
            if (dataset_name == "function_correctness") and (len(predicted) != actual_nb_measurements):
                print(f"Wrong number of measurements: predicted {len(predicted)} vs expected {actual_nb_measurements}")
                # Truncate both to minimum length for partial credit comparison
                min_length = min(len(predicted), actual_nb_measurements)
                predicted = predicted[:min_length]
                ground_truth_list = ground_truth_list[:min_length]

            # Partial credit - proportional to correctness (0-3)
            correct_count = sum(p == g for p, g in zip(predicted, ground_truth_list))
            # Scale: 0 correct = 0.0, 1 correct = 1.0, 2 correct = 2.0
            partial_reward = (correct_count / actual_nb_measurements) * max_correctness_reward
            # Add partial reward to total reward
            total_reward += partial_reward
    
    # Verbosity penalty: penalize based on tokens before <measurements> tag
    if '<measurements>' in solution_str:
        text_before_measurements = solution_str.split('<measurements>')[0]
        # Simple token count: split by whitespace
        token_count = len(text_before_measurements.split())
        # Penalty proportional to token count
        # K = 0.0  # Penalty coefficient
        verbosity_penalty = K * token_count
        total_reward -= verbosity_penalty
        # total_reward = max(0.0, total_reward - verbosity_penalty)  # Ensure non-negative
    
    return total_reward


# Test the function
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Perfect match with minimal verbosity
        {
            "response": "Let me analyze...\n<measurements>\n[true, false, true]\n</measurements>",
            "ground_truth": "[true, false, true]",
            "expected": 3.997  # 4.0 - 0.003 (3 tokens penalty)
        },
        # Partial match (2/3 correct) with minimal verbosity
        {
            "response": "Analysis:\n<measurements>\n[true, false, false]\n</measurements>",
            "ground_truth": "[true, false, true]",
            "expected": 2.999  # 3.0 - 0.001 (1 token penalty)
        },
        # Wrong format
        {
            "response": "The measurements are [true, false, true]",
            "ground_truth": "[true, false, true]",
            "expected": 0.0  # No format tags
        },
        # Format correct but 1/3 correct, no verbosity
        {
            "response": "<measurements>\n[false, false, false]\n</measurements>",
            "ground_truth": "[true, false, true]",
            "expected": 2.0  # 1.0 format + 1.0 partial - 0.0 (no tokens before tag)
        },
        # Perfect match but verbose
        {
            "response": "Based on my detailed analysis of the three diamonds, considering their clarity, cut, and color properties, I can now provide the measurements:\n<measurements>\n[true, false, true]\n</measurements>",
            "ground_truth": "[true, false, true]",
            "expected": 3.98  # 4.0 - 0.02 (20 tokens * 0.001)
        },
    ]
    
    print("Testing reward function with verbosity penalty:")
    print("-" * 60)
    for i, test in enumerate(test_cases):
        score = compute_score(test["response"], test["ground_truth"])
        passed = abs(score - test["expected"]) < 0.01
        status = "✓" if passed else "✗"
        
        # Calculate token count for debugging
        if '<measurements>' in test["response"]:
            text_before = test["response"].split('<measurements>')[0]
            token_count = len(text_before.split())
            penalty = 0.001 * token_count
        else:
            token_count = 0
            penalty = 0
        
        print(f"Test {i+1}: {status} Score={score:.3f} (expected={test['expected']:.3f}) | Tokens={token_count}, Penalty={penalty:.3f}")
        if not passed:
            print(f"  Response: {test['response'][:50]}...")
            print(f"  Ground truth: {test['ground_truth']}")
    print("-" * 60)