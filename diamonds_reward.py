#!/usr/bin/env python3
"""
Custom reward function for diamonds task.
This will be used by VERL's GRPO trainer.
"""

import re
import numpy as np


def extract_measurements(text: str) -> list:
    """Extracts the measurements list from model output."""
    try:
        match = re.search(r'<measurements>\s*\[(.*?)\]\s*</measurements>', text, re.DOTALL)
        if match:
            list_content = match.group(1).strip()
            measurements = []
            for item in list_content.split(','):
                item = item.strip().lower()
                if item == 'true':
                    measurements.append(True)
                elif item == 'false':
                    measurements.append(False)
                else:
                    return None
            if len(measurements) == 3:
                return measurements
            else:
                return None
        else:
            return None
    except Exception as e:
        return None


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


def compute_score(solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute score for diamonds task.
    
    Args:
        solution_str: Model's response containing measurements prediction
        ground_truth: String representation of correct measurements "[true, false, true]"
        extra_info: Additional info from the dataset (optional)
    
    Returns:
        float: Score between 0 and 4 based on correctness and format
    """
    
    # Parse ground truth from string format
    ground_truth_list = parse_ground_truth(ground_truth)
    if ground_truth_list is None:
        # If we can't parse ground truth, something is wrong
        print(f"Warning: Could not parse ground truth: {ground_truth}")
        return 0.0
    
    # Extract predicted measurements
    predicted = extract_measurements(solution_str)
    
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
            total_reward += 3.0
        else:
            # Partial credit - proportional to correctness
            correct_count = sum(p == g for p, g in zip(predicted, ground_truth_list))
            # Scale: 0 correct = 0.0, 1 correct = 1.0, 2 correct = 2.0
            partial_reward = correct_count * 1.0
            total_reward += partial_reward
    
    return total_reward


# Test the function
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Perfect match
        {
            "response": "Let me analyze...\n<measurements>\n[true, false, true]\n</measurements>",
            "ground_truth": "[true, false, true]",
            "expected": 4.0  # 1.0 format + 3.0 perfect match
        },
        # Partial match (2/3 correct)
        {
            "response": "Analysis:\n<measurements>\n[true, false, false]\n</measurements>",
            "ground_truth": "[true, false, true]",
            "expected": 3.0  # 1.0 format + 2.0 partial
        },
        # Wrong format
        {
            "response": "The measurements are [true, false, true]",
            "ground_truth": "[true, false, true]",
            "expected": 0.0  # No format tags
        },
        # Format correct but 1/3 correct
        {
            "response": "<measurements>\n[false, false, false]\n</measurements>",
            "ground_truth": "[true, false, true]",
            "expected": 2.0  # 1.0 format + 1.0 partial (1/3 correct)
        },
    ]
    
    print("Testing reward function:")
    print("-" * 60)
    for i, test in enumerate(test_cases):
        score = compute_score(test["response"], test["ground_truth"])
        passed = abs(score - test["expected"]) < 0.01
        status = "✓" if passed else "✗"
        print(f"Test {i+1}: {status} Score={score:.1f} (expected={test['expected']:.1f})")
        if not passed:
            print(f"  Response: {test['response'][:50]}...")
            print(f"  Ground truth: {test['ground_truth']}")
    print("-" * 60)