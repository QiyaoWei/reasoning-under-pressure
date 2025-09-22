#!/usr/bin/env python3
"""
Script to display a sample from a JSONL file in a readable format.
"""

import json
import sys
import argparse
from pathlib import Path


def load_samples(json_path):
    """
    Load samples from a file that may be JSONL (one JSON per line) or a JSON array.
    Returns a list of dict samples.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read()

    stripped = content.lstrip()
    # JSON array file (e.g., *_monitor_predictions.json)
    if stripped.startswith('['):
        data = json.loads(content)
        # Ensure list of dicts
        if isinstance(data, dict):
            return [data]
        return list(data)

    # Single JSON object
    if stripped.startswith('{'):
        return [json.loads(content)]

    # Fallback: JSONL
    lines = [line for line in content.splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def display_sample(jsonl_file, sample_index=0, max_input_length=None, max_output_length=None):
    """
    Display a sample from a JSONL file in a readable format.
    
    Args:
        jsonl_file: Path to the JSONL file
        sample_index: Index of the sample to display (0-based)
        max_input_length: Maximum length of input to display (None for no limit)
        max_output_length: Maximum length of output to display (None for no limit)
    """
    try:
        samples = load_samples(jsonl_file)
        if sample_index >= len(samples):
            print(f"Error: Sample index {sample_index} is out of range. File has {len(samples)} samples.")
            return
        sample = samples[sample_index]
        
        print("=" * 80)
        print(f"SAMPLE {sample_index + 1} from {Path(jsonl_file).name}")
        print("=" * 80)
        
        # Display basic metadata
        print(f"Step: {sample.get('step', 'N/A')}")
        print(f"Sample Idx: {sample.get('sample_idx', 'N/A')}")
        print(f"Score: {sample.get('score', 'N/A')}")
        print(f"Reward: {sample.get('reward', 'N/A')}")
        print(f"Correctness Reward: {sample.get('correctness_reward', 'N/A')}")
        print(f"Format Reward: {sample.get('format_reward', 'N/A')}")
        print(f"Proportion Correct: {sample.get('proportion_correct', 'N/A')}")
        print(f"All Correct: {sample.get('all_correct', 'N/A')}")
        print()
        
        # Display input (truncated if too long)
        input_text = sample.get('input') or sample.get('prompt')
        if input_text is not None:
            if max_input_length is not None and len(input_text) > max_input_length:
                input_text = input_text[:max_input_length] + "... [TRUNCATED]"
            print("INPUT:")
            print("-" * 40)
            print(input_text)
            print()
        
        # Display output (truncated if too long)
        output_text = (
            sample.get('output')
            or sample.get('original_response')
            or sample.get('monitor_full_response')
            or ''
        )
        if output_text:
            if max_output_length is not None and len(output_text) > max_output_length:
                output_text = output_text[:max_output_length] + "... [TRUNCATED]"
            print("OUTPUT:")
            print("-" * 40)
            print(output_text)
            print()
        
        # Display ground truth and predictions
        gts = sample.get('gts') if 'gts' in sample else sample.get('ground_truth', 'N/A')
        print(f"GROUND TRUTH: {gts}")
        
        # Try to extract measurements from output
        measurements = "N/A"
        if output_text and '<measurements>' in output_text and '</measurements>' in output_text:
            start = output_text.find('<measurements>') + len('<measurements>')
            end = output_text.find('</measurements>')
            measurements = output_text[start:end].strip()
        elif 'original_prediction' in sample:
            measurements = sample.get('original_prediction')
        elif 'monitor_prediction' in sample:
            measurements = sample.get('monitor_prediction')
        
        print(f"PREDICTIONS: {measurements}")
        print()
        
        # Display all other fields
        print("OTHER FIELDS:")
        print("-" * 40)
        for key, value in sample.items():
            if key not in [
                'input', 'prompt', 'output', 'original_response', 'monitor_full_response',
                'step', 'score', 'reward', 'correctness_reward', 'format_reward',
                'proportion_correct', 'all_correct', 'gts', 'ground_truth',
                'original_prediction', 'monitor_prediction', 'sample_idx'
            ]:
                print(f"{key}: {value}")
        
        print("=" * 80)
        
    except FileNotFoundError:
        print(f"Error: File '{jsonl_file}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON on line {sample_index + 1}: {e}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Display a sample from a JSONL or JSON array file in a readable format')
    parser.add_argument('jsonl_file', help='Path to the JSONL file')
    parser.add_argument('-i', '--index', type=int, default=0, 
                       help='Index of the sample to display (0-based, default: 0)')
    parser.add_argument('--max-input', type=int, default=None,
                       help='Maximum length of input to display (default: no limit)')
    parser.add_argument('--max-output', type=int, default=None,
                       help='Maximum length of output to display (default: no limit)')
    parser.add_argument('--list', action='store_true',
                       help='List available samples and their basic info')
    
    args = parser.parse_args()
    
    if args.list:
        # List mode - show basic info for all samples
        try:
            samples = load_samples(args.jsonl_file)
            print(f"Found {len(samples)} samples in {Path(args.jsonl_file).name}")
            print("=" * 80)

            for i, sample in enumerate(samples[:10]):  # Show first 10 samples
                try:
                    if 'step' in sample:
                        step = str(sample.get('step', 'N/A'))
                        score = str(sample.get('score', 'N/A'))
                        proportion_correct = str(sample.get('proportion_correct', 'N/A'))
                        print(f"Sample {i:3d}: Step={step:4s} Score={score:6s} Correct={proportion_correct:6s}")
                    elif 'sample_idx' in sample or 'monitor_prediction' in sample:
                        sample_idx = str(sample.get('sample_idx', 'N/A'))
                        monitor_pred = str(sample.get('monitor_prediction', 'N/A'))
                        print(f"Sample {i:3d}: sample_idx={sample_idx:6s} monitor_pred={monitor_pred:5s}")
                    else:
                        keys = list(sample.keys())
                        preview = ', '.join(keys[:3])
                        print(f"Sample {i:3d}: keys=[{preview}]")
                except Exception:
                    print(f"Sample {i:3d}: [Invalid sample]")

            if len(samples) > 10:
                print(f"... and {len(samples) - 10} more samples")
                
        except FileNotFoundError:
            print(f"Error: File '{args.jsonl_file}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Display mode
        display_sample(args.jsonl_file, args.index, args.max_input, args.max_output)


if __name__ == "__main__":
    main()
