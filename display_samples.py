#!/usr/bin/env python3
"""
Script to display a sample from a JSONL file in a readable format.
"""

import json
import sys
import argparse
from pathlib import Path


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
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if sample_index >= len(lines):
            print(f"Error: Sample index {sample_index} is out of range. File has {len(lines)} lines.")
            return
        
        # Parse the JSON line
        sample = json.loads(lines[sample_index])
        
        print("=" * 80)
        print(f"SAMPLE {sample_index + 1} from {Path(jsonl_file).name}")
        print("=" * 80)
        
        # Display basic metadata
        print(f"Step: {sample.get('step', 'N/A')}")
        print(f"Score: {sample.get('score', 'N/A')}")
        print(f"Reward: {sample.get('reward', 'N/A')}")
        print(f"Correctness Reward: {sample.get('correctness_reward', 'N/A')}")
        print(f"Format Reward: {sample.get('format_reward', 'N/A')}")
        print(f"Proportion Correct: {sample.get('proportion_correct', 'N/A')}")
        print(f"All Correct: {sample.get('all_correct', 'N/A')}")
        print()
        
        # Display input (truncated if too long)
        input_text = sample.get('input', '')
        if max_input_length is not None and len(input_text) > max_input_length:
            input_text = input_text[:max_input_length] + "... [TRUNCATED]"
        
        print("INPUT:")
        print("-" * 40)
        print(input_text)
        print()
        
        # Display output (truncated if too long)
        output_text = sample.get('output', '')
        if max_output_length is not None and len(output_text) > max_output_length:
            output_text = output_text[:max_output_length] + "... [TRUNCATED]"
        
        print("OUTPUT:")
        print("-" * 40)
        print(output_text)
        print()
        
        # Display ground truth and predictions
        gts = sample.get('gts', 'N/A')
        print(f"GROUND TRUTH: {gts}")
        
        # Try to extract measurements from output
        measurements = "N/A"
        if '<measurements>' in output_text and '</measurements>' in output_text:
            start = output_text.find('<measurements>') + len('<measurements>')
            end = output_text.find('</measurements>')
            measurements = output_text[start:end].strip()
        
        print(f"PREDICTIONS: {measurements}")
        print()
        
        # Display all other fields
        print("OTHER FIELDS:")
        print("-" * 40)
        for key, value in sample.items():
            if key not in ['input', 'output', 'step', 'score', 'reward', 'correctness_reward', 
                          'format_reward', 'proportion_correct', 'all_correct', 'gts']:
                print(f"{key}: {value}")
        
        print("=" * 80)
        
    except FileNotFoundError:
        print(f"Error: File '{jsonl_file}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON on line {sample_index + 1}: {e}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Display a sample from a JSONL file in a readable format')
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
            with open(args.jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"Found {len(lines)} samples in {Path(args.jsonl_file).name}")
            print("=" * 80)
            
            for i, line in enumerate(lines[:10]):  # Show first 10 samples
                try:
                    sample = json.loads(line)
                    step = sample.get('step', 'N/A')
                    score = sample.get('score', 'N/A')
                    proportion_correct = sample.get('proportion_correct', 'N/A')
                    print(f"Sample {i:3d}: Step={step:4s} Score={score:6s} Correct={proportion_correct:6s}")
                except json.JSONDecodeError:
                    print(f"Sample {i:3d}: [Invalid JSON]")
            
            if len(lines) > 10:
                print(f"... and {len(lines) - 10} more samples")
                
        except FileNotFoundError:
            print(f"Error: File '{args.jsonl_file}' not found.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Display mode
        display_sample(args.jsonl_file, args.index, args.max_input, args.max_output)


if __name__ == "__main__":
    main()
