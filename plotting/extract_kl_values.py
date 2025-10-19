#!/usr/bin/env python3
import subprocess
import re
import os
import json
from pathlib import Path
import argparse
import time

def find_checkpoints(base_dir):
    """Find all global_step_* checkpoint directories"""
    checkpoints = []
    for path in Path(base_dir).rglob("global_step_*"):
        if path.is_dir():
            checkpoints.append(path)
    return sorted(checkpoints, key=lambda x: int(x.name.split('_')[-1]))

def run_training_step(checkpoint_path, dataset="val_for_train", timeout=300):
    """Run a single training step with lr=0 to extract KL value"""
    cmd = [
        "./run_grpo.sh",
        "--resume-from-path", str(checkpoint_path),
        "--dataset", dataset,
        "--lr", "0",
        "--kl-coef", "1.0",
        "--experiment-name", f"kl_extract_{checkpoint_path.name}"
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Search for the KL loss value in the output
        # Looking for pattern: actor/kl_loss:0.0015064716204733486
        kl_pattern = r"actor/kl_loss:([\d.e-]+)"
        kl_std_pattern = r"actor/kl_loss_std:([\d.e-]+)"

        # Search in both stdout and stderr
        output = result.stdout + result.stderr

        # Find the first occurrence (first training step)
        kl_match = re.search(kl_pattern, output)
        kl_std_match = re.search(kl_std_pattern, output)

        if kl_match:
            kl_value = float(kl_match.group(1))
            kl_std_value = float(kl_std_match.group(1)) if kl_std_match else None
            return kl_value, kl_std_value
        else:
            print(f"Warning: No KL value found for {checkpoint_path}")
            return None, None

    except subprocess.TimeoutExpired:
        print(f"Error: Timeout for {checkpoint_path}")
        return None, None
    except Exception as e:
        print(f"Error processing {checkpoint_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Extract PPO KL values from checkpoints")
    parser.add_argument("--base-dir", default="reward_kl_1e-3",
                       help="Base directory containing checkpoints")
    parser.add_argument("--dataset", default="val_for_train",
                       help="Dataset to use for evaluation")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in seconds for each checkpoint evaluation")
    parser.add_argument("--output", default="kl_values.json",
                       help="Output file for results")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of checkpoints to process (for testing)")

    args = parser.parse_args()

    # Find all checkpoints
    print(f"Searching for checkpoints in {args.base_dir}...")
    checkpoints = find_checkpoints(args.base_dir)

    if args.limit:
        checkpoints = checkpoints[:args.limit]

    print(f"Found {len(checkpoints)} checkpoints")

    # Process each checkpoint
    results = {}
    kl_values = []

    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] Processing {checkpoint}")

        # Extract step number from checkpoint name
        step_num = int(checkpoint.name.split('_')[-1])

        # Run training step and extract KL
        kl_value, kl_std_value = run_training_step(checkpoint, args.dataset, args.timeout)

        if kl_value is not None:
            results[str(checkpoint)] = {
                "step": step_num,
                "kl_value": kl_value,
                "kl_std_value": kl_std_value
            }
            kl_values.append(kl_value)

            # Print current statistics
            if kl_values:
                print(f"  KL value: {kl_value:.6e}")
                if kl_std_value is not None:
                    print(f"  KL std value: {kl_std_value:.6e}")
                print(f"  Current min KL: {min(kl_values):.6e}")
                print(f"  Current max KL: {max(kl_values):.6e}")
                print(f"  Current mean KL: {sum(kl_values)/len(kl_values):.6e}")

        # Add a small delay to avoid overwhelming the system
        time.sleep(1)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    if kl_values:
        min_kl = min(kl_values)
        max_kl = max(kl_values)
        mean_kl = sum(kl_values) / len(kl_values)

        # Find which checkpoints have min/max values
        min_checkpoint = [k for k, v in results.items() if v['kl_value'] == min_kl][0]
        max_checkpoint = [k for k, v in results.items() if v['kl_value'] == max_kl][0]

        print(f"Processed {len(kl_values)} checkpoints successfully")
        print(f"Minimum KL: {min_kl:.6e} at {min_checkpoint}")
        print(f"Maximum KL: {max_kl:.6e} at {max_checkpoint}")
        print(f"Mean KL: {mean_kl:.6e}")
        print(f"Range: {max_kl - min_kl:.6e}")
        print(f"\nResults saved to {args.output}")
    else:
        print("No KL values were successfully extracted")

if __name__ == "__main__":
    main()
