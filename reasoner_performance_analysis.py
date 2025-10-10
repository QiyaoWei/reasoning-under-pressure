#!/usr/bin/env python3
"""
Compute additional group-wise metrics from a predictions JSON and append them
to the existing reasoner summary JSON. Groups (based on ground-truth measurement pattern and latent variable):

- all_false_lv_false: all measurements False AND latent_variable == False
- mixed_lv_false: mixture of True and False measurements AND latent_variable == False
- all_true_lv_true: all measurements True AND latent_variable == True
- all_true_lv_false: all measurements True AND latent_variable == False

Accuracies are computed the same way as in run_inference.py:
- measurement-wise_accuracy: mean of per-sample proportion_correct
- all_correct_accuracy: fraction of samples with is_correct == True
- format_accuracy: fraction of samples with correct_format == True

Confidence intervals:
- measurement-wise CI: t-interval over per-sample proportions (mean with unbiased variance)
- others: Wilson score for binomial proportions (successes/total)

Usage:
  python reasoner_performance_analysis.py \
    --predictions-file results/regular_RL/step_200/predictions/func-corr-deepseek-regularRL-global_step_200_predictions.json

This will update the sibling summary file in the same directory:
  results/regular_RL/step_200/predictions/reasoner_accuracy_summary.json
"""

import os
import json
import argparse
from typing import Dict, Any, List

from datetime import datetime
from utils.stats import (
    calculate_wilson_confidence_interval,
    calculate_t_confidence_interval,
)


def is_all_false(values: List[bool]) -> bool:
    return len(values) > 0 and all(v is False for v in values)


def is_all_true(values: List[bool]) -> bool:
    return len(values) > 0 and all(v is True for v in values)


def is_mixed(values: List[bool]) -> bool:
    if not values:
        return False
    has_true = any(values)
    has_false = any(v is False for v in values)
    return has_true and has_false


def aggregate_group_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate counters required to compute metrics and CIs."""
    total = 0
    proportion_sum = 0.0
    proportion_sq_sum = 0.0
    correct_count = 0
    format_correct_count = 0

    for s in samples:
        total += 1
        pc = float(s.get('proportion_correct', 0.0))
        proportion_sum += pc
        proportion_sq_sum += (pc * pc)
        if s.get('is_correct', False):
            correct_count += 1
        if s.get('correct_format', False):
            format_correct_count += 1

    # Metrics
    measurement_wise_acc = (proportion_sum / total) if total > 0 else 0.0
    all_correct_acc = (correct_count / total) if total > 0 else 0.0
    format_acc = (format_correct_count / total) if total > 0 else 0.0

    # CIs
    if total > 0:
        if total > 1:
            sample_variance = (proportion_sq_sum - total * (measurement_wise_acc ** 2)) / (total - 1)
        else:
            sample_variance = 0.0
        mw_lower, mw_upper, mw_margin = calculate_t_confidence_interval(measurement_wise_acc, sample_variance, total)
    else:
        mw_lower, mw_upper, mw_margin = (0.0, 0.0, 0.0)
    ac_lower, ac_upper, ac_margin = calculate_wilson_confidence_interval(correct_count, total)
    fm_lower, fm_upper, fm_margin = calculate_wilson_confidence_interval(format_correct_count, total)

    return {
        'n_samples': total,
        'measurement-wise_accuracy': {
            'value': measurement_wise_acc,
            'confidence_interval': [mw_lower, mw_upper, mw_margin],
        },
        'all_correct_accuracy': {
            'value': all_correct_acc,
            'confidence_interval': [ac_lower, ac_upper, ac_margin],
            'correct_count': correct_count,
        },
        'format_accuracy': {
            'value': format_acc,
            'confidence_interval': [fm_lower, fm_upper, fm_margin],
            'format_correct_count': format_correct_count,
        },
    }


def compute_measurementwise_t_ci_by_difficulty(predictions: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Compute t-interval CIs for measurement-wise accuracy overall and by difficulty.

    Returns a map: suffix -> [lower, upper, margin], where suffix is 'overall' or the difficulty value.
    """
    # NOTE: This function is used to correct earlier summaries that used Wilson CI for measurement-wise accuracy.
    #       It can be removed later once all summaries are regenerated with the correct t-interval.

    accumulators: Dict[Any, Dict[str, float]] = {}
    # Prepare an 'overall' bucket
    accumulators['overall'] = {'n': 0.0, 'sum': 0.0, 'sumsq': 0.0}

    for p in predictions:
        pc = float(p.get('proportion_correct', 0.0))
        diff = p.get('difficulty', -1)

        # Overall
        accumulators['overall']['n'] += 1.0
        accumulators['overall']['sum'] += pc
        accumulators['overall']['sumsq'] += pc * pc

        # Per-difficulty
        if diff not in accumulators:
            accumulators[diff] = {'n': 0.0, 'sum': 0.0, 'sumsq': 0.0}
        accumulators[diff]['n'] += 1.0
        accumulators[diff]['sum'] += pc
        accumulators[diff]['sumsq'] += pc * pc

    ci_map: Dict[str, List[float]] = {}
    for suffix, acc in accumulators.items():
        n = int(acc['n'])
        if n <= 0:
            ci_map[str(suffix)] = [0.0, 0.0, 0.0]
            continue
        mean_val = acc['sum'] / n
        if n > 1:
            sample_var = (acc['sumsq'] - n * (mean_val ** 2)) / (n - 1)
        else:
            sample_var = 0.0
        lower, upper, margin = calculate_t_confidence_interval(mean_val, sample_var, n)
        ci_map[str(suffix)] = [lower, upper, margin]

    return ci_map


def apply_measurementwise_ci_correction_file(file_path: str, ci_map: Dict[str, List[float]]):
    """Update measurement-wise CI fields in a JSON file (summary or predictions) using provided t-intervals.

    NOTE: Temporary correction to fix earlier files that used Wilson CI for measurement-wise accuracy.
    This can be removed once all artifacts are regenerated with the correct t-interval.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return

    # Determine where measurement-wise keys live
    # - Summary JSON: keys are at top level
    # - Predictions JSON: keys are under metrics
    if 'metrics' in data and isinstance(data['metrics'], dict):
        target_container = data['metrics']
    else:
        target_container = data

    updated = False
    for suffix, ci in ci_map.items():
        keys_to_try = [f'measurement-wise_accuracy_{suffix}']

        for key in keys_to_try:
            if key in target_container and isinstance(target_container[key], dict):
                target_container[key]['confidence_interval'] = ci
                updated = True

    if updated:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

def compute_groups(predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    all_false_lv_false = []
    mixed_lv_false = []
    all_true_lv_true = []
    all_true_lv_false = []

    for p in predictions:
        gt = p.get('ground_truth', []) or []
        lv = p.get('latent_variable', None)

        if is_all_false(gt) and (lv is False):
            all_false_lv_false.append(p)
        elif is_mixed(gt) and (lv is False):
            mixed_lv_false.append(p)
        elif is_all_true(gt) and (lv is True):
            all_true_lv_true.append(p)
        elif is_all_true(gt) and (lv is False):
            all_true_lv_false.append(p)

    return {
        'all_false_lv_false': aggregate_group_metrics(all_false_lv_false),
        'mixed_lv_false': aggregate_group_metrics(mixed_lv_false),
        'all_true_lv_true': aggregate_group_metrics(all_true_lv_true),
        'all_true_lv_false': aggregate_group_metrics(all_true_lv_false),
    }


def append_to_summary(summary_path: str, group_metrics: Dict[str, Dict[str, Any]]):
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Store under nested structure: group_metrics[group_id] = {...}
    if 'group_metrics' not in summary or not isinstance(summary['group_metrics'], dict):
        summary['group_metrics'] = {}
    for group_id, metrics in group_metrics.items():
        mw = metrics['measurement-wise_accuracy']
        ac = metrics['all_correct_accuracy']
        fm = metrics['format_accuracy']
        summary['group_metrics'][group_id] = {
            'measurement-wise_accuracy': mw,
            'all_correct_accuracy': ac,
            'format_accuracy': fm,
            'n_samples': metrics['n_samples'],
        }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Append group-wise metrics to reasoner summary from predictions JSON")
    parser.add_argument('--predictions-file', type=str, required=True,
                        help='Path to predictions JSON (contains {metrics, predictions})')
    parser.add_argument('--summary-file', type=str, default=None,
                        help='Optional explicit path to reasoner_accuracy_summary.json; default is sibling to predictions-file')

    args = parser.parse_args()

    predictions_file = os.path.abspath(args.predictions_file)
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    with open(predictions_file, 'r') as f:
        data = json.load(f)

    if 'predictions' not in data or not isinstance(data['predictions'], list):
        raise ValueError('Invalid predictions file: missing "predictions" list')

    predictions = data['predictions']

    # Where to write the summary
    if args.summary_file is not None:
        summary_path = os.path.abspath(args.summary_file)
    else:
        pred_dir = os.path.dirname(predictions_file)
        summary_path = os.path.join(pred_dir, 'reasoner_accuracy_summary.json')

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    # Compute and append
    group_metrics = compute_groups(predictions)
    append_to_summary(summary_path, group_metrics)

    # --- Temporary correction block ---
    # NOTE: This block corrects earlier summaries where measurement-wise CI used Wilson instead of t-interval.
    #       It recomputes t-intervals from the per-sample proportions in predictions and updates the summary.
    #       This code can be removed once all summaries are regenerated correctly.
    ci_map = compute_measurementwise_t_ci_by_difficulty(predictions)
    apply_measurementwise_ci_correction_file(summary_path, ci_map)
    # Also correct the predictions JSON metrics for measurement-wise CI.
    # NOTE: Temporary correction; can be removed once predictions are regenerated correctly.
    apply_measurementwise_ci_correction_file(predictions_file, ci_map)

    print("Appended group metrics to:", summary_path)


if __name__ == '__main__':
    main()


