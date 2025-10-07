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

Confidence intervals use the Wilson score interval with the same inputs as run_inference.py:
- measurement-wise CI: successes = sum of per-sample proportions, total = number of samples
- others: successes = counts, total = number of samples

Usage:
  python reasoner_performance_analysis.py \
    --predictions-file results/regular_RL/step_200/predictions/func-corr-deepseek-regularRL-global_step_200_predictions.json

This will update the sibling summary file in the same directory:
  results/regular_RL/step_200/predictions/reasoner_accuracy_summary.json
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

from datetime import datetime

try:
    from scipy import stats
except Exception as _:
    stats = None


def calculate_confidence_interval(successes: float, total: int, confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """Wilson score interval for a proportion.

    Mirrors run_inference.py, using scipy.stats.norm.ppf when available; falls back to z=1.96.
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    alpha = 1 - confidence_level
    if stats is not None:
        z_score = stats.norm.ppf(1 - alpha / 2)
    else:
        # 95% fallback
        z_score = 1.959963984540054

    n = float(total)
    p_hat = float(successes) / n

    denominator = 1.0 + (z_score * z_score / n)
    center = (p_hat + (z_score * z_score / (2.0 * n))) / denominator
    margin = (z_score / denominator) * ((p_hat * (1.0 - p_hat) / n) + (z_score * z_score / (4.0 * n * n))) ** 0.5

    lower_bound = max(0.0, center - margin)
    upper_bound = min(1.0, center + margin)
    return lower_bound, upper_bound, margin


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
    correct_count = 0
    format_correct_count = 0

    for s in samples:
        total += 1
        proportion_sum += float(s.get('proportion_correct', 0.0))
        if s.get('is_correct', False):
            correct_count += 1
        if s.get('correct_format', False):
            format_correct_count += 1

    # Metrics
    measurement_wise_acc = (proportion_sum / total) if total > 0 else 0.0
    all_correct_acc = (correct_count / total) if total > 0 else 0.0
    format_acc = (format_correct_count / total) if total > 0 else 0.0

    # CIs
    mw_lower, mw_upper, mw_margin = calculate_confidence_interval(proportion_sum, total)
    ac_lower, ac_upper, ac_margin = calculate_confidence_interval(correct_count, total)
    fm_lower, fm_upper, fm_margin = calculate_confidence_interval(format_correct_count, total)

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

    print("Appended group metrics to:", summary_path)


if __name__ == '__main__':
    main()


