#!/usr/bin/env python3
"""
Statistical significance testing for comparing experimental runs against baseline.

Uses McNemar's test for paired binary outcomes (monitor correct/incorrect).
Includes multiple comparison corrections and effect size reporting.

Usage:
  python statistical_significance_test.py plotting/diamond/main/config.yaml

Features:
- McNemar's test (exact when needed)
- Multiple comparison corrections (Bonferroni, Holm-Bonferroni, FDR)
- Sample alignment verification
- Effect size reporting
- Conditional accuracy testing (optional)
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
from scipy.stats import chi2, binom
from statsmodels.stats.multitest import multipletests


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_monitor_predictions(predictions_path: str) -> Dict[int, Dict[str, Any]]:
    """Load monitor predictions and index by sample_idx."""
    with open(predictions_path, 'r') as f:
        data = json.load(f)

    # Index by sample_idx
    indexed = {}
    for sample in data:
        if 'sample_idx' in sample:
            idx = int(sample['sample_idx'])
            indexed[idx] = sample

    return indexed


def mcnemar_test(b: int, c: int, exact: bool = None) -> Tuple[float, float, str]:
    """
    McNemar's test for paired binary data.

    Args:
        b: Count where method1 correct, method2 incorrect
        c: Count where method1 incorrect, method2 correct
        exact: If True, use exact binomial test. If None, auto-decide based on sample size.

    Returns:
        statistic, p_value, test_type
    """
    n_discordant = b + c

    # Decide whether to use exact test
    if exact is None:
        exact = n_discordant < 25  # Use exact test for small samples

    if exact:
        # Exact binomial test
        # Under null hypothesis, b ~ Binomial(n_discordant, 0.5)
        p_value = 2 * min(
            binom.cdf(min(b, c), n_discordant, 0.5),
            1 - binom.cdf(max(b, c) - 1, n_discordant, 0.5)
        )
        statistic = min(b, c)
        test_type = "exact"
    else:
        # Chi-square approximation with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
        p_value = 1 - chi2.cdf(statistic, df=1)
        test_type = "chi-square"

    return statistic, p_value, test_type


def compare_methods(
    baseline_preds: Dict[int, Dict[str, Any]],
    method_preds: Dict[int, Dict[str, Any]],
    metric_key: str = 'monitor_correct'
) -> Dict[str, Any]:
    """
    Compare two methods using McNemar's test.

    Args:
        baseline_preds: Baseline predictions indexed by sample_idx
        method_preds: Method predictions indexed by sample_idx
        metric_key: Key to use for correctness ('monitor_correct' or custom)

    Returns:
        Dictionary with test results and alignment info
    """
    # Find common samples
    baseline_indices = set(baseline_preds.keys())
    method_indices = set(method_preds.keys())
    common_indices = baseline_indices & method_indices

    if len(common_indices) == 0:
        return {
            'error': 'No common samples found',
            'baseline_samples': len(baseline_indices),
            'method_samples': len(method_indices),
        }

    # Build contingency table
    both_correct = 0      # a
    method_only = 0       # b (method correct, baseline incorrect)
    baseline_only = 0     # c (baseline correct, method incorrect)
    both_incorrect = 0    # d

    baseline_correct_count = 0
    method_correct_count = 0

    for idx in sorted(common_indices):
        baseline_sample = baseline_preds[idx]
        method_sample = method_preds[idx]

        # Get correctness - compute if not present
        if metric_key in baseline_sample:
            baseline_correct = baseline_sample[metric_key]
        else:
            # Compute from monitor_prediction and latent_variable
            baseline_correct = (baseline_sample.get('monitor_prediction') ==
                              baseline_sample.get('latent_variable'))

        if metric_key in method_sample:
            method_correct = method_sample[metric_key]
        else:
            # Compute from monitor_prediction and latent_variable
            method_correct = (method_sample.get('monitor_prediction') ==
                            method_sample.get('latent_variable'))

        if baseline_correct:
            baseline_correct_count += 1
        if method_correct:
            method_correct_count += 1

        # Update contingency table
        if baseline_correct and method_correct:
            both_correct += 1
        elif method_correct and not baseline_correct:
            method_only += 1
        elif baseline_correct and not method_correct:
            baseline_only += 1
        else:
            both_incorrect += 1

    # Calculate accuracies
    n_total = len(common_indices)
    baseline_acc = baseline_correct_count / n_total
    method_acc = method_correct_count / n_total
    effect_size = method_acc - baseline_acc

    # Run McNemar's test
    statistic, p_value, test_type = mcnemar_test(method_only, baseline_only)

    return {
        'n_total_samples': n_total,
        'n_common_samples': len(common_indices),
        'baseline_only_samples': len(baseline_indices - method_indices),
        'method_only_samples': len(method_indices - baseline_indices),
        'contingency_table': {
            'both_correct': both_correct,
            'method_correct_baseline_incorrect': method_only,
            'baseline_correct_method_incorrect': baseline_only,
            'both_incorrect': both_incorrect,
        },
        'baseline_accuracy': baseline_acc,
        'method_accuracy': method_acc,
        'effect_size': effect_size,
        'effect_size_percentage': effect_size * 100,
        'mcnemar_statistic': statistic,
        'p_value': p_value,
        'test_type': test_type,
        'discordant_pairs': method_only + baseline_only,
    }


def run_significance_tests(
    config_path: Path,
    monitor_names: List[str] = None,
    alpha: float = 0.05,
    test_conditionals: bool = False
) -> Dict[str, Any]:
    """
    Run significance tests for all methods in config against baseline.

    Args:
        config_path: Path to YAML config file
        monitor_names: List of monitor names to test (default: ['gpt-4o-mini', 'gpt-4o'])
        alpha: Significance level (default: 0.05)
        test_conditionals: Whether to test conditional accuracies

    Returns:
        Dictionary with all test results and corrections
    """
    config = load_config(config_path)

    if monitor_names is None:
        monitor_names = ['gpt-4o-mini', 'gpt-4o']

    # Load baseline data
    baseline_data = config.get('baseline')
    if not baseline_data:
        raise ValueError("No baseline found in config")

    # Load baseline predictions for each monitor
    baseline_monitors = {}
    for monitor in baseline_data['monitors']:
        if monitor['name'] in monitor_names:
            pred_path = monitor['predictions_path']
            # Paths in config are relative to project root, not config file
            if not Path(pred_path).is_absolute():
                pred_path = str(Path.cwd() / pred_path)
            baseline_monitors[monitor['name']] = load_monitor_predictions(pred_path)

    # Run tests for each experimental run
    runs = config.get('runs', [])
    all_results = []

    for run in runs:
        run_name = run['name']
        run_results = {
            'run_name': run_name,
            'monitors': {}
        }

        for monitor in run['monitors']:
            monitor_name = monitor['name']
            if monitor_name not in monitor_names:
                continue

            pred_path = monitor['predictions_path']
            # Paths in config are relative to project root, not config file
            if not Path(pred_path).is_absolute():
                pred_path = str(Path.cwd() / pred_path)
            method_preds = load_monitor_predictions(pred_path)
            baseline_preds = baseline_monitors[monitor_name]

            # Overall accuracy test
            result = compare_methods(baseline_preds, method_preds)

            monitor_result = {
                'overall': result
            }

            # Conditional accuracy tests (optional)
            if test_conditionals:
                conditionals = {}
                # Test on subset where latent_variable is True
                baseline_lv_true = {k: v for k, v in baseline_preds.items()
                                   if v.get('latent_variable') == True}
                method_lv_true = {k: v for k, v in method_preds.items()
                                 if v.get('latent_variable') == True}
                if baseline_lv_true and method_lv_true:
                    conditionals['when_lv_true'] = compare_methods(
                        baseline_lv_true, method_lv_true
                    )

                # Test on subset where latent_variable is False
                baseline_lv_false = {k: v for k, v in baseline_preds.items()
                                    if v.get('latent_variable') == False}
                method_lv_false = {k: v for k, v in method_preds.items()
                                  if v.get('latent_variable') == False}
                if baseline_lv_false and method_lv_false:
                    conditionals['when_lv_false'] = compare_methods(
                        baseline_lv_false, method_lv_false
                    )

                if conditionals:
                    monitor_result['conditional'] = conditionals

            run_results['monitors'][monitor_name] = monitor_result

        all_results.append(run_results)

    # Apply multiple comparison corrections
    # Collect all p-values from overall tests
    p_values = []
    test_labels = []
    for run_result in all_results:
        for monitor_name, monitor_result in run_result['monitors'].items():
            if 'overall' in monitor_result and 'p_value' in monitor_result['overall']:
                p_values.append(monitor_result['overall']['p_value'])
                test_labels.append(f"{run_result['run_name']} - {monitor_name}")

    # Apply corrections
    corrections = {}
    if len(p_values) > 0:
        # Bonferroni correction
        bonferroni_alpha = alpha / len(p_values)
        bonferroni_reject = [p < bonferroni_alpha for p in p_values]

        # Holm-Bonferroni correction
        holm_reject, holm_pvals, _, _ = multipletests(
            p_values, alpha=alpha, method='holm'
        )

        # FDR (Benjamini-Hochberg)
        fdr_reject, fdr_pvals, _, _ = multipletests(
            p_values, alpha=alpha, method='fdr_bh'
        )

        corrections = {
            'n_tests': len(p_values),
            'alpha': alpha,
            'bonferroni': {
                'corrected_alpha': bonferroni_alpha,
                'rejections': [
                    {'test': test_labels[i], 'p_value': p_values[i], 'reject': bonferroni_reject[i]}
                    for i in range(len(p_values))
                ]
            },
            'holm_bonferroni': {
                'rejections': [
                    {'test': test_labels[i], 'p_value': p_values[i],
                     'adjusted_p_value': holm_pvals[i], 'reject': bool(holm_reject[i])}
                    for i in range(len(p_values))
                ]
            },
            'fdr_bh': {
                'rejections': [
                    {'test': test_labels[i], 'p_value': p_values[i],
                     'adjusted_p_value': fdr_pvals[i], 'reject': bool(fdr_reject[i])}
                    for i in range(len(p_values))
                ]
            }
        }

    return {
        'config_path': str(config_path),
        'baseline_name': baseline_data['name'],
        'alpha': alpha,
        'results': all_results,
        'multiple_comparison_corrections': corrections
    }


def print_summary(results: Dict[str, Any], show_conditionals: bool = False):
    """Print a formatted summary of test results."""
    print("\n" + "="*80)
    print(f"STATISTICAL SIGNIFICANCE TESTS")
    print(f"Baseline: {results['baseline_name']}")
    print(f"Significance level (α): {results['alpha']}")
    print("="*80)

    # Print individual test results
    for run_result in results['results']:
        print(f"\n{'─'*80}")
        print(f"Method: {run_result['run_name']}")
        print(f"{'─'*80}")

        for monitor_name, monitor_result in run_result['monitors'].items():
            print(f"\n  Monitor: {monitor_name}")

            if 'overall' in monitor_result:
                overall = monitor_result['overall']

                if 'error' in overall:
                    print(f"    ERROR: {overall['error']}")
                    continue

                print(f"    Sample alignment:")
                print(f"      Common samples: {overall['n_common_samples']}")
                print(f"      Baseline only: {overall.get('baseline_only_samples', 0)}")
                print(f"      Method only: {overall.get('method_only_samples', 0)}")

                print(f"\n    Contingency table:")
                ct = overall['contingency_table']
                print(f"      Both correct: {ct['both_correct']}")
                print(f"      Method correct, baseline incorrect: {ct['method_correct_baseline_incorrect']}")
                print(f"      Baseline correct, method incorrect: {ct['baseline_correct_method_incorrect']}")
                print(f"      Both incorrect: {ct['both_incorrect']}")
                print(f"      Discordant pairs: {overall['discordant_pairs']}")

                print(f"\n    Accuracies:")
                print(f"      Baseline: {overall['baseline_accuracy']:.4f}")
                print(f"      Method:   {overall['method_accuracy']:.4f}")
                print(f"      Effect size: {overall['effect_size']:+.4f} ({overall['effect_size_percentage']:+.2f}%)")

                print(f"\n    McNemar's test ({overall['test_type']}):")
                print(f"      Statistic: {overall['mcnemar_statistic']:.4f}")
                print(f"      p-value: {overall['p_value']:.6f}")

                # Quick interpretation
                if overall['p_value'] < results['alpha']:
                    print(f"      → Significant at α={results['alpha']}")
                else:
                    print(f"      → Not significant at α={results['alpha']}")

            # Conditional results (if requested)
            if show_conditionals and 'conditional' in monitor_result:
                for cond_name, cond_result in monitor_result['conditional'].items():
                    print(f"\n    Conditional: {cond_name}")
                    print(f"      Samples: {cond_result['n_common_samples']}")
                    print(f"      Baseline accuracy: {cond_result['baseline_accuracy']:.4f}")
                    print(f"      Method accuracy: {cond_result['method_accuracy']:.4f}")
                    print(f"      Effect size: {cond_result['effect_size']:+.4f}")
                    print(f"      p-value: {cond_result['p_value']:.6f}")

    # Print multiple comparison corrections
    if 'multiple_comparison_corrections' in results and results['multiple_comparison_corrections']:
        print("\n" + "="*80)
        print("MULTIPLE COMPARISON CORRECTIONS")
        print("="*80)

        corr = results['multiple_comparison_corrections']
        print(f"\nTotal tests: {corr['n_tests']}")
        print(f"Family-wise error rate (α): {corr['alpha']}")

        print(f"\n{'─'*80}")
        print("Bonferroni Correction")
        print(f"{'─'*80}")
        print(f"Corrected α: {corr['bonferroni']['corrected_alpha']:.6f}")
        print(f"\n{'Test':<40} {'p-value':<12} {'Reject H0'}")
        print("─"*80)
        for item in corr['bonferroni']['rejections']:
            reject_str = "YES" if item['reject'] else "NO"
            print(f"{item['test']:<40} {item['p_value']:<12.6f} {reject_str}")

        print(f"\n{'─'*80}")
        print("Holm-Bonferroni Correction")
        print(f"{'─'*80}")
        print(f"\n{'Test':<40} {'p-value':<12} {'Adj p-value':<12} {'Reject H0'}")
        print("─"*80)
        for item in corr['holm_bonferroni']['rejections']:
            reject_str = "YES" if item['reject'] else "NO"
            print(f"{item['test']:<40} {item['p_value']:<12.6f} {item['adjusted_p_value']:<12.6f} {reject_str}")

        print(f"\n{'─'*80}")
        print("FDR (Benjamini-Hochberg) Correction")
        print(f"{'─'*80}")
        print(f"\n{'Test':<40} {'p-value':<12} {'Adj p-value':<12} {'Reject H0'}")
        print("─"*80)
        for item in corr['fdr_bh']['rejections']:
            reject_str = "YES" if item['reject'] else "NO"
            print(f"{item['test']:<40} {item['p_value']:<12.6f} {item['adjusted_p_value']:<12.6f} {reject_str}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Run statistical significance tests on experimental runs vs baseline'
    )
    parser.add_argument('config', type=str, help='Path to YAML config file')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('--monitors', type=str, nargs='+',
                       default=['gpt-4o-mini', 'gpt-4o'],
                       help='Monitor names to test (default: gpt-4o-mini gpt-4o)')
    parser.add_argument('--test-conditionals', action='store_true',
                       help='Also test conditional accuracies (when_lv_true, when_lv_false)')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path (optional)')
    parser.add_argument('--show-conditionals', action='store_true',
                       help='Show conditional results in summary')

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    # Run tests
    print("Running statistical significance tests...")
    results = run_significance_tests(
        config_path,
        monitor_names=args.monitors,
        alpha=args.alpha,
        test_conditionals=args.test_conditionals
    )

    # Print summary
    print_summary(results, show_conditionals=args.show_conditionals)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
