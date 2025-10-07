#!/usr/bin/env python3
"""
Script to create conditional reasoner plots based on ground-truth measurement pattern and latent variable.
Uses YAML config file to specify runs and their data sources.

reasoner_accuracy_summary.json should contain "group_metrics", run reasoner_performance_analysis.py first.
"""

import json
import yaml
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: Path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def read_summary(summary_path: Path):
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with open(summary_path, 'r') as f:
        return json.load(f)


def sanitize_name(name: str) -> str:
    return ''.join(c if (c.isalnum() or c in ('-', '_')) else '_' for c in name.strip())


def get_dataset_name_from_any(summaries: list) -> str:
    for path in summaries:
        try:
            data = read_summary(path)
            ds = data.get('dataset_name')
            if isinstance(ds, str) and ds:
                return ds
        except Exception:
            continue
    return 'unknown_dataset'


def extract_overall(summary: dict, metric_key_overall: str):
    obj = summary.get(metric_key_overall)
    if not isinstance(obj, dict) or 'value' not in obj:
        return None
    value = obj['value']
    ci = obj.get('confidence_interval') or [None, None, None]
    lower, upper, margin = None, None, None
    if isinstance(ci, list) and len(ci) >= 3:
        lower, upper, margin = ci[0], ci[1], ci[2]
    # Fallback if only margin available
    if (lower is None or upper is None) and margin is not None and isinstance(value, (int, float)):
        lower = max(0.0, float(value) - float(margin))
        upper = min(1.0, float(value) + float(margin))
    return {
        'value': value,
        'lower': lower,
        'upper': upper,
        'margin': margin,
    }


def darker_color(color, factor=0.8):
    try:
        r, g, b = color
        return (max(0.0, min(1.0, r * factor)),
                max(0.0, min(1.0, g * factor)),
                max(0.0, min(1.0, b * factor)))
    except Exception:
        return color


def plot_single(run_name: str, summary: dict, output_dir: Path):
    if 'group_metrics' not in summary or not isinstance(summary['group_metrics'], dict):
        print(
            f"Warning: Missing group_metrics in {output_dir}. Please run reasoner_performance_analysis.py for the corresponding predictions file first."
        )

    group_metrics = summary['group_metrics']
    # Prefer a stable order if present
    preferred_order = [
        'all_false_lv_false',
        'mixed_lv_false',
        'all_true_lv_true',
        'all_true_lv_false',
    ]
    group_ids = [g for g in preferred_order if g in group_metrics] + [
        g for g in group_metrics.keys() if g not in preferred_order
    ]

    # Prepare plotting
    colors = sns.color_palette("pastel", 3)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        {
            'name': 'measurement-wise_accuracy',
            'file': 'conditional_measurement_wise_accuracy.png',
            'overall_key': 'measurement-wise_accuracy_overall',
            'ylabel': 'Measurement-wise accuracy',
            'bar_color': None,  # set later to colors[2]
        },
        {
            'name': 'all_correct_accuracy',
            'file': 'conditional_all_correct_accuracy.png',
            'overall_key': 'all_correct_accuracy_overall',
            'ylabel': 'All-correct accuracy',
            'bar_color': (0.80, 0.30, 0.50),  # red/magenta tone
        },
        {
            'name': 'format_accuracy',
            'file': 'conditional_format_accuracy.png',
            'overall_key': 'format_accuracy_overall',
            'ylabel': 'Format accuracy',
            'bar_color': (0.60, 0.60, 0.60),  # grey
        },
    ]

    # Assign measurement-wise bar color to match create_main_plot (green-like)
    metric_specs[0]['bar_color'] = colors[2]

    for spec in metric_specs:
        metric_name = spec['name']
        overall = extract_overall(summary, spec['overall_key'])
        if overall is None or overall.get('value') is None:
            raise ValueError(f"Missing overall metric {spec['overall_key']} in summary JSON.")

        # Gather bar values
        x_labels = []
        values = []
        errors = []
        n_samples = []
        for gid in group_ids:
            gm = group_metrics.get(gid, {})
            acc = gm.get(metric_name, {})
            val = acc['value']
            ci = acc.get('confidence_interval') or [None, None, None]
            margin = ci[2] if isinstance(ci, list) and len(ci) >= 3 else None
            x_labels.append(gid)
            values.append(val)
            errors.append(margin or 0)
            n_samples.append(gm.get('n_samples', None))

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(x_labels))
        width = 0.6
        bar_color = spec['bar_color']
        ax.bar(x, values, width, yerr=errors, capsize=3, color=bar_color, alpha=0.9)

        # Overall line and shaded CI
        overall_val = overall['value']
        line_color = darker_color(bar_color, 0.8)
        line = ax.axhline(y=overall_val, color=line_color, linestyle='--', linewidth=2, alpha=0.8, label='Overall accuracy')
        if overall.get('lower') is not None and overall.get('upper') is not None:
            ax.axhspan(overall['lower'], overall['upper'], xmin=0, xmax=1, color=bar_color, alpha=0.25, zorder=0)

        ax.set_ylabel(spec['ylabel'])
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha='right')
        ax.set_ylim(0.0, 1.0)
        # Match grid/line styling similar to create_main_plot
        ax.grid(True, which='major', axis='y', alpha=0.8, linestyle='-', linewidth=0.8)
        ax.grid(True, which='major', axis='x', alpha=0.8, linestyle='-', linewidth=0.8)
        ax.minorticks_on()
        ax.grid(True, which='minor', axis='y', alpha=0.5, linestyle=':', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend for overall accuracy line
        ax.legend(loc='upper right', fontsize=11)

        # Annotate bars with n samples
        for i, (xi, val, err, n) in enumerate(zip(x, values, errors, n_samples)):
            y_text = float(val) + float(err or 0) + 0.02
            label = f"n={n}" if n is not None else "n=?"
            ax.text(xi, y_text, label, ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        out_path = output_dir / spec['file']
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create conditional reasoner plots from YAML config.')
    parser.add_argument('config', type=str, help='Path to YAML config file (same schema as create_main_plot.py).')
    parser.add_argument('--output-dir', type=str, default=None, help='Base directory to save plots; default uses dataset_name.')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    to_process = []
    baseline = cfg.get('baseline')
    if baseline and 'accuracy_path' in baseline:
        to_process.append((baseline['name'], Path(baseline['accuracy_path'])))
    for run in cfg.get('runs', []):
        if 'accuracy_path' in run:
            to_process.append((run['name'], Path(run['accuracy_path'])))

    if not to_process:
        raise ValueError('No runs with accuracy_path found in config.')

    dataset_name = get_dataset_name_from_any([p for _, p in to_process])
    if args.output_dir:
        base_out = Path(args.output_dir)
    else:
        base_out = Path('results') / 'plots' / dataset_name / 'conditional_reasoner'

    for run_name, summary_path in to_process:
        data = read_summary(summary_path)
        if 'group_metrics' not in data:
            print(
                f"\nWarning: group_metrics missing in {summary_path}. Please run reasoner_performance_analysis.py for the corresponding predictions file first."
            )
            continue
        out_dir = base_out / sanitize_name(run_name)
        plot_single(run_name, data, out_dir)


if __name__ == '__main__':
    main()


