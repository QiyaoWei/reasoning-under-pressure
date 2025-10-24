#!/usr/bin/env python3
"""
Script to create conditional reasoner plots based on ground-truth measurement pattern and latent variable.

Two usage modes:
1) Config mode (original): provide --config path/to/config.yaml containing runs.
2) Directory mode (new): provide --dir path/to/predictions directory containing:
   - reasoner_accuracy_summary.json
   - optionally a comparison folder: "compare with baseline" or "compare_with_baseline"
     with reasoner_accuracy_comparison.json for delta plots.

reasoner_accuracy_summary.json should contain "group_metrics"; run reasoner_performance_analysis.py first.
"""

import json
import yaml
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def load_config(config_path: Path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def read_summary(summary_path: Path):
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with open(summary_path, 'r') as f:
        return json.load(f)


def read_comparison_for_summary(summary_path: Path):
    # summary_path -> .../predictions/reasoner_accuracy_summary.json
    # comparison expected at .../predictions/compare with baseline/reasoner_accuracy_comparison.json
    pred_dir = summary_path.parent
    # Try both folder names for compatibility
    candidates = [
        pred_dir / 'compare with baseline',
        pred_dir / 'compare_with_baseline',
    ]
    compare_path = None
    for c in candidates:
        p = c / 'reasoner_accuracy_comparison.json'
        if p.exists():
            compare_path = p
            break
    if compare_path is None:
        return None
    with open(compare_path, 'r') as f:
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


def plot_compare_single(run_name: str, summary: dict, comparison: dict, output_dir: Path):
    if 'group_metrics' not in summary or not isinstance(summary['group_metrics'], dict):
        return
    group_metrics = summary['group_metrics']
    preferred_order = [
        'all_false_lv_false',
        'mixed_lv_false',
        'all_true_lv_true',
        'all_true_lv_false',
    ]
    group_ids = [g for g in preferred_order if g in group_metrics] + [
        g for g in group_metrics.keys() if g not in preferred_order
    ]

    colors = sns.color_palette("pastel", 3)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        {
            'name': 'measurement-wise_accuracy',
            'file': 'compare_conditional_measurement_wise_accuracy.png',
            'ylabel': 'Δ Measurement-wise accuracy (incentive − baseline)',
            'bar_color': colors[2],  # keep green-like for measurement-wise
        },
        {
            'name': 'all_correct_accuracy',
            'file': 'compare_conditional_all_correct_accuracy.png',
            'ylabel': 'Δ All-correct accuracy (incentive − baseline)',
            'bar_color': (0.80, 0.30, 0.50),  # red/pinkish
        },
        {
            'name': 'format_accuracy',
            'file': 'compare_conditional_format_accuracy.png',
            'ylabel': 'Δ Format accuracy (incentive − baseline)',
            'bar_color': (0.60, 0.60, 0.60),  # grey
        },
    ]

    # Baseline CI color
    baseline_color = (0.60, 0.60, 0.60)

    comp_groups = comparison.get('group_metrics', {}) if isinstance(comparison, dict) else {}

    for spec in metric_specs:
        metric_name = spec['name']
        bar_color = spec.get('bar_color', colors[2])
        overlap_color = darker_color(bar_color, 0.6)

        x_labels = []
        diffs = []
        inc_margins = []
        base_margins = []
        overlaps = []  # tuples (low, high) or None

        for gid in group_ids:
            comp_group = comp_groups.get(gid, {}) if isinstance(comp_groups, dict) else {}
            entry = comp_group.get(metric_name, {}) if isinstance(comp_group, dict) else {}
            diff_val = entry.get('diff')
            ci = entry.get('ci', {}) if isinstance(entry, dict) else {}
            inc_margin = ci.get('incentive_margin')
            base_margin = ci.get('baseline_margin')

            # Compute overlap interval between [diff-inc_margin, diff+inc_margin] and [-base_margin, +base_margin]
            ovl = None
            if isinstance(diff_val, (int, float)) and isinstance(inc_margin, (int, float)) and isinstance(base_margin, (int, float)):
                i_low = diff_val - inc_margin
                i_high = diff_val + inc_margin
                b_low = -base_margin
                b_high = base_margin
                low = max(i_low, b_low)
                high = min(i_high, b_high)
                if high > low:
                    ovl = (low, high)

            x_labels.append(gid)
            diffs.append(diff_val or 0.0)
            inc_margins.append(inc_margin or 0.0)
            base_margins.append(base_margin or 0.0)
            overlaps.append(ovl)

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(x_labels))
        width = 0.6

        # Bars centered at 0 showing delta
        ax.bar(x, diffs, width, yerr=inc_margins, capsize=3, color=bar_color, alpha=0.9)

        # Baseline CI around 0 for each group (vertical segment)
        for xi, bmar in zip(x, base_margins):
            if bmar and bmar > 0:
                ax.vlines(x=xi, ymin=-bmar, ymax=bmar, colors=baseline_color, linestyles='-', linewidth=3, alpha=0.9)

        # Overlap highlight where applicable
        for xi, ovl in zip(x, overlaps):
            if ovl is not None:
                ax.vlines(x=xi, ymin=ovl[0], ymax=ovl[1], colors=overlap_color, linestyles='-', linewidth=8, alpha=0.5)

        # Zero reference line (baseline)
        zero_line = ax.axhline(y=0.0, color=baseline_color, linestyle='--', linewidth=2, alpha=0.8, label='Baseline (0)')

        ax.set_ylabel(spec['ylabel'])
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha='right')
        # Dynamic y-limit: ensure full bars and baseline CI are visible
        max_extent = 0.25
        for d, im, bm in zip(diffs, inc_margins, base_margins):
            try:
                d_val = float(d)
                im_val = float(im or 0.0)
                bm_val = float(bm or 0.0)
                local = max(abs(d_val) + im_val, bm_val)
                if np.isfinite(local):
                    max_extent = max(max_extent, local)
            except Exception:
                continue
        limit = min(1.0, max_extent + 0.03)
        ax.set_ylim(-limit, limit)
        ax.grid(True, which='major', axis='y', alpha=0.8, linestyle='-', linewidth=0.8)
        ax.grid(True, which='major', axis='x', alpha=0.8, linestyle='-', linewidth=0.8)
        ax.minorticks_on()
        ax.grid(True, which='minor', axis='y', alpha=0.5, linestyle=':', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend: Baseline CI, Incentive CI, Overlap
        baseline_ci_proxy = Line2D([0], [0], color=baseline_color, linestyle='-', linewidth=3, label='Baseline 95% CI')
        incentive_ci_proxy = Line2D([0], [0], color=bar_color, linestyle='-', linewidth=1.5, marker='_', markersize=10, label='Incentive 95% CI (error bars)')
        overlap_proxy = Line2D([0], [0], color=overlap_color, linestyle='-', linewidth=8, alpha=0.5, label='CI overlap')
        ax.legend(handles=[zero_line, baseline_ci_proxy, incentive_ci_proxy, overlap_proxy], loc='upper right', fontsize=11)

        plt.tight_layout()
        out_path = output_dir / spec['file']
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create conditional reasoner plots from config or a predictions directory.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (same schema as create_main_plot.py).')
    parser.add_argument('--dir', type=str, default=None, help='Path to a predictions directory containing reasoner_accuracy_summary.json.')
    parser.add_argument('--output-dir', type=str, default=None, help='Base directory to save plots; default uses dataset_name.')
    args = parser.parse_args()

    if not args.config and not args.dir:
        raise ValueError('Please provide either --config path/to/config.yaml or --dir path/to/predictions directory.')

    to_process = []
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = load_config(config_path)
        baseline = cfg.get('baseline')
        if baseline and 'accuracy_path' in baseline:
            to_process.append((baseline.get('name', 'baseline'), Path(baseline['accuracy_path'])))
        for run in cfg.get('runs', []):
            if 'accuracy_path' in run:
                to_process.append((run.get('name', 'run'), Path(run['accuracy_path'])))
        if not to_process:
            raise ValueError('No runs with accuracy_path found in config.')
    else:
        pred_dir = Path(args.dir)
        if not pred_dir.exists() or not pred_dir.is_dir():
            raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")
        reasoner_path = pred_dir / 'reasoner_accuracy_summary.json'
        if not reasoner_path.exists():
            raise FileNotFoundError(f"reasoner_accuracy_summary.json not found in directory: {pred_dir}")
        run_name = pred_dir.name
        to_process.append((run_name, reasoner_path))

    dataset_name = get_dataset_name_from_any([p for _, p in to_process])
    if args.output_dir:
        base_out = Path(args.output_dir)
    else:
        base_out = Path('results') / 'plots' / dataset_name / 'conditional_reasoner'

    # If using directory mode with an explicit output dir, write directly into that folder
    write_direct_to_base = bool(args.dir) and bool(args.output_dir)

    for run_name, summary_path in to_process:
        data = read_summary(summary_path)
        if 'group_metrics' not in data:
            print(
                f"\nWarning: group_metrics missing in {summary_path}. Please run reasoner_performance_analysis.py for the corresponding predictions file first."
            )
            continue
        out_dir = base_out if write_direct_to_base else (base_out / sanitize_name(run_name))
        plot_single(run_name, data, out_dir)
        # Comparison plots (differences vs. baseline) if available
        comparison = read_comparison_for_summary(summary_path)
        if comparison is not None:
            plot_compare_single(run_name, data, comparison, out_dir)


if __name__ == '__main__':
    main()


