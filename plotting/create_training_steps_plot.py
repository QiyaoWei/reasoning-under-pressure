#!/usr/bin/env python3
"""
Script to create a plot showing training progress across different steps.
Uses YAML config file to specify checkpoints and their data sources.
"""

import json
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime

def _darker_color(color, factor=0.8):
    """Return a slightly darker shade of the given RGB color by blending toward black.
    factor < 1 darkens; keep within [0,1].
    """
    try:
        r, g, b = color
        return (max(0.0, min(1.0, r * factor)),
                max(0.0, min(1.0, g * factor)),
                max(0.0, min(1.0, b * factor)))
    except Exception:
        return color

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_accuracy_from_json(json_path):
    """Extract accuracy and confidence interval from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Try to get accuracy from different possible locations
        accuracy = None
        ci_lower = None
        ci_upper = None
        ci_margin = None

        # Check for metrics.measurement-wise_accuracy_overall.value
        if 'metrics' in data and isinstance(data['metrics'], dict):
            metrics = data['metrics']
            if 'measurement-wise_accuracy_overall' in metrics:
                mwao = metrics['measurement-wise_accuracy_overall']
                if isinstance(mwao, dict) and 'value' in mwao:
                    accuracy = mwao['value']
                    ci = mwao.get('confidence_interval')
                    if isinstance(ci, list) and len(ci) >= 3:
                        ci_lower, ci_upper, ci_margin = ci[0], ci[1], ci[2]

        # Fallback: check root level
        if accuracy is None and 'measurement-wise_accuracy_overall' in data:
            val = data['measurement-wise_accuracy_overall']
            if isinstance(val, dict) and 'value' in val:
                accuracy = val['value']
                ci = val.get('confidence_interval')
                if isinstance(ci, list) and len(ci) >= 3:
                    ci_lower, ci_upper, ci_margin = ci[0], ci[1], ci[2]
            elif isinstance(val, (int, float)):
                accuracy = float(val)

        # For monitor files, look for 'accuracy' field
        if accuracy is None and 'accuracy' in data:
            accuracy = data['accuracy']
            ci_obj = data.get('accuracy_95_ci')
            if isinstance(ci_obj, dict):
                ci_lower = ci_obj.get('lower_bound')
                ci_upper = ci_obj.get('upper_bound')
                ci_margin = ci_obj.get('margin')

        return {
            'accuracy': accuracy,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_margin': ci_margin
        }

    except Exception as e:
        return {
            'accuracy': None,
            'ci_lower': None,
            'ci_upper': None,
            'ci_margin': None
        }

def extract_training_data(config_path):
    """Extract training progress data from YAML config."""
    config = load_config(config_path)
    data = []
    setting = config.get('setting', 'unknown')

    # Load KL values if provided
    kl_data_by_step = {}
    if 'kl_values_path' in config:
        try:
            with open(config['kl_values_path'], 'r') as f:
                kl_data = json.load(f)
            # Index by step number for easy lookup
            for path, values in kl_data.items():
                step = values['step']
                kl_data_by_step[step] = {
                    'kl_value': values['kl_value'],
                    'kl_std': values['kl_std_value']
                }
        except Exception as e:
            print(f"Warning: Could not load KL values from {config['kl_values_path']}: {e}")

    for checkpoint in config['checkpoints']:
        step = checkpoint['step']

        # Extract reasoner accuracy
        reasoner_data = extract_accuracy_from_json(checkpoint['accuracy_path'])

        # Extract monitor accuracies
        monitor_data = {}
        for monitor in checkpoint['monitors']:
            monitor_name = monitor['name']
            monitor_acc = extract_accuracy_from_json(monitor['accuracy_path'])
            monitor_data[monitor_name] = monitor_acc

        # Extract incentive value if present (from per-checkpoint file or global KL file)
        incentive_value = None
        incentive_std = None

        if step in kl_data_by_step:
            # Use global KL values file
            incentive_value = kl_data_by_step[step]['kl_value']
            incentive_std = kl_data_by_step[step]['kl_std']
        elif 'incentive_path' in checkpoint:
            # Fallback to per-checkpoint incentive file
            incentive_data = extract_accuracy_from_json(checkpoint['incentive_path'])
            incentive_value = incentive_data['accuracy']  # Reuse the same extraction logic

        # Only add data if we have at least the reasoner accuracy
        if reasoner_data['accuracy'] is not None:
            data.append({
                'step': step,
                'setting': setting,
                'reasoner_accuracy': reasoner_data['accuracy'],
                'reasoner_accuracy_ci': {
                    'lower': reasoner_data['ci_lower'],
                    'upper': reasoner_data['ci_upper'],
                    'margin': reasoner_data['ci_margin'],
                },
                'monitors': monitor_data,
                'incentive_value': incentive_value,
                'incentive_std': incentive_std
            })

    return data

def create_plot(data, output_dir: Path):
    """Create separate accuracy and monitorability plots showing training progress."""
    if not data:
        print("No data found!")
        return

    # Sort data by step
    data = sorted(data, key=lambda x: x['step'])

    # Prepare data for plotting
    steps = []
    reasoner_accs = []
    gpt4o_mini_accs = []
    gpt4o_accs = []
    incentive_values = []
    incentive_stds = []

    # Extract error bars
    reasoner_errors = []
    gpt4o_mini_errors = []
    gpt4o_errors = []

    for item in data:
        steps.append(item['step'])
        reasoner_accs.append(item['reasoner_accuracy'])
        incentive_values.append(item.get('incentive_value'))
        incentive_stds.append(item.get('incentive_std'))

        # Get monitor accuracies
        gpt4o_mini_acc = None
        gpt4o_acc = None
        gpt4o_mini_error = 0
        gpt4o_error = 0

        for monitor_name, monitor_data in item['monitors'].items():
            if monitor_name == 'gpt-4o-mini':
                gpt4o_mini_acc = monitor_data['accuracy']
                gpt4o_mini_error = monitor_data['ci_margin'] or 0
            elif monitor_name == 'gpt-4o':
                gpt4o_acc = monitor_data['accuracy']
                gpt4o_error = monitor_data['ci_margin'] or 0

        gpt4o_mini_accs.append(gpt4o_mini_acc)
        gpt4o_accs.append(gpt4o_acc)
        reasoner_errors.append(item['reasoner_accuracy_ci']['margin'] or 0)
        gpt4o_mini_errors.append(gpt4o_mini_error)
        gpt4o_errors.append(gpt4o_error)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setting = data[0].get('setting', 'unknown') if data else 'unknown'
    setting_first_word = setting.split()[0] if setting else 'unknown'
    output_subdir = output_dir / 'plots' / f"{setting_first_word}_training_{timestamp}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Get seaborn pastel colors
    colors = sns.color_palette("pastel", 3)

    # Replace None with NaN so lines simply don't render for missing values
    def safe_vals(vals):
        return [v if isinstance(v, (int, float)) else np.nan for v in vals]

    reasoner_vals = safe_vals(reasoner_accs)
    mini_vals = safe_vals(gpt4o_mini_accs)
    gpt4o_vals = safe_vals(gpt4o_accs)

    # Create accuracy plot
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    ax1.plot(steps, reasoner_vals, marker='o', linewidth=2.5, markersize=8,
             label='Reasoner accuracy', color=colors[2], alpha=0.9)
    ax1.fill_between(steps,
                      [r - e for r, e in zip(reasoner_vals, reasoner_errors)],
                      [r + e for r, e in zip(reasoner_vals, reasoner_errors)],
                      alpha=0.2, color=colors[2])

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='best', fontsize=11)
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, which='major', axis='both', alpha=0.8, linestyle='-', linewidth=0.8)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', axis='both', alpha=0.5, linestyle=':', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.tight_layout()
    accuracy_path = output_subdir / 'accuracy_training_plot.png'
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create monitorability plot with dual y-axis
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    # Left y-axis: Monitorability and Reasoner Accuracy
    ax2.plot(steps, reasoner_vals, marker='o', linewidth=2.5, markersize=8,
             label='Reasoner accuracy', color=colors[2], alpha=0.9)
    ax2.fill_between(steps,
                      [r - e for r, e in zip(reasoner_vals, reasoner_errors)],
                      [r + e for r, e in zip(reasoner_vals, reasoner_errors)],
                      alpha=0.2, color=colors[2])

    ax2.plot(steps, mini_vals, marker='s', linewidth=2.5, markersize=8,
             label='GPT-4o mini', color=colors[1], alpha=0.9)
    ax2.fill_between(steps,
                      [m - e if not np.isnan(m) else np.nan for m, e in zip(mini_vals, gpt4o_mini_errors)],
                      [m + e if not np.isnan(m) else np.nan for m, e in zip(mini_vals, gpt4o_mini_errors)],
                      alpha=0.2, color=colors[1])

    ax2.plot(steps, gpt4o_vals, marker='^', linewidth=2.5, markersize=8,
             label='GPT-4o', color=colors[0], alpha=0.9)
    ax2.fill_between(steps,
                      [g - e if not np.isnan(g) else np.nan for g, e in zip(gpt4o_vals, gpt4o_errors)],
                      [g + e if not np.isnan(g) else np.nan for g, e in zip(gpt4o_vals, gpt4o_errors)],
                      alpha=0.2, color=colors[0])

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Accuracy / Monitorability', fontsize=12)
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, which='major', axis='both', alpha=0.8, linestyle='-', linewidth=0.8)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', axis='both', alpha=0.5, linestyle=':', linewidth=0.5)

    # Right y-axis: Incentive values
    incentive_vals = safe_vals(incentive_values)
    incentive_std_vals = safe_vals(incentive_stds)
    if any(not np.isnan(v) for v in incentive_vals):
        ax2_right = ax2.twinx()
        incentive_color = sns.color_palette("muted", 4)[3]  # Different color for incentive
        ax2_right.plot(steps, incentive_vals, marker='D', linewidth=2.5, markersize=8,
                      label='Incentive value (KL)', color=incentive_color, alpha=0.9, linestyle='--')

        # Add error bars/shading if std values are available
        if any(not np.isnan(v) for v in incentive_std_vals):
            ax2_right.fill_between(steps,
                                   [v - s if not np.isnan(v) and not np.isnan(s) else np.nan
                                    for v, s in zip(incentive_vals, incentive_std_vals)],
                                   [v + s if not np.isnan(v) and not np.isnan(s) else np.nan
                                    for v, s in zip(incentive_vals, incentive_std_vals)],
                                   alpha=0.2, color=incentive_color)

        ax2_right.set_ylabel('Incentive Value (KL Divergence)', fontsize=12)
        ax2_right.spines['top'].set_visible(False)

        # Combine legends from both axes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
    else:
        ax2.legend(loc='best', fontsize=11)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    monitorability_path = output_subdir / 'monitorability_training_plot.png'
    plt.savefig(monitorability_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save data as JSON
    data_json_path = output_subdir / 'training_plot_data.json'
    with open(data_json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Accuracy plot saved as {accuracy_path}")
    print(f"Monitorability plot saved as {monitorability_path}")
    print(f"Data JSON saved as {data_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training progress plots from YAML config.")
    parser.add_argument("config", type=str,
                       help="Path to YAML config file.")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save the plot and JSON outputs.")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist!")
        exit(1)

    data = extract_training_data(config_path)
    create_plot(data, output_dir)
