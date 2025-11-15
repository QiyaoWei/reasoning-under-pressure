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

def extract_training_data_from_yaml(config_path):
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
                    'kl_se': values['kl_se_value']
                }
        except Exception as e:
            print(f"Warning: Could not load KL values from {config['kl_values_path']}: {e}")

    # Load verbosity values if provided
    if 'verbosity_values_path' in config:
        try:
            with open(config['verbosity_values_path'], 'r') as f:
                verbosity_data = json.load(f)
            # Index by step number for easy lookup
            for path, values in verbosity_data.items():
                step = values['step']
                kl_data_by_step[step] = {
                    'kl_value': values['verbosity_value'],
                    'kl_se': values['verbosity_se_value']
                }
        except Exception as e:
            print(f"Warning: Could not load verbosity values from {config['verbosity_values_path']}: {e}")

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
        incentive_se = None

        if step in kl_data_by_step:
            # Use global KL values file
            incentive_value = kl_data_by_step[step]['kl_value']
            incentive_se = kl_data_by_step[step]['kl_se']
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
                'incentive_se': incentive_se
            })

    return data

def extract_training_data_from_json(json_path):
    """Extract training progress data from JSON file (func_corr format)."""
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    data = []

    # Extract setting from description or use coefficient if available
    description = json_data.get('description', '')

    # Try to extract coefficient from different possible fields
    kl_coef = json_data.get('kl_coef')
    verbosity_k = json_data.get('verbosity_k')
    monitorability_k = json_data.get('monitorability_k')

    if kl_coef is not None:
        setting = f"KL coef {kl_coef} training progress"
    elif verbosity_k is not None:
        setting = f"Response length k {verbosity_k} training progress"
    elif monitorability_k is not None:
        setting = f"Monitorability k {monitorability_k} training progress"
    else:
        setting = description or 'unknown'

    checkpoints = json_data.get('checkpoints', [])

    for checkpoint in checkpoints:
        step = checkpoint.get('step')
        if step is None:
            continue

        # Extract reasoner accuracy (measurement_accuracy)
        measurement_acc = checkpoint.get('measurement_accuracy', {})
        reasoner_accuracy = measurement_acc.get('mean')
        ci_info = measurement_acc.get('t_confidence_interval_95', {})

        # Extract monitor accuracies
        monitor_data = {}

        # GPT-4o-mini monitor
        gpt4o_mini_acc = checkpoint.get('monitor_accuracy_gpt_4o_mini')
        gpt4o_mini_ci = checkpoint.get('monitor_accuracy_gpt_4o_mini_ci_95', {})
        if gpt4o_mini_acc is not None:
            monitor_data['gpt-4o-mini'] = {
                'accuracy': gpt4o_mini_acc,
                'ci_lower': gpt4o_mini_ci.get('lower_bound'),
                'ci_upper': gpt4o_mini_ci.get('upper_bound'),
                'ci_margin': gpt4o_mini_ci.get('margin')
            }

        # GPT-4o monitor
        gpt4o_acc = checkpoint.get('monitor_accuracy_gpt_4o')
        gpt4o_ci = checkpoint.get('monitor_accuracy_gpt_4o_ci_95', {})
        if gpt4o_acc is not None:
            monitor_data['gpt-4o'] = {
                'accuracy': gpt4o_acc,
                'ci_lower': gpt4o_ci.get('lower_bound'),
                'ci_upper': gpt4o_ci.get('upper_bound'),
                'ci_margin': gpt4o_ci.get('margin')
            }

        # Extract incentive value - could be from different sources
        incentive_value = None
        incentive_se = None

        # For monitorability files, use monitor_accuracy_incentive
        if 'monitor_accuracy_incentive' in checkpoint:
            incentive_value = checkpoint.get('monitor_accuracy_incentive')
            # SE not available for this field
            incentive_se = None
        else:
            # For KL files, try to get from kl_values
            kl_values = checkpoint.get('kl_values', {})
            # Only use mean if it exists and is not None
            if 'mean' in kl_values and kl_values['mean'] is not None:
                incentive_value = kl_values.get('mean')
                incentive_se = kl_values.get('se')
            # For verbosity files, get from num_words_before_measurements
            elif 'num_words_before_measurements' in checkpoint:
                verbosity_data = checkpoint.get('num_words_before_measurements', {})
                if 'mean' in verbosity_data and verbosity_data['mean'] is not None:
                    incentive_value = verbosity_data.get('mean')
                    incentive_se = verbosity_data.get('se')

        # Only add data if we have at least the reasoner accuracy
        if reasoner_accuracy is not None:
            data.append({
                'step': step,
                'setting': setting,
                'reasoner_accuracy': reasoner_accuracy,
                'reasoner_accuracy_ci': {
                    'lower': ci_info.get('lower_bound'),
                    'upper': ci_info.get('upper_bound'),
                    'margin': ci_info.get('margin'),
                },
                'monitors': monitor_data,
                'incentive_value': incentive_value,
                'incentive_se': incentive_se
            })

    return data

def extract_training_data(config_path):
    """Extract training progress data from either YAML or JSON file."""
    config_path = Path(config_path)

    if config_path.suffix.lower() == '.json':
        return extract_training_data_from_json(config_path)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        return extract_training_data_from_yaml(config_path)
    else:
        raise ValueError(f"Unsupported file format: {config_path.suffix}. Please provide a .yaml or .json file.")

def create_plot(data, output_dir: Path, highlight_step=None, experiment_name=None, source_filename=None):
    """Create separate accuracy and monitorability plots showing training progress.

    Args:
        data: Training data to plot
        output_dir: Directory to save plots
        highlight_step: Optional step number to highlight with a star marker
        experiment_name: Experiment prefix (e.g., 'diamond_vault' or 'functional_correctness')
        source_filename: Original config/json filename to extract action from
    """
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
    incentive_ses = []

    # Extract error bars
    reasoner_errors = []
    gpt4o_mini_errors = []
    gpt4o_errors = []

    for item in data:
        steps.append(item['step'])
        reasoner_accs.append(item['reasoner_accuracy'])
        incentive_values.append(item.get('incentive_value'))
        incentive_ses.append(item.get('incentive_se'))

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

    # Create output directory with descriptive name
    setting = data[0].get('setting', 'unknown') if data else 'unknown'

    # Create a more descriptive folder name based on the setting
    # Examples: "penalize_kl_1e-3", "reward_verbosity_1e-7", etc.
    base_name = setting.lower().replace(' training progress', '').replace(' ', '_').replace('coef_', 'coef').replace('=', '')

    # Add experiment prefix if provided
    if experiment_name:
        folder_name = f"{experiment_name}_{base_name}"
        filename_prefix = f"{experiment_name}_{base_name}"
    else:
        folder_name = base_name
        filename_prefix = base_name

    # Create simplified name for unified folder: {experiment}_{action}_{metric}
    # Extract action (penalise/reward) and metric (kl/verbosity/monitorability)
    setting_lower = setting.lower()

    # Try to extract action from setting first, then from source filename
    if 'penalize' in setting_lower or 'penalise' in setting_lower:
        action = 'penalise'
    elif 'reward' in setting_lower:
        action = 'reward'
    elif source_filename:
        filename_lower = source_filename.lower()
        if 'penalize' in filename_lower or 'penalise' in filename_lower:
            action = 'penalise'
        elif 'reward' in filename_lower:
            action = 'reward'
        else:
            action = 'unknown'
    else:
        action = 'unknown'

    if 'response length' in setting_lower or 'verbosity' in setting_lower:
        metric = 'verbosity'
    elif 'kl' in setting_lower:
        metric = 'kl'
    elif 'monitorability' in setting_lower:
        metric = 'monitorability'
    else:
        metric = 'unknown'

    if experiment_name:
        simplified_name = f"{experiment_name}_{action}_{metric}"
    else:
        simplified_name = f"{action}_{metric}"

    output_subdir = output_dir / 'plots' / folder_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Also create unified monitorability folder
    monitorability_unified_dir = output_dir / 'plots' / 'all_monitorability_plots'
    monitorability_unified_dir.mkdir(parents=True, exist_ok=True)

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

    # Highlight specific step if requested
    if highlight_step is not None and highlight_step in steps:
        idx = steps.index(highlight_step)
        ax1.scatter([highlight_step], [reasoner_vals[idx]],
                   marker='*', s=500, color='gold', edgecolors='black',
                   linewidths=1.5, zorder=5, label=f'Main plot step ({highlight_step})')

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='best', fontsize=11)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True, which='major', axis='both', alpha=0.8, linestyle='-', linewidth=0.8)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', axis='both', alpha=0.5, linestyle=':', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.tight_layout()
    accuracy_path = output_subdir / 'accuracy_training_plot.pdf'
    plt.savefig(accuracy_path, bbox_inches='tight')
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

    # Highlight specific step if requested
    if highlight_step is not None and highlight_step in steps:
        idx = steps.index(highlight_step)
        # Highlight all three lines at this step
        ax2.scatter([highlight_step], [reasoner_vals[idx]],
                   marker='*', s=500, color='gold', edgecolors='black',
                   linewidths=1.5, zorder=5)
        if not np.isnan(mini_vals[idx]):
            ax2.scatter([highlight_step], [mini_vals[idx]],
                       marker='*', s=500, color='gold', edgecolors='black',
                       linewidths=1.5, zorder=5)
        if not np.isnan(gpt4o_vals[idx]):
            ax2.scatter([highlight_step], [gpt4o_vals[idx]],
                       marker='*', s=500, color='gold', edgecolors='black',
                       linewidths=1.5, zorder=5, label=f'Main plot step ({highlight_step})')

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Accuracy / Monitorability', fontsize=12)
    ax2.set_ylim(0.5, 1.0)
    ax2.grid(True, which='major', axis='both', alpha=0.8, linestyle='-', linewidth=0.8)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', axis='both', alpha=0.5, linestyle=':', linewidth=0.5)

    # Right y-axis: Incentive values
    incentive_vals = safe_vals(incentive_values)
    incentive_se_vals = safe_vals(incentive_ses)
    if any(not np.isnan(v) for v in incentive_vals):
        ax2_right = ax2.twinx()
        incentive_color = sns.color_palette("muted", 4)[3]  # Different color for incentive

        # Determine incentive type from setting name
        setting_lower = setting.lower()
        if 'response length' in setting_lower or 'verbosity' in setting_lower:
            incentive_label = 'Response length (words)'
            ylabel = 'Response Length (# words)'
        elif 'kl' in setting_lower:
            incentive_label = 'KL divergence'
            ylabel = 'KL Divergence'
        elif 'monitorability' in setting_lower:
            incentive_label = 'Monitorability'
            ylabel = 'Monitorability'
        else:
            incentive_label = 'Incentive value'
            ylabel = 'Incentive Value'

        ax2_right.plot(steps, incentive_vals, marker='D', linewidth=2.5, markersize=8,
                      label=incentive_label, color=incentive_color, alpha=0.9, linestyle='--')

        # Add error bars/shading if SE values are available
        if any(not np.isnan(v) for v in incentive_se_vals):
            ax2_right.fill_between(steps,
                                   [v - s if not np.isnan(v) and not np.isnan(s) else np.nan
                                    for v, s in zip(incentive_vals, incentive_se_vals)],
                                   [v + s if not np.isnan(v) and not np.isnan(s) else np.nan
                                    for v, s in zip(incentive_vals, incentive_se_vals)],
                                   alpha=0.2, color=incentive_color)

        ax2_right.set_ylabel(ylabel, fontsize=12)
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
    monitorability_path = output_subdir / 'monitorability_training_plot.pdf'
    plt.savefig(monitorability_path, bbox_inches='tight')

    # Also save to unified monitorability folder with simplified name
    unified_filename = f'{simplified_name}.pdf'
    unified_monitorability_path = monitorability_unified_dir / unified_filename
    plt.savefig(unified_monitorability_path, bbox_inches='tight')

    plt.close()

    # Save data as JSON
    data_json_path = output_subdir / 'training_plot_data.json'
    with open(data_json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Accuracy plot saved as {accuracy_path}")
    print(f"Monitorability plot saved as {monitorability_path}")
    print(f"Data JSON saved as {data_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training progress plots from YAML or JSON config.")
    parser.add_argument("config", type=str,
                       help="Path to YAML or JSON config file. "
                            "YAML format: training_steps.yaml with checkpoints and paths. "
                            "JSON format: training data with checkpoints array (func_corr format).")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save the plot and JSON outputs.")
    parser.add_argument("--highlight-step", type=int, default=None,
                       help="Optional training step to highlight with a star marker (e.g., the step used in main plot).")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name prefix (e.g., 'diamond_vault' or 'functional_correctness').")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist!")
        exit(1)

    data = extract_training_data(config_path)
    create_plot(data, output_dir, highlight_step=args.highlight_step, experiment_name=args.experiment_name, source_filename=config_path.name)
