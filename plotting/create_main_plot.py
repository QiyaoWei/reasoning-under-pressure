#!/usr/bin/env python3
"""
Script to create a plot showing different incentives and their corresponding accuracies.
Uses YAML config file to specify runs and their data sources.
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
from matplotlib.patches import Patch

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

def extract_accuracy_data(config_path):
    """Extract accuracy data from YAML config."""
    config = load_config(config_path)
    data = []
    baseline_data = None
    setting = config.get('setting', 'unknown')
    
    # Extract baseline data if present
    if 'baseline' in config:
        baseline = config['baseline']
        print(f"Processing baseline: {baseline['name']}")
        
        # Extract reasoner accuracy
        reasoner_data = extract_accuracy_from_json(baseline['accuracy_path'])
        
        # Extract monitor accuracies
        monitor_data = {}
        for monitor in baseline['monitors']:
            monitor_name = monitor['name']
            monitor_acc = extract_accuracy_from_json(monitor['accuracy_path'])
            monitor_data[monitor_name] = {
                **monitor_acc,
                'significant': bool(monitor.get('significant', False))
            }
        
        if reasoner_data['accuracy'] is not None:
            baseline_data = {
                'name': baseline['name'],
                'setting': setting,
                'reasoner_accuracy': reasoner_data['accuracy'],
                'reasoner_accuracy_ci': {
                    'lower': reasoner_data['ci_lower'],
                    'upper': reasoner_data['ci_upper'],
                    'margin': reasoner_data['ci_margin'],
                },
                'monitors': monitor_data,
                'reasoner_significant': bool(baseline.get('reasoner_significant', False))
            }
    
    for run in config['runs']:
        run_name = run['name']
        
        # Extract reasoner accuracy
        reasoner_data = extract_accuracy_from_json(run['accuracy_path'])
        
        # Extract monitor accuracies
        monitor_data = {}
        for monitor in run['monitors']:
            monitor_name = monitor['name']
            monitor_acc = extract_accuracy_from_json(monitor['accuracy_path'])
            monitor_data[monitor_name] = {
                **monitor_acc,
                'significant': bool(monitor.get('significant', False))
            }
        
        # Only add data if we have at least the reasoner accuracy
        if reasoner_data['accuracy'] is not None:
            data.append({
                'name': run_name,
                'setting': setting,
                'reasoner_accuracy': reasoner_data['accuracy'],
                'reasoner_accuracy_ci': {
                    'lower': reasoner_data['ci_lower'],
                    'upper': reasoner_data['ci_upper'],
                    'margin': reasoner_data['ci_margin'],
                },
                'monitors': monitor_data,
                'reasoner_significant': bool(run.get('reasoner_significant', False))
            })
    
    return data, baseline_data

def create_plot(data, baseline_data, output_dir: Path, save_as_pdf=False, font_size=10):
    """Create separate accuracy and monitorability plots."""
    if not data:
        print("No data found!")
        return
    
    # Prepare data for plotting
    x_labels = []
    reasoner_accs = []
    gpt4o_mini_accs = []
    gpt4o_accs = []
    
    # Extract error bars
    reasoner_errors = []
    gpt4o_mini_errors = []
    gpt4o_errors = []
    
    reasoner_significance = []
    gpt4o_mini_significance = []
    gpt4o_significance = []

    baseline_reasoner = baseline_data['reasoner_accuracy'] if baseline_data else None
    baseline_monitors = baseline_data['monitors'] if baseline_data else {}

    for item in data:
        x_labels.append(item['name'])
        reasoner_val = item['reasoner_accuracy']
        if baseline_reasoner is not None:
            reasoner_val = reasoner_val - baseline_reasoner
        reasoner_accs.append(reasoner_val)
        reasoner_significance.append(bool(item.get('reasoner_significant', False)))
        
        # Get monitor accuracies
        gpt4o_mini_acc = None
        gpt4o_acc = None
        gpt4o_mini_error = 0
        gpt4o_error = 0
        gpt4o_mini_sig = False
        gpt4o_sig = False
        
        for monitor_name, monitor_data in item['monitors'].items():
            if monitor_name == 'gpt-4o-mini':
                monitor_val = monitor_data['accuracy']
                baseline_val = baseline_monitors.get(monitor_name, {}).get('accuracy') if baseline_monitors else None
                if monitor_val is not None and baseline_val is not None:
                    monitor_val = monitor_val - baseline_val
                gpt4o_mini_acc = monitor_val
                gpt4o_mini_error = monitor_data['ci_margin'] or 0
                gpt4o_mini_sig = bool(monitor_data.get('significant', False))
            elif monitor_name == 'gpt-4o':
                monitor_val = monitor_data['accuracy']
                baseline_val = baseline_monitors.get(monitor_name, {}).get('accuracy') if baseline_monitors else None
                if monitor_val is not None and baseline_val is not None:
                    monitor_val = monitor_val - baseline_val
                gpt4o_acc = monitor_val
                gpt4o_error = monitor_data['ci_margin'] or 0
                gpt4o_sig = bool(monitor_data.get('significant', False))
        
        gpt4o_mini_accs.append(gpt4o_mini_acc)
        gpt4o_accs.append(gpt4o_acc)
        reasoner_errors.append(item['reasoner_accuracy_ci']['margin'] or 0)
        gpt4o_mini_errors.append(gpt4o_mini_error)
        gpt4o_errors.append(gpt4o_error)
        gpt4o_mini_significance.append(gpt4o_mini_sig)
        gpt4o_significance.append(gpt4o_sig)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setting = data[0].get('setting', 'unknown') if data else 'unknown'
    setting_first_word = setting.split()[0] if setting else 'unknown'
    output_subdir = output_dir / 'plots' / f"{setting_first_word}_main_{timestamp}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Get seaborn pastel colors
    colors = sns.color_palette("pastel", 3)
    
    # Replace None with NaN so bars simply don't render for missing values
    def safe_vals(vals):
        return [v if isinstance(v, (int, float)) else np.nan for v in vals]

    def compute_y_lim(values, errors):
        pairs = []
        for v, e in zip(values, errors):
            if v is None or np.isnan(v):
                continue
            err = e or 0
            pairs.append((v - err, v + err))
        if not pairs:
            return (-0.05, 0.05)
        min_val = min(lo for lo, _ in pairs)
        max_val = max(hi for _, hi in pairs)
        if min_val == max_val:
            min_val -= 0.01
            max_val += 0.01
        padding = 0.01
        return (min_val - padding, max_val + padding)
    
    reasoner_vals = safe_vals(reasoner_accs)
    mini_vals = safe_vals(gpt4o_mini_accs)
    gpt4o_vals = safe_vals(gpt4o_accs)
    
    x_pos = np.arange(len(x_labels))
    
    # Create accuracy plot
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.0))
    width = 0.4  # Original width
    x_pos_accuracy = np.arange(len(x_labels)) * 0.5  # Much more compact spacing for accuracy plot
    
    bars1 = ax1.bar(x_pos_accuracy, reasoner_vals, width, yerr=reasoner_errors, 
                   label='Reasoner accuracy', alpha=0.9, color=colors[0], capsize=3)
    for bar, sig in zip(bars1, reasoner_significance):
        if sig:
            bar.set_edgecolor('black')
            bar.set_linewidth(1.6)
        else:
            bar.set_edgecolor('none')
    
    ax1.set_ylabel('Accuracy change vs baseline', fontsize=font_size)
    ax1.set_xticks(x_pos_accuracy)
    ax1.set_xticklabels(x_labels, rotation=15, ha='center', fontsize=font_size)
    
    accuracy_patch = Patch(facecolor=colors[0], edgecolor='none', alpha=0.9, label='Reasoner accuracy Δ')
    ax1.legend([accuracy_patch], ['Reasoner accuracy Δ'], loc='upper left', fontsize=font_size)
    ax1.set_axisbelow(True)
    ax1.grid(True, which='major', axis='y', alpha=0.8, linestyle='-', linewidth=0.8, zorder=0)
    ax1.grid(True, which='major', axis='x', alpha=0.8, linestyle='-', linewidth=0.8, zorder=0)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', axis='y', alpha=0.5, linestyle=':', linewidth=0.5, zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ymin, ymax = compute_y_lim(reasoner_vals, reasoner_errors)
    ax1.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    file_ext = '.pdf' if save_as_pdf else '.png'
    accuracy_path = output_subdir / f'accuracy_plot{file_ext}'
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create monitorability plot
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.0))
    
    bars2 = ax2.bar(x_pos - width/2, mini_vals, width, yerr=gpt4o_mini_errors, 
                   label='GPT-4o mini', alpha=0.9, color=colors[1], capsize=3)
    bars3 = ax2.bar(x_pos + width/2, gpt4o_vals, width, yerr=gpt4o_errors, 
                   label='GPT-4o', alpha=0.9, color=colors[0], capsize=3)
    for bars, sigs in ((bars2, gpt4o_mini_significance), (bars3, gpt4o_significance)):
        for bar, sig in zip(bars, sigs):
            if sig:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.6)
            else:
                bar.set_edgecolor('none')

    # Draw shared boundaries if at least one adjacent bar is significant
    for idx, center in enumerate(x_pos):
        left_sig = gpt4o_mini_significance[idx] and not np.isnan(mini_vals[idx])
        right_sig = gpt4o_significance[idx] and not np.isnan(gpt4o_vals[idx])
        if not (left_sig or right_sig):
            continue
        heights = []
        if left_sig and not np.isnan(mini_vals[idx]):
            heights.append(mini_vals[idx])
        if right_sig and not np.isnan(gpt4o_vals[idx]):
            heights.append(gpt4o_vals[idx])
        if not heights:
            continue
        max_height = np.nanmax(heights)
        min_height = np.nanmin(heights)
        if np.isnan(max_height) or np.isnan(min_height):
            continue
        top = max(0, max_height)
        bottom = min(0, min_height)
        ax2.vlines(center, bottom, top, colors='black', linewidth=1.6, zorder=4)
    
    ax2.set_ylabel('Monitorability change vs baseline', fontsize=font_size)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=15, ha='center', fontsize=font_size)
    
    # Legend (monitor colors)
    monitor_handles = [
        Patch(facecolor=colors[1], edgecolor='none', alpha=0.9, label='GPT-4o mini'),
        Patch(facecolor=colors[0], edgecolor='none', alpha=0.9, label='GPT-4o')
    ]
    ax2.legend(monitor_handles, ['GPT-4o mini', 'GPT-4o'], loc='upper left', fontsize=font_size)
    ax2.set_axisbelow(True)
    ax2.grid(True, which='major', axis='y', alpha=0.8, linestyle='-', linewidth=0.8, zorder=0)
    ax2.grid(True, which='major', axis='x', alpha=0.8, linestyle='-', linewidth=0.8, zorder=0)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', axis='y', alpha=0.5, linestyle=':', linewidth=0.5, zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    mon_min_1, mon_max_1 = compute_y_lim(mini_vals, gpt4o_mini_errors)
    mon_min_2, mon_max_2 = compute_y_lim(gpt4o_vals, gpt4o_errors)
    ymin_mon = min(mon_min_1, mon_min_2)
    ymax_mon = max(mon_max_1, mon_max_2)
    ax2.set_ylim(ymin_mon, ymax_mon)
    
    plt.tight_layout()
    monitorability_path = output_subdir / f'monitorability_plot{file_ext}'
    plt.savefig(monitorability_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data as JSON
    data_json_path = output_subdir / 'accuracy_plot_data.json'
    with open(data_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Accuracy plot saved as {accuracy_path}")
    print(f"Monitorability plot saved as {monitorability_path}")
    print(f"Data JSON saved as {data_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create accuracy plot from YAML config.")
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
    
    data, baseline_data = extract_accuracy_data(config_path)
    
    config = load_config(config_path)
    save_as_pdf = config.get('save_as_pdf', False)
    font_size = config.get('font_size', 10)
    
    create_plot(data, baseline_data, output_dir, save_as_pdf, font_size)
