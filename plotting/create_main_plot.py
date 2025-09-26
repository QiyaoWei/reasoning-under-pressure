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
            monitor_data[monitor_name] = monitor_acc
        
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
                'monitors': monitor_data
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
            monitor_data[monitor_name] = monitor_acc
        
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
                'monitors': monitor_data
            })
    
    return data, baseline_data

def create_plot(data, baseline_data, output_dir: Path):
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
    
    for item in data:
        x_labels.append(item['name'])
        reasoner_accs.append(item['reasoner_accuracy'])
        
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
    output_subdir = output_dir / 'plots' / f"{setting_first_word}_main_{timestamp}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Get seaborn pastel colors
    colors = sns.color_palette("pastel", 3)
    
    # Replace None with NaN so bars simply don't render for missing values
    def safe_vals(vals):
        return [v if isinstance(v, (int, float)) else np.nan for v in vals]
    
    reasoner_vals = safe_vals(reasoner_accs)
    mini_vals = safe_vals(gpt4o_mini_accs)
    gpt4o_vals = safe_vals(gpt4o_accs)
    
    x_pos = np.arange(len(x_labels))
    
    # Create accuracy plot
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    width = 0.4  # Original width
    x_pos_accuracy = np.arange(len(x_labels)) * 0.5  # Much more compact spacing for accuracy plot
    
    bars1 = ax1.bar(x_pos_accuracy, reasoner_vals, width, yerr=reasoner_errors, 
                   label='Reasoner accuracy', alpha=0.9, color=colors[2], capsize=3)
    
    # Add baseline line for accuracy
    baseline_handles = []
    baseline_labels = []
    if baseline_data and baseline_data['reasoner_accuracy'] is not None:
        line1 = ax1.axhline(y=baseline_data['reasoner_accuracy'], color=_darker_color(colors[2], 0.8), linestyle='--', 
                           linewidth=2, alpha=0.8)
        baseline_handles.append(line1)
        baseline_labels.append('RL baseline reasoner accuracy')
        # Add CI shading around baseline line if available
        ci_info = baseline_data.get('reasoner_accuracy_ci', {})
        ci_lower = ci_info.get('lower')
        ci_upper = ci_info.get('upper')
        ci_margin = ci_info.get('margin')
        baseline_mean = baseline_data['reasoner_accuracy']
        if (ci_lower is None or ci_upper is None) and ci_margin is not None:
            ci_lower = baseline_mean - ci_margin
            ci_upper = baseline_mean + ci_margin
        if ci_lower is not None and ci_upper is not None:
            # Shade full width of axis so it matches baseline line extent
            ax1.axhspan(ci_lower, ci_upper, xmin=0, xmax=1, color=colors[2], alpha=0.25, zorder=0)
    
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xticks(x_pos_accuracy)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    # Ensure only a small margin on each side (quarter step for accuracy plot)
    if len(x_pos_accuracy) > 0:
        ax1.set_xlim(x_pos_accuracy[0] - 0.3, x_pos_accuracy[-1] + 0.3)
    
    # Create two separate legends for accuracy plot
    if baseline_handles:
        # Baseline legend (top right)
        legend1 = ax1.legend(baseline_handles, baseline_labels, loc='upper right', fontsize=11)
        ax1.add_artist(legend1)
    
    # Experimental runs legend (top left)
    legend2 = ax1.legend([bars1], ['Reasoner accuracy'], loc='upper left', fontsize=11)
    
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, which='major', axis='y', alpha=0.8, linestyle='-', linewidth=0.8)
    ax1.grid(True, which='major', axis='x', alpha=0.8, linestyle='-', linewidth=0.8)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', axis='y', alpha=0.5, linestyle=':', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    accuracy_path = output_subdir / 'accuracy_plot.png'
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create monitorability plot
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    bars2 = ax2.bar(x_pos - width/2, mini_vals, width, yerr=gpt4o_mini_errors, 
                   label='GPT-4o mini', alpha=0.9, color=colors[1], capsize=3)
    bars3 = ax2.bar(x_pos + width/2, gpt4o_vals, width, yerr=gpt4o_errors, 
                   label='GPT-4o', alpha=0.9, color=colors[0], capsize=3)
    
    # Add baseline lines for monitorability
    baseline_handles = []
    baseline_labels = []
    if baseline_data:
        baseline_mini = None
        baseline_gpt4o = None
        baseline_mini_ci = None
        baseline_gpt4o_ci = None
        
        for monitor_name, monitor_data in baseline_data['monitors'].items():
            if monitor_name == 'gpt-4o-mini':
                baseline_mini = monitor_data['accuracy']
                baseline_mini_ci = {
                    'lower': monitor_data.get('ci_lower'),
                    'upper': monitor_data.get('ci_upper'),
                    'margin': monitor_data.get('ci_margin'),
                }
            elif monitor_name == 'gpt-4o':
                baseline_gpt4o = monitor_data['accuracy']
                baseline_gpt4o_ci = {
                    'lower': monitor_data.get('ci_lower'),
                    'upper': monitor_data.get('ci_upper'),
                    'margin': monitor_data.get('ci_margin'),
                }
        
        if baseline_mini is not None:
            line2 = ax2.axhline(y=baseline_mini, color=_darker_color(colors[1], 0.8), linestyle='--', 
                               linewidth=2, alpha=0.8)
            baseline_handles.append(line2)
            baseline_labels.append('RL baseline monitorability (GPT-4o mini)')
            # Add CI shading for GPT-4o mini baseline
            if baseline_mini_ci is not None:
                ci_lower = baseline_mini_ci.get('lower')
                ci_upper = baseline_mini_ci.get('upper')
                ci_margin = baseline_mini_ci.get('margin')
                if (ci_lower is None or ci_upper is None) and ci_margin is not None:
                    ci_lower = baseline_mini - ci_margin
                    ci_upper = baseline_mini + ci_margin
                if ci_lower is not None and ci_upper is not None:
                    # Shade full width of axis so it matches baseline line extent
                    ax2.axhspan(ci_lower, ci_upper, xmin=0, xmax=1, color=colors[1], alpha=0.25, zorder=0)
        if baseline_gpt4o is not None:
            line3 = ax2.axhline(y=baseline_gpt4o, color=_darker_color(colors[0], 0.8), linestyle='--', 
                               linewidth=2, alpha=0.8)
            baseline_handles.append(line3)
            baseline_labels.append('RL baseline monitorability (GPT-4o)')
            # Add CI shading for GPT-4o baseline
            if baseline_gpt4o_ci is not None:
                ci_lower = baseline_gpt4o_ci.get('lower')
                ci_upper = baseline_gpt4o_ci.get('upper')
                ci_margin = baseline_gpt4o_ci.get('margin')
                if (ci_lower is None or ci_upper is None) and ci_margin is not None:
                    ci_lower = baseline_gpt4o - ci_margin
                    ci_upper = baseline_gpt4o + ci_margin
                if ci_lower is not None and ci_upper is not None:
                    # Shade full width of axis so it matches baseline line extent
                    ax2.axhspan(ci_lower, ci_upper, xmin=0, xmax=1, color=colors[0], alpha=0.25, zorder=0)
    
    ax2.set_ylabel('Monitorability', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    # Ensure only a small margin on each side (half unit for standard bar positions)
    if len(x_pos) > 0:
        ax2.set_xlim(-0.6, len(x_pos) - 0.4)
    
    # Create two separate legends for monitorability plot
    if baseline_handles:
        # Baseline legend (top right)
        legend1 = ax2.legend(baseline_handles, baseline_labels, loc='upper right', fontsize=11)
        ax2.add_artist(legend1)
    
    # Experimental runs legend (top left)
    legend2 = ax2.legend([bars2, bars3], ['GPT-4o mini', 'GPT-4o'], loc='upper left', fontsize=11)
    
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, which='major', axis='y', alpha=0.8, linestyle='-', linewidth=0.8)
    ax2.grid(True, which='major', axis='x', alpha=0.8, linestyle='-', linewidth=0.8)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', axis='y', alpha=0.5, linestyle=':', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    monitorability_path = output_subdir / 'monitorability_plot.png'
    plt.savefig(monitorability_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data as JSON
    data_json_path = output_subdir / 'plot_data.json'
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
    create_plot(data, baseline_data, output_dir)
