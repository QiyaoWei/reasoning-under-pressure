#!/usr/bin/env python3
"""
Script to create a plot showing different incentives and their corresponding accuracies.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator

def extract_accuracy_data():
    """Extract accuracy data from all result directories.

    - Model (Reasoner) accuracy: metrics.measurement-wise_accuracy_overall.value if present,
      otherwise measurement-wise_accuracy_overall.value at root.
    - Monitor accuracies: from any file matching "*_monitor_predictions_summary.json";
      monitor name is parsed from filename prefix before "_monitor_predictions".
    """
    results_dir = Path("/root/MATS/results")
    data = []
    
    # Define the incentives and their directories
    incentives = {
        "regular_RL": "Regular RL",
        "reward_verbosity": "Reward Verbosity", 
        "penalize_verbosity": "Penalize Verbosity",
        "reward_kl": "Reward KL",
        "penalize_kl": "Penalize KL", 
        "penalize_monitorability": "Penalize Monitorability"
    }
    
    for incentive_dir, incentive_name in incentives.items():
        incentive_path = results_dir / incentive_dir
        
        if not incentive_path.exists():
            print(f"Warning: {incentive_path} does not exist")
            continue
        
        print(f"Processing {incentive_name} at {incentive_path}")
            
        # Find all hyperparameter directories
        for hyperparam_dir in incentive_path.iterdir():
            if not hyperparam_dir.is_dir():
                continue
            
            print(f"  Found hyperparam dir: {hyperparam_dir.name}")
                
            # Find all step directories
            for step_dir in hyperparam_dir.iterdir():
                if not step_dir.is_dir():
                    continue
                
                print(f"    Found step dir: {step_dir.name}")
                    
                predictions_dir = step_dir / "predictions"
                if not predictions_dir.exists():
                    print(f"      No predictions dir found")
                    continue
                
                print(f"      Processing predictions in {predictions_dir}")
                
                # Extract model accuracy from predictions.json
                model_accuracy = None
                model_ci_lower = None
                model_ci_upper = None
                model_ci_margin = None
                for pred_file in predictions_dir.glob("*_predictions.json"):
                    if "monitor" in pred_file.name:
                        continue
                    try:
                        with open(pred_file, 'r') as f:
                            pred_data = json.load(f)
                            # prefer metrics.measurement-wise_accuracy_overall.value
                            metrics = pred_data.get("metrics", {}) if isinstance(pred_data, dict) else {}
                            if isinstance(metrics, dict) and "measurement-wise_accuracy_overall" in metrics:
                                mwao = metrics["measurement-wise_accuracy_overall"]
                                if isinstance(mwao, dict) and "value" in mwao:
                                    model_accuracy = mwao["value"]
                                    ci = mwao.get("confidence_interval")
                                    if isinstance(ci, list) and len(ci) >= 3:
                                        model_ci_lower, model_ci_upper, model_ci_margin = ci[0], ci[1], ci[2]
                                    break
                            # fallback: root key
                            if "measurement-wise_accuracy_overall" in pred_data:
                                val = pred_data["measurement-wise_accuracy_overall"]
                                if isinstance(val, dict) and "value" in val:
                                    model_accuracy = val["value"]
                                    ci = val.get("confidence_interval")
                                    if isinstance(ci, list) and len(ci) >= 3:
                                        model_ci_lower, model_ci_upper, model_ci_margin = ci[0], ci[1], ci[2]
                                elif isinstance(val, (int, float)):
                                    model_accuracy = float(val)
                                break
                    except Exception as e:
                        print(f"Error reading {pred_file}: {e}")

                # Extract monitor accuracies generically
                monitor_name_to_accuracy = {}
                monitor_name_to_ci = {}
                for summary_file in predictions_dir.glob("*_monitor_predictions_summary.json"):
                    try:
                        with open(summary_file, 'r') as f:
                            sdata = json.load(f)
                        acc = sdata.get("accuracy")
                        # monitor name is filename prefix before _monitor_predictions
                        fname = summary_file.name
                        monitor_name = fname.split("_monitor_predictions")[0]
                        monitor_name_to_accuracy[monitor_name] = acc
                        ci_obj = sdata.get("accuracy_95_ci")
                        if isinstance(ci_obj, dict):
                            monitor_name_to_ci[monitor_name] = {
                                "lower": ci_obj.get("lower_bound"),
                                "upper": ci_obj.get("upper_bound"),
                                "margin": ci_obj.get("margin"),
                            }
                    except Exception as e:
                        print(f"Error reading {summary_file}: {e}")

                # Only add data if we have at least the model accuracy
                if model_accuracy is not None:
                    print(f"        Found model accuracy: {model_accuracy}")
                    hyperparam_name = hyperparam_dir.name
                    step_name = step_dir.name

                    # retain specific keys for expected legend entries if present
                    gpt4o_mini_accuracy = monitor_name_to_accuracy.get("gpt-4o-mini")
                    gpt4o_accuracy = monitor_name_to_accuracy.get("gpt-4o")
                    gpt4o_mini_ci = monitor_name_to_ci.get("gpt-4o-mini")
                    gpt4o_ci = monitor_name_to_ci.get("gpt-4o")

                    data.append({
                        'incentive': incentive_name,
                        'hyperparameter': hyperparam_name,
                        'step': step_name,
                        'model_accuracy': model_accuracy,
                        'model_accuracy_ci': {
                            'lower': model_ci_lower,
                            'upper': model_ci_upper,
                            'margin': model_ci_margin,
                        },
                        'gpt4o_mini_accuracy': gpt4o_mini_accuracy,
                        'gpt4o_mini_accuracy_ci': gpt4o_mini_ci,
                        'gpt4o_accuracy': gpt4o_accuracy,
                        'gpt4o_accuracy_ci': gpt4o_ci,
                        'full_name': f"{incentive_name}\n({hyperparam_name})"
                    })

            # Handle layout without hyperparameter layer (e.g., Regular RL: incentive/step/predictions)
            # If a direct child contains a predictions dir, treat it as a step
            if (hyperparam_dir / "predictions").exists():
                # Only include step_200 for Regular RL baseline
                if incentive_name == "Regular RL" and hyperparam_dir.name != "step_200":
                    continue
                predictions_dir = hyperparam_dir / "predictions"
                model_accuracy = None
                model_ci_lower = None
                model_ci_upper = None
                model_ci_margin = None
                for pred_file in predictions_dir.glob("*_predictions.json"):
                    if "monitor" in pred_file.name:
                        continue
                    try:
                        with open(pred_file, 'r') as f:
                            pred_data = json.load(f)
                            metrics = pred_data.get("metrics", {}) if isinstance(pred_data, dict) else {}
                            if isinstance(metrics, dict) and "measurement-wise_accuracy_overall" in metrics:
                                mwao = metrics["measurement-wise_accuracy_overall"]
                                if isinstance(mwao, dict) and "value" in mwao:
                                    model_accuracy = mwao["value"]
                                    ci = mwao.get("confidence_interval")
                                    if isinstance(ci, list) and len(ci) >= 3:
                                        model_ci_lower, model_ci_upper, model_ci_margin = ci[0], ci[1], ci[2]
                                    break
                            if "measurement-wise_accuracy_overall" in pred_data:
                                val = pred_data["measurement-wise_accuracy_overall"]
                                if isinstance(val, dict) and "value" in val:
                                    model_accuracy = val["value"]
                                    ci = val.get("confidence_interval")
                                    if isinstance(ci, list) and len(ci) >= 3:
                                        model_ci_lower, model_ci_upper, model_ci_margin = ci[0], ci[1], ci[2]
                                elif isinstance(val, (int, float)):
                                    model_accuracy = float(val)
                                break
                    except Exception as e:
                        print(f"Error reading {pred_file}: {e}")

                monitor_name_to_accuracy = {}
                monitor_name_to_ci = {}
                for summary_file in predictions_dir.glob("*_monitor_predictions_summary.json"):
                    try:
                        with open(summary_file, 'r') as f:
                            sdata = json.load(f)
                        acc = sdata.get("accuracy")
                        fname = summary_file.name
                        monitor_name = fname.split("_monitor_predictions")[0]
                        monitor_name_to_accuracy[monitor_name] = acc
                        ci_obj = sdata.get("accuracy_95_ci")
                        if isinstance(ci_obj, dict):
                            monitor_name_to_ci[monitor_name] = {
                                "lower": ci_obj.get("lower_bound"),
                                "upper": ci_obj.get("upper_bound"),
                                "margin": ci_obj.get("margin"),
                            }
                    except Exception as e:
                        print(f"Error reading {summary_file}: {e}")

                if model_accuracy is not None:
                    step_name = hyperparam_dir.name
                    gpt4o_mini_accuracy = monitor_name_to_accuracy.get("gpt-4o-mini")
                    gpt4o_accuracy = monitor_name_to_accuracy.get("gpt-4o")
                    gpt4o_mini_ci = monitor_name_to_ci.get("gpt-4o-mini")
                    gpt4o_ci = monitor_name_to_ci.get("gpt-4o")
                    data.append({
                        'incentive': incentive_name,
                        'hyperparameter': step_name,
                        'step': step_name,
                        'model_accuracy': model_accuracy,
                        'model_accuracy_ci': {
                            'lower': model_ci_lower,
                            'upper': model_ci_upper,
                            'margin': model_ci_margin,
                        },
                        'gpt4o_mini_accuracy': gpt4o_mini_accuracy,
                        'gpt4o_mini_accuracy_ci': gpt4o_mini_ci,
                        'gpt4o_accuracy': gpt4o_accuracy,
                        'gpt4o_accuracy_ci': gpt4o_ci,
                        'full_name': f"{incentive_name}\n({step_name})"
                    })
    
    return data

def create_plot(data, output_dir: Path):
    """Create the accuracy plot as a bar chart with error bars, and save JSONs."""
    if not data:
        print("No data found!")
        return
    
    # Group data by incentive
    incentives = {}
    for item in data:
        incentive = item['incentive']
        if incentive not in incentives:
            incentives[incentive] = []
        incentives[incentive].append(item)
    
    # Prepare data for plotting
    x_labels = []
    model_accs = []
    gpt4o_mini_accs = []
    gpt4o_accs = []
    
    # Sort incentives for consistent ordering
    incentive_order = ["Regular RL", "Reward Verbosity", "Penalize Verbosity", 
                      "Reward KL", "Penalize KL", "Penalize Monitorability"]
    
    for incentive in incentive_order:
        if incentive in incentives:
            items = incentives[incentive]
            # Sort by hyperparameter name for consistent ordering
            items.sort(key=lambda x: x['hyperparameter'])
            
            for item in items:
                x_labels.append(item['full_name'])
                model_accs.append(item['model_accuracy'])
                gpt4o_mini_accs.append(item['gpt4o_mini_accuracy'])
                gpt4o_accs.append(item['gpt4o_accuracy'])
    
    # Order items consistently for plotting
    ordered_items = []
    for incentive in incentive_order:
        if incentive in incentives:
            items = sorted(incentives[incentive], key=lambda x: x['hyperparameter'])
            for item in items:
                ordered_items.append(item)

    # Create the plot (bar chart without error bars)
    fig, ax = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(x_labels))
    width = 0.25

    # Replace None with NaN so bars simply don't render for missing values
    def safe_vals(vals):
        return [v if isinstance(v, (int, float)) else np.nan for v in vals]

    model_vals = safe_vals(model_accs)
    mini_vals = safe_vals(gpt4o_mini_accs)
    gpt4o_vals = safe_vals(gpt4o_accs)

    # Bars
    bars1 = ax.bar(x_pos - width, model_vals, width, label='Reasoner Accuracy', alpha=0.9, color='#1f77b4')
    bars2 = ax.bar(x_pos, mini_vals, width, label='Monitorability (gpt-4o-mini)', alpha=0.9, color='#ff7f0e')
    bars3 = ax.bar(x_pos + width, gpt4o_vals, width, label='Monitorability (gpt-4o)', alpha=0.9, color='#2ca02c')
    
    # Customize the plot
    ax.set_xlabel('Incentive (Hyperparameter)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model and Monitor Accuracies by Incentive', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    # Fine-grained grid and y-axis limits
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, which='major', axis='y', alpha=0.35, linestyle='-')
    ax.minorticks_on()
    ax.grid(True, which='minor', axis='y', alpha=0.15, linestyle=':')
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Optional: annotate numeric values above bars (commented out to reduce clutter)
    # for bar, val in zip(bars1, model_accs):
    #     if val is not None:
    #         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=8, color='#1f77b4')
    # for bar, val in zip(bars2, gpt4o_mini_accs):
    #     if val is not None:
    #         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=8, color='#ff7f0e')
    # for bar, val in zip(bars3, gpt4o_accs):
    #     if val is not None:
    #         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=8, color='#2ca02c')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'accuracy_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    try:
        plt.close(fig)
    except Exception:
        pass

    # Save data and summary as JSON
    data_json_path = output_dir / 'accuracy_plot_data.json'
    with open(data_json_path, 'w') as f:
        json.dump(data, f, indent=2)

    summary = {}
    for item in data:
        entry = {
            'reasoner_accuracy': item.get('model_accuracy'),
            'reasoner_accuracy_ci': item.get('model_accuracy_ci'),
        }
        if item.get('gpt4o_mini_accuracy') is not None:
            entry['monitorability_gpt4o_mini'] = item['gpt4o_mini_accuracy']
            entry['monitorability_gpt4o_mini_ci'] = item.get('gpt4o_mini_accuracy_ci')
        if item.get('gpt4o_accuracy') is not None:
            entry['monitorability_gpt4o'] = item['gpt4o_accuracy']
            entry['monitorability_gpt4o_ci'] = item.get('gpt4o_accuracy_ci')
        summary[item['full_name']] = entry

    summary_json_path = output_dir / 'accuracy_plot_summary.json'
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary to console
    print("\nData Summary:")
    print("=" * 80)
    for name, entry in summary.items():
        print(name + ":")
        print(f"  Reasoner: {entry['reasoner_accuracy']:.4f}  CI: {entry['reasoner_accuracy_ci']}")
        if 'monitorability_gpt4o_mini' in entry:
            print(f"  Monitorability (gpt-4o-mini): {entry['monitorability_gpt4o_mini']:.4f}  CI: {entry.get('monitorability_gpt4o_mini_ci')}")
        if 'monitorability_gpt4o' in entry:
            print(f"  Monitorability (gpt-4o): {entry['monitorability_gpt4o']:.4f}  CI: {entry.get('monitorability_gpt4o_ci')}")
        print()

    print(f"Plot saved as {plot_path}")
    print(f"Data JSON saved as {data_json_path}")
    print(f"Summary JSON saved as {summary_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create accuracy plot and export data.")
    parser.add_argument("--output-dir", type=str, default="/root/MATS/results", help="Directory to save the plot and JSON outputs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("Extracting accuracy data...")
    data = extract_accuracy_data()
    print(f"Found {len(data)} data points")
    
    print("Creating plot...")
    create_plot(data, output_dir)
