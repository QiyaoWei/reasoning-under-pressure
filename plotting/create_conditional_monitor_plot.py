#!/usr/bin/env python3
"""
Create conditional monitor plots with multiple monitors on the same plot.
Adapted to show side-by-side bars for different monitors, similar to create_main_plot.py.

Inputs:
- YAML config (same schema as plotting/func_corr/main/config.yaml) or direct summary paths

Outputs:
- results/plots/<dataset_name>/conditional_monitor/<run>/*.png (combined plots)
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_name_from_any(paths: List[Path]) -> str:
    for p in paths:
        try:
            d = load_json(p)
            ds = d.get('dataset_name')
            if isinstance(ds, str) and ds:
                return ds
        except Exception:
            continue
    return 'unknown_dataset'


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_block(d: Dict[str, Any]) -> Tuple[float, float]:
    acc = float(d.get('accuracy', 0.0))
    ci = d.get('95_ci') or {}
    margin = float(ci.get('margin', 0.0)) if isinstance(ci, dict) else 0.0
    return acc, margin


def extract_total(d: Dict[str, Any]) -> int:
    try:
        return int(d.get('total_samples', 0))
    except Exception:
        return 0


def extract_overall(summary: Dict[str, Any]) -> Tuple[float, float, float]:
    acc = float(summary.get('accuracy', 0.0))
    ci = summary.get('accuracy_95_ci') or {}
    lower = float(ci.get('lower_bound')) if isinstance(ci, dict) and 'lower_bound' in ci else None
    upper = float(ci.get('upper_bound')) if isinstance(ci, dict) and 'upper_bound' in ci else None
    return acc, lower, upper


def darker_color(color, factor: float = 0.8):
    try:
        import matplotlib.colors as mcolors
        rgba = mcolors.to_rgba(color)
        r, g, b, a = rgba
        return (max(0.0, min(1.0, r * factor)),
                max(0.0, min(1.0, g * factor)),
                max(0.0, min(1.0, b * factor)))
    except Exception:
        return color


def plot_lv_combined(monitor_data: List[Tuple[str, Dict[str, Any]]], out_dir: Path) -> None:
    """Plot LV conditional accuracies for multiple monitors on the same plot."""
    if not monitor_data:
        return
    
    # Extract data for all monitors
    labels = ['LV True', 'LV False']
    monitor_names = []
    all_vals = []  # [monitor][condition]
    all_errs = []  # [monitor][condition]
    all_ns = []    # [monitor][condition]
    
    for monitor_name, summary in monitor_data:
        ca = summary.get('conditional_accuracies', {})
        lv_true = ca.get('when_lv_true', {})
        lv_false = ca.get('when_lv_false', {})
        
        monitor_names.append(monitor_name)
        vals = []
        errs = []
        ns = []
        
        for blk in [lv_true, lv_false]:
            v, m = extract_block(blk)
            vals.append(v)
            errs.append(m)
            ns.append(extract_total(blk))
        
        all_vals.append(vals)
        all_errs.append(errs)
        all_ns.append(ns)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.8 / len(monitor_names)  # Adjust width based on number of monitors
    
    # Custom color assignment: gpt-4o-mini gets orange, gpt-4o gets blue
    palette = sns.color_palette('pastel', 3)
    monitor_colors = []
    for monitor_name in monitor_names:
        if 'mini' in monitor_name.lower():
            monitor_colors.append(palette[1])  # orange
        else:
            monitor_colors.append(palette[0])  # blue
    
    bars = []
    for i, (monitor_name, vals, errs) in enumerate(zip(monitor_names, all_vals, all_errs)):
        x_pos = x + (i - len(monitor_names)/2 + 0.5) * width
        bar = ax.bar(x_pos, vals, width, yerr=errs, capsize=3, 
                    color=monitor_colors[i], alpha=0.95, label=monitor_name)
        bars.append(bar)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Monitor accuracy', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='y', linestyle='-', alpha=0.8)
    
    # Add overall lines for each monitor
    for i, (monitor_name, summary) in enumerate(monitor_data):
        ov, lower, upper = extract_overall(summary)
        line_color = darker_color(monitor_colors[i], 0.8)
        line = ax.axhline(y=ov, color=line_color, linestyle='--', linewidth=2, alpha=0.9)
        if lower is not None and upper is not None:
            ax.axhspan(lower, upper, xmin=0, xmax=1, color=line_color, alpha=0.3, zorder=0)
    
    # Annotate n above bars
    for i, (x_pos, vals, errs, ns) in enumerate(zip(x, zip(*all_vals), zip(*all_errs), zip(*all_ns))):
        for j, (v, e, n) in enumerate(zip(vals, errs, ns)):
            y_text = min(1.0, float(v) + float(e or 0) + 0.03)
            ax.text(x_pos + (j - len(monitor_names)/2 + 0.5) * width, y_text, 
                   f"n={n}", ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(out_dir / 'monitor_accuracy_by_lv.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_reasoner_combined(monitor_data: List[Tuple[str, Dict[str, Any]]], out_dir: Path) -> None:
    """Plot reasoner conditional accuracies for multiple monitors on the same plot."""
    if not monitor_data:
        return
    
    # Extract data for all monitors
    labels = ['Reasoner correct', 'Reasoner incorrect']
    monitor_names = []
    all_vals = []  # [monitor][condition]
    all_errs = []  # [monitor][condition]
    all_ns = []     # [monitor][condition]
    
    for monitor_name, summary in monitor_data:
        ca = summary.get('conditional_accuracies', {})
        rc = ca.get('when_reasoner_correct', {})
        ri = ca.get('when_reasoner_incorrect', {})
        
        monitor_names.append(monitor_name)
        vals = []
        errs = []
        ns = []
        
        for blk in [rc, ri]:
            v, m = extract_block(blk)
            vals.append(v)
            errs.append(m)
            ns.append(extract_total(blk))
        
        all_vals.append(vals)
        all_errs.append(errs)
        all_ns.append(ns)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.8 / len(monitor_names)
    
    # Custom color assignment: gpt-4o-mini gets orange, gpt-4o gets blue
    palette = sns.color_palette('pastel', 3)
    monitor_colors = []
    for monitor_name in monitor_names:
        if 'mini' in monitor_name.lower():
            monitor_colors.append(palette[1])  # orange
        else:
            monitor_colors.append(palette[0])  # blue
    
    bars = []
    for i, (monitor_name, vals, errs) in enumerate(zip(monitor_names, all_vals, all_errs)):
        x_pos = x + (i - len(monitor_names)/2 + 0.5) * width
        bar = ax.bar(x_pos, vals, width, yerr=errs, capsize=3, 
                    color=monitor_colors[i], alpha=0.95, label=monitor_name)
        bars.append(bar)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Monitor accuracy', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='y', linestyle='-', alpha=0.8)
    
    # Add overall lines for each monitor
    for i, (monitor_name, summary) in enumerate(monitor_data):
        ov, lower, upper = extract_overall(summary)
        line_color = darker_color(monitor_colors[i], 0.8)
        line = ax.axhline(y=ov, color=line_color, linestyle='--', linewidth=2, alpha=0.9)
        if lower is not None and upper is not None:
            ax.axhspan(lower, upper, xmin=0, xmax=1, color=line_color, alpha=0.3, zorder=0)
    
    # Annotate n above bars
    for i, (x_pos, vals, errs, ns) in enumerate(zip(x, zip(*all_vals), zip(*all_errs), zip(*all_ns))):
        for j, (v, e, n) in enumerate(zip(vals, errs, ns)):
            y_text = min(1.0, float(v) + float(e or 0) + 0.03)
            ax.text(x_pos + (j - len(monitor_names)/2 + 0.5) * width, y_text, 
                   f"n={n}", ha='center', va='bottom', fontsize=9)
    
    ax.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(out_dir / 'monitor_accuracy_by_reasoner_correctness.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_groups_combined(monitor_data: List[Tuple[str, Dict[str, Any]]], out_dir: Path) -> None:
    """Plot group conditional accuracies for multiple monitors on the same plot.
    Bars are arranged as: Overall(mon1), Overall(mon2), RC(mon1), RC(mon2), RI(mon1), RI(mon2)
    """
    if not monitor_data:
        return
    
    # Extract data for all monitors
    groups = ['all_false_lv_false', 'mixed_lv_false', 'all_true_lv_true', 'all_true_lv_false']
    monitor_names = []
    all_group_data = []  # [monitor][group][overall/rc/ri]
    
    for monitor_name, summary in monitor_data:
        ca = summary.get('conditional_accuracies', {})
        monitor_names.append(monitor_name)
        group_data = []
        
        for gid in groups:
            g = ca.get(gid)
            if not isinstance(g, dict):
                group_data.append(None)
                continue
            
            # Extract overall, rc, ri data for this group
            overall_v, overall_m = extract_block(g)
            overall_n = extract_total(g)
            
            rc_v, rc_m = extract_block(g.get('when_reasoner_correct', {}))
            rc_n = extract_total(g.get('when_reasoner_correct', {}))
            
            ri_v, ri_m = extract_block(g.get('when_reasoner_incorrect', {}))
            ri_n = extract_total(g.get('when_reasoner_incorrect', {}))
            
            group_data.append({
                'overall': (overall_v, overall_m, overall_n),
                'rc': (rc_v, rc_m, rc_n),
                'ri': (ri_v, ri_m, ri_n)
            })
        
        all_group_data.append(group_data)
    
    # Filter out groups that have no data for any monitor
    valid_groups = []
    valid_indices = []
    for i, gid in enumerate(groups):
        if any(all_group_data[j][i] is not None for j in range(len(monitor_names))):
            valid_groups.append(gid)
            valid_indices.append(i)
    
    if not valid_groups:
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(valid_groups))
    # Calculate width: we have 6 bars per group (2 monitors × 3 conditions)
    width = 0.8 / (len(monitor_names) * 3)  # 3 conditions per monitor
    
    # Reduce space on sides by adjusting x limits
    ax.set_xlim(-0.2, len(valid_groups) - 0.15)
    
    # Custom color assignment: gpt-4o-mini gets orange, gpt-4o gets blue
    palette = sns.color_palette('pastel', 3)
    monitor_colors = []
    for monitor_name in monitor_names:
        if 'mini' in monitor_name.lower():
            monitor_colors.append(palette[1])  # orange
        else:
            monitor_colors.append(palette[0])  # blue
    
    # Plot bars alternating between monitors for each condition
    conditions = [('Overall', 'overall'), ('RC', 'rc'), ('RI', 'ri')]
    
    for cond_idx, (cond_name, cond_key) in enumerate(conditions):
        for monitor_idx, (monitor_name, group_data) in enumerate(zip(monitor_names, all_group_data)):
            vals = []
            errs = []
            ns = []
            
            for group_idx in valid_indices:
                if group_data[group_idx] is not None:
                    v, m, n = group_data[group_idx][cond_key]
                    vals.append(v)
                    errs.append(m)
                    ns.append(n)
                else:
                    vals.append(0)
                    errs.append(0)
                    ns.append(0)
            
            # Calculate x positions: alternate between monitors for each condition
            # Position within the condition group: monitor_idx * width
            # Offset for condition: cond_idx * (len(monitor_names) * width)
            x_pos = x + (cond_idx * len(monitor_names) + monitor_idx) * width
            
            base_color = monitor_colors[monitor_idx]
            
            # Use hatching to distinguish conditions
            hatch = '.' if cond_name == 'RC' else '\\\\' if cond_name == 'RI' else None
            
            bar = ax.bar(x_pos, vals, width, yerr=errs, capsize=3, 
                        color=base_color, alpha=0.95, hatch=hatch,
                        label=f'{monitor_name} {cond_name}' if cond_idx == 0 else "")
            
            # Annotate n above bars
            for i, (v, e, n) in enumerate(zip(vals, errs, ns)):
                if n > 0:  # Only annotate if there's data
                    y_text = min(1.0, float(v) + float(e or 0) + 0.03)
                    ax.text(x_pos[i], y_text, f"n={n}", ha='center', va='bottom', fontsize=8)
    
    # Position x-axis ticks in the middle of each group of bars
    # Each group has 6 bars, so middle is at position 2.5 (0-indexed)
    middle_positions = x + (len(monitor_names) * 3 - 1) * width / 2
    ax.set_xticks(middle_positions)
    ax.set_xticklabels(valid_groups, rotation=0, ha='center', fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Monitor accuracy', fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='y', linestyle='-', alpha=0.8)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add overall lines for each monitor
    for i, (monitor_name, summary) in enumerate(monitor_data):
        ov, lower, upper = extract_overall(summary)
        line_color = darker_color(monitor_colors[i], 0.8)
        line = ax.axhline(y=ov, color=line_color, linestyle='--', linewidth=2, alpha=0.9)
        if lower is not None and upper is not None:
            ax.axhspan(lower, upper, xmin=0, xmax=1, color=line_color, alpha=0.3, zorder=0)
    
    # Create custom legend - simplified with just monitor colors and condition indicators
    import matplotlib.patches as mpatches
    legend_elements = []
    
    # Add monitor color patches
    for i, monitor_name in enumerate(monitor_names):
        base_color = monitor_colors[i]
        legend_elements.append(mpatches.Patch(facecolor=base_color, label=monitor_name))
    
    # Add condition indicators with larger rectangles and more prominent hatching
    legend_elements.extend([
        mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', 
                          hatch='...', linewidth=1.5, label='When reasoner correct'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', 
                          hatch='\\\\\\', linewidth=1.5, label='When reasoner incorrect')
    ])
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
              handlelength=3.0, handletextpad=1.0, columnspacing=1.5, 
              frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(out_dir / 'monitor_accuracy_by_group.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create combined conditional monitor plots from YAML or summary paths.')
    parser.add_argument('--dataset-name', type=str, default=None, help='Dataset name to use for output directory.')
    parser.add_argument('--config', type=str, default=None, help='YAML config, same schema as func_corr/main/config.yaml.')
    parser.add_argument('--summaries', type=str, nargs='*', default=None, help='One or more monitor summary JSON paths.')
    parser.add_argument('--output-dir', type=str, default=None, help='Base directory to save plots; default is results/plots/<dataset_name>/conditional_monitor.')
    args = parser.parse_args()

    runs: List[Tuple[str, List[Path]]] = []  # (run_name, [monitor_summary_paths])
    if args.config:
        cfg = load_yaml(Path(args.config))
        # baseline
        mon = (cfg.get('baseline') or {}).get('monitors') or []
        paths = [Path(m['accuracy_path']) for m in mon if isinstance(m, dict) and 'accuracy_path' in m]
        if paths:
            runs.append((cfg.get('baseline', {}).get('name', 'baseline'), paths))
        # others
        for run in cfg.get('runs', []):
            mon = (run or {}).get('monitors') or []
            paths = [Path(m['accuracy_path']) for m in mon if isinstance(m, dict) and 'accuracy_path' in m]
            if paths:
                runs.append((run.get('name', 'run'), paths))
    elif args.summaries:
        runs.append(('manual', [Path(p) for p in args.summaries]))
    else:
        raise ValueError('Provide --config or --summaries.')

    # Determine dataset name
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = get_dataset_name_from_any([p for _, ps in runs for p in ps])
    base_out = Path(args.output_dir) if args.output_dir else (Path('results') / 'plots' / dataset_name / 'conditional_monitor')

    for run_name, monitor_paths in runs:
        # Load all monitor data for this run
        monitor_data = []
        for monitor_path in monitor_paths:
            try:
                summary = load_json(monitor_path)
                monitor_name = monitor_path.stem.replace('_monitor_predictions_summary', '').replace('_', '-')
                monitor_data.append((monitor_name, summary))
            except Exception as e:
                print(f"Failed to load {monitor_path}: {e}")
                continue
        
        if not monitor_data:
            continue
        
        # Create combined plots for this run
        out_dir = base_out / run_name.replace(' ', '_')
        
        plot_lv_combined(monitor_data, out_dir)
        plot_reasoner_combined(monitor_data, out_dir)
        plot_groups_combined(monitor_data, out_dir)
        
        print(f"Created combined conditional monitor plots for {run_name} with {len(monitor_data)} monitors")


if __name__ == '__main__':
    main()
