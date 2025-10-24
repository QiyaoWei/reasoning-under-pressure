#!/usr/bin/env python3
"""
Create conditional monitor plots with multiple monitors on the same plot.
Adapted to show side-by-side bars for different monitors, similar to create_main_plot.py.

Inputs:
- YAML config (same schema as plotting/func_corr/main/config.yaml) or direct summary paths
- Or a predictions directory via --dir that contains monitor summary JSONs

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
from matplotlib.lines import Line2D


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


def read_comparison_for_monitor(summary_path: Path) -> Dict[str, Any]:
    # summary_path -> .../predictions/<name>_monitor_predictions_summary.json
    # comparison expected at .../predictions/compare with baseline/<name>_monitor_predictions_comparison_with_baseline.json
    pred_dir = summary_path.parent
    stem = summary_path.stem  # e.g., gpt_4o_mini_monitor_predictions_summary
    compare_name = stem.replace('_summary', '_comparison_with_baseline') + '.json'
    candidates = [
        pred_dir / 'compare with baseline' / compare_name,
        pred_dir / 'compare_with_baseline' / compare_name,
    ]
    for p in candidates:
        if p.exists():
            try:
                return load_json(p)
            except Exception:
                continue
    return {}


def compute_compare_inline(incentive: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a minimal comparison struct in the same shape used by read_comparison_for_monitor.
    Includes overall and conditional_accuracies with diff and ci margins.
    """
    def ci_obj(summary_ci: Dict[str, Any]) -> Tuple[float, float, float]:
        if not isinstance(summary_ci, dict):
            return None, None, None
        lb = summary_ci.get('lower_bound')
        ub = summary_ci.get('upper_bound')
        margin = summary_ci.get('margin')
        return lb, ub, margin

    def overlap_from_margins(diff_val: float, inc_margin: float, base_margin: float) -> Tuple[float, float, bool]:
        i_low = diff_val - inc_margin
        i_high = diff_val + inc_margin
        b_low = -base_margin
        b_high = base_margin
        low = max(i_low, b_low)
        high = min(i_high, b_high)
        return low, high, (high > low)

    out: Dict[str, Any] = {}
    inc_acc = incentive.get('accuracy')
    base_acc = baseline.get('accuracy')
    inc_lb, inc_ub, inc_m = ci_obj(incentive.get('accuracy_95_ci') or {})
    base_lb, base_ub, base_m = ci_obj(baseline.get('accuracy_95_ci') or {})
    diff_val = (inc_acc - base_acc) if isinstance(inc_acc, (int, float)) and isinstance(base_acc, (int, float)) else None
    overall = {
        'incentive_value': inc_acc,
        'baseline_value': base_acc,
        'diff': diff_val,
        'ci': {
            'incentive_margin': inc_m,
            'baseline_margin': base_m,
            'overlap': None,
            'has_overlap': None,
        },
    }
    # Add counts for overall from incentive summary
    try:
        overall['incentive_correct_predictions'] = int(incentive.get('correct_predictions'))
    except Exception:
        pass
    try:
        overall['incentive_total_samples'] = int(incentive.get('total_samples'))
    except Exception:
        pass
    if isinstance(diff_val, (int, float)) and isinstance(inc_m, (int, float)) and isinstance(base_m, (int, float)):
        low, high, has = overlap_from_margins(diff_val, inc_m, base_m)
        overall['ci']['overlap'] = max(0.0, high - low)
        overall['ci']['has_overlap'] = has
    out['overall'] = overall

    inc_conds = incentive.get('conditional_accuracies', {}) or {}
    base_conds = baseline.get('conditional_accuracies', {}) or {}
    conds_result: Dict[str, Any] = {}

    def make_entry(inc_node: Dict[str, Any], base_node: Dict[str, Any]) -> Dict[str, Any]:
        inc_a = inc_node.get('accuracy')
        base_a = base_node.get('accuracy')
        diff = (inc_a - base_a) if isinstance(inc_a, (int, float)) and isinstance(base_a, (int, float)) else None
        inc_m = ((inc_node.get('95_ci') or {}).get('margin') if isinstance(inc_node.get('95_ci'), dict) else None)
        base_m = ((base_node.get('95_ci') or {}).get('margin') if isinstance(base_node.get('95_ci'), dict) else None)
        entry = {
            'incentive_value': inc_a,
            'baseline_value': base_a,
            'diff': diff,
            'ci': {
                'incentive_margin': inc_m,
                'baseline_margin': base_m,
                'overlap': None,
                'has_overlap': None,
            },
        }
        # Add incentive counts when available
        try:
            entry['incentive_correct_predictions'] = int(inc_node.get('correct_predictions'))
        except Exception:
            pass
        try:
            entry['incentive_total_samples'] = int(inc_node.get('total_samples'))
        except Exception:
            pass
        if isinstance(diff, (int, float)) and isinstance(inc_m, (int, float)) and isinstance(base_m, (int, float)):
            low, high, has = overlap_from_margins(diff, inc_m, base_m)
            entry['ci']['overlap'] = max(0.0, high - low)
            entry['ci']['has_overlap'] = has
        return entry

    for cond_name, inc_node in inc_conds.items():
        base_node = base_conds.get(cond_name, {}) if isinstance(base_conds, dict) else {}
        if not isinstance(inc_node, dict) or not isinstance(base_node, dict):
            continue
        entry = make_entry(inc_node, base_node)
        # nested
        nested: Dict[str, Any] = {}
        for k in ('when_reasoner_correct', 'when_reasoner_incorrect'):
            if isinstance(inc_node.get(k), dict):
                nested[k] = make_entry(inc_node.get(k), base_node.get(k, {}))
        if nested:
            entry['nested'] = nested
        conds_result[cond_name] = entry
    out['conditional_accuracies'] = conds_result
    return out


def get_compare_for_monitor(summary_path: Path, baseline_dir: Path) -> Dict[str, Any]:
    comp = read_comparison_for_monitor(summary_path)
    if comp:
        return comp
    # Fallback: compute on the fly from baseline dir
    base_path = baseline_dir / summary_path.name
    if base_path.exists():
        try:
            inc = load_json(summary_path)
            base = load_json(base_path)
            return compute_compare_inline(inc, base)
        except Exception:
            return {}
    return {}


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


def plot_lv_combined_compare(monitor_data: List[Tuple[str, Dict[str, Any]]], compare_data: List[Tuple[str, Dict[str, Any]]], out_dir: Path) -> None:
    if not monitor_data or not compare_data:
        return
    labels = ['LV True', 'LV False']
    keys = ['when_lv_true', 'when_lv_false']

    monitor_names = [name for name, _ in monitor_data]
    palette = sns.color_palette('pastel', 3)
    monitor_colors = [(palette[1] if 'mini' in name.lower() else palette[0]) for name in monitor_names]
    baseline_color = (0.60, 0.60, 0.60)
    overlap_colors = [darker_color(c, 0.6) for c in monitor_colors]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.8 / len(monitor_names)

    for i, (name, comp) in enumerate(compare_data):
        vals = []
        inc_margins = []
        base_margins = []
        overlaps = []
        labels_n = []
        labels_pct = []
        conds = comp.get('conditional_accuracies', {}) if isinstance(comp, dict) else {}
        for k in keys:
            node = conds.get(k, {}) if isinstance(conds, dict) else {}
            vals.append(node.get('diff', 0.0))
            ci = node.get('ci', {}) if isinstance(node, dict) else {}
            inc_margins.append(ci.get('incentive_margin', 0.0) or 0.0)
            base_margins.append(ci.get('baseline_margin', 0.0) or 0.0)
            # build incentive count label if present
            inc_correct = node.get('incentive_correct_predictions')
            inc_total = node.get('incentive_total_samples')
            if isinstance(inc_correct, int) and isinstance(inc_total, int) and inc_total > 0:
                labels_n.append(f"{inc_correct}/{inc_total}")
                labels_pct.append(f"{(inc_correct / inc_total) * 100:.1f}%")
            else:
                labels_n.append(None)
                # Try to use incentive_value if provided
                inc_val = node.get('incentive_value')
                if isinstance(inc_val, (int, float)):
                    labels_pct.append(f"{inc_val * 100:.1f}%")
                else:
                    labels_pct.append(None)
            diff_val = vals[-1]
            ovl = None
            if isinstance(diff_val, (int, float)) and isinstance(inc_margins[-1], (int, float)) and isinstance(base_margins[-1], (int, float)):
                i_low = diff_val - inc_margins[-1]
                i_high = diff_val + inc_margins[-1]
                b_low = -base_margins[-1]
                b_high = base_margins[-1]
                low = max(i_low, b_low)
                high = min(i_high, b_high)
                if high > low:
                    ovl = (low, high)
            overlaps.append(ovl)

        x_pos = x + (i - len(monitor_names)/2 + 0.5) * width
        ax.bar(x_pos, vals, width, yerr=inc_margins, capsize=3, color=monitor_colors[i], alpha=0.95, label=name)
        for idx_bar, (xi, bmar) in enumerate(zip(x_pos, base_margins)):
            if bmar and bmar > 0:
                ax.vlines(x=xi, ymin=-bmar, ymax=bmar, colors=baseline_color, linestyles='-', linewidth=3, alpha=0.9)
        for xi, ovl in zip(x_pos, overlaps):
            if ovl is not None:
                ax.vlines(x=xi, ymin=ovl[0], ymax=ovl[1], colors=overlap_colors[i], linestyles='-', linewidth=8, alpha=0.5)
        # annotate incentive accuracy % and counts above bars
        for xi, v, e, pct, label in zip(x_pos, vals, inc_margins, labels_pct, labels_n):
            base_y = float(v) + float(e or 0)
            if pct:
                ax.text(xi, base_y + 0.03, pct, ha='center', va='bottom', fontsize=9)
            if label:
                ax.text(xi, base_y + 0.01, label, ha='center', va='bottom', fontsize=9)

    zero_line = ax.axhline(y=0.0, color=baseline_color, linestyle='--', linewidth=2, alpha=0.8, label='Baseline (0)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    # Dynamic y-limit: ensure full bars and baseline CI are visible
    max_extent = 0.25
    for i, (name, comp) in enumerate(compare_data):
        conds = comp.get('conditional_accuracies', {}) if isinstance(comp, dict) else {}
        for k in keys:
            node = conds.get(k, {}) if isinstance(conds, dict) else {}
            d = node.get('diff', 0.0)
            ci = node.get('ci', {}) if isinstance(node, dict) else {}
            im = ci.get('incentive_margin', 0.0) or 0.0
            bm = ci.get('baseline_margin', 0.0) or 0.0
            try:
                local = max(abs(float(d)) + float(im), float(bm))
                if np.isfinite(local):
                    max_extent = max(max_extent, local)
            except Exception:
                pass
    limit = min(1.0, max_extent + 0.03)
    ax.set_ylim(-limit, limit)
    ax.set_yticks(np.arange(-limit, limit + 1e-6, 0.05))
    ax.set_ylabel('Δ Monitor accuracy (incentive − baseline)', fontsize=14)
    ax.grid(True, axis='y', linestyle='-', alpha=0.8)
    baseline_ci_proxy = Line2D([0], [0], color=baseline_color, linestyle='-', linewidth=3, label='Baseline 95% CI')
    incentive_ci_proxy = Line2D([0], [0], color=[0,0,0], linestyle='-', label='Incentive 95% CI (error bars)')
    overlap_proxy = Line2D([0], [0], color=darker_color(palette[0], 0.6), linestyle='-', linewidth=8, alpha=0.5, label='CI overlap')
    # Monitor color legend
    import matplotlib.patches as mpatches
    monitor_patches = [mpatches.Patch(facecolor=monitor_colors[i], label=monitor_names[i]) for i in range(len(monitor_names))]
    ax.legend(handles=monitor_patches + [zero_line, baseline_ci_proxy, incentive_ci_proxy, overlap_proxy], loc='upper right', fontsize=12)
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(out_dir / 'compare_monitor_accuracy_by_lv.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_reasoner_combined_compare(monitor_data: List[Tuple[str, Dict[str, Any]]], compare_data: List[Tuple[str, Dict[str, Any]]], out_dir: Path) -> None:
    if not monitor_data or not compare_data:
        return
    labels = ['Reasoner correct', 'Reasoner incorrect']
    keys = ['when_reasoner_correct', 'when_reasoner_incorrect']

    monitor_names = [name for name, _ in monitor_data]
    palette = sns.color_palette('pastel', 3)
    monitor_colors = [(palette[1] if 'mini' in name.lower() else palette[0]) for name in monitor_names]
    baseline_color = (0.60, 0.60, 0.60)
    overlap_colors = [darker_color(c, 0.6) for c in monitor_colors]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.8 / len(monitor_names)

    for i, (name, comp) in enumerate(compare_data):
        vals = []
        inc_margins = []
        base_margins = []
        overlaps = []
        labels_n = []
        labels_pct = []
        conds = comp.get('conditional_accuracies', {}) if isinstance(comp, dict) else {}
        for k in keys:
            node = conds.get(k, {}) if isinstance(conds, dict) else {}
            vals.append(node.get('diff', 0.0))
            ci = node.get('ci', {}) if isinstance(node, dict) else {}
            inc_margins.append(ci.get('incentive_margin', 0.0) or 0.0)
            base_margins.append(ci.get('baseline_margin', 0.0) or 0.0)
            inc_correct = node.get('incentive_correct_predictions')
            inc_total = node.get('incentive_total_samples')
            if isinstance(inc_correct, int) and isinstance(inc_total, int) and inc_total > 0:
                labels_n.append(f"{inc_correct}/{inc_total}")
                labels_pct.append(f"{(inc_correct / inc_total) * 100:.1f}%")
            else:
                labels_n.append(None)
                inc_val = node.get('incentive_value')
                if isinstance(inc_val, (int, float)):
                    labels_pct.append(f"{inc_val * 100:.1f}%")
                else:
                    labels_pct.append(None)
            diff_val = vals[-1]
            ovl = None
            if isinstance(diff_val, (int, float)) and isinstance(inc_margins[-1], (int, float)) and isinstance(base_margins[-1], (int, float)):
                i_low = diff_val - inc_margins[-1]
                i_high = diff_val + inc_margins[-1]
                b_low = -base_margins[-1]
                b_high = base_margins[-1]
                low = max(i_low, b_low)
                high = min(i_high, b_high)
                if high > low:
                    ovl = (low, high)
            overlaps.append(ovl)

        x_pos = x + (i - len(monitor_names)/2 + 0.5) * width
        bar = ax.bar(x_pos, vals, width, yerr=inc_margins, capsize=3, color=monitor_colors[i], alpha=0.95, label=name)
        # Apply hatches per condition: RC -> '.', RI -> '\\'
        for b_idx, patch in enumerate(bar):
            patch.set_hatch('.' if b_idx == 0 else '\\\\')
        for xi, bmar in zip(x_pos, base_margins):
            if bmar and bmar > 0:
                ax.vlines(x=xi, ymin=-bmar, ymax=bmar, colors=baseline_color, linestyles='-', linewidth=3, alpha=0.9)
        for xi, ovl in zip(x_pos, overlaps):
            if ovl is not None:
                ax.vlines(x=xi, ymin=ovl[0], ymax=ovl[1], colors=overlap_colors[i], linestyles='-', linewidth=8, alpha=0.5)
        for xi, v, e, pct, label in zip(x_pos, vals, inc_margins, labels_pct, labels_n):
            base_y = float(v) + float(e or 0)
            if pct:
                ax.text(xi, base_y + 0.03, pct, ha='center', va='bottom', fontsize=9)
            if label:
                ax.text(xi, base_y + 0.01, label, ha='center', va='bottom', fontsize=9)

    zero_line = ax.axhline(y=0.0, color=baseline_color, linestyle='--', linewidth=2, alpha=0.8, label='Baseline (0)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    # Dynamic y-limit: ensure full bars and baseline CI are visible
    max_extent = 0.25
    for i, (name, comp) in enumerate(compare_data):
        conds = comp.get('conditional_accuracies', {}) if isinstance(comp, dict) else {}
        for k in keys:
            node = conds.get(k, {}) if isinstance(conds, dict) else {}
            d = node.get('diff', 0.0)
            ci = node.get('ci', {}) if isinstance(node, dict) else {}
            im = ci.get('incentive_margin', 0.0) or 0.0
            bm = ci.get('baseline_margin', 0.0) or 0.0
            try:
                local = max(abs(float(d)) + float(im), float(bm))
                if np.isfinite(local):
                    max_extent = max(max_extent, local)
            except Exception:
                pass
    limit = min(1.0, max_extent + 0.03)
    ax.set_ylim(-limit, limit)
    ax.set_yticks(np.arange(-limit, limit + 1e-6, 0.05))
    ax.set_ylabel('Δ Monitor accuracy (incentive − baseline)', fontsize=14)
    ax.grid(True, axis='y', linestyle='-', alpha=0.8)
    baseline_ci_proxy = Line2D([0], [0], color=baseline_color, linestyle='-', linewidth=3, label='Baseline 95% CI')
    incentive_ci_proxy = Line2D([0], [0], color=[0,0,0], linestyle='-', label='Incentive 95% CI (error bars)')
    overlap_proxy = Line2D([0], [0], color=darker_color(palette[0], 0.6), linestyle='-', linewidth=8, alpha=0.5, label='CI overlap')
    # Legend includes monitor colors and RC/RI hatches
    import matplotlib.patches as mpatches
    monitor_patches = [mpatches.Patch(facecolor=monitor_colors[i], label=monitor_names[i]) for i in range(len(monitor_names))]
    rc_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch='...', linewidth=1.5, label='When reasoner correct')
    ri_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch='\\\\', linewidth=1.5, label='When reasoner incorrect')
    ax.legend(handles=monitor_patches + [rc_patch, ri_patch, zero_line, baseline_ci_proxy, incentive_ci_proxy, overlap_proxy], loc='upper right', fontsize=12)
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(out_dir / 'compare_monitor_accuracy_by_reasoner_correctness.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_groups_combined_compare(monitor_data: List[Tuple[str, Dict[str, Any]]], compare_data: List[Tuple[str, Dict[str, Any]]], out_dir: Path) -> None:
    if not monitor_data or not compare_data:
        return
    groups = ['all_false_lv_false', 'mixed_lv_false', 'all_true_lv_true', 'all_true_lv_false']
    monitor_names = [name for name, _ in monitor_data]
    palette = sns.color_palette('pastel', 3)
    monitor_colors = [(palette[1] if 'mini' in name.lower() else palette[0]) for name in monitor_names]
    baseline_color = (0.60, 0.60, 0.60)
    overlap_colors = [darker_color(c, 0.6) for c in monitor_colors]

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(groups))
    width = 0.8 / (len(monitor_names) * 3)
    ax.set_xlim(-0.2, len(groups) - 0.15)

    conditions = [('Overall', 'overall'), ('RC', 'when_reasoner_correct'), ('RI', 'when_reasoner_incorrect')]

    for cond_idx, (cond_name, cond_key) in enumerate(conditions):
        for monitor_idx, (name, comp) in enumerate(compare_data):
            vals = []
            inc_margins = []
            base_margins = []
            overlaps = []
            labels_n = []
            labels_pct = []
            ca = comp.get('conditional_accuracies', {}) if isinstance(comp, dict) else {}
            for gid in groups:
                node = ca.get(gid, {}) if isinstance(ca, dict) else {}
                if cond_key == 'overall':
                    v = node.get('diff', 0.0)
                    ci = node.get('ci', {}) if isinstance(node, dict) else {}
                else:
                    nested = node.get('nested', {}) if isinstance(node, dict) else {}
                    child = nested.get(cond_key, {}) if isinstance(nested, dict) else {}
                    v = child.get('diff', 0.0)
                    ci = child.get('ci', {}) if isinstance(child, dict) else {}
                vals.append(v)
                inc_m = ci.get('incentive_margin', 0.0) or 0.0
                base_m = ci.get('baseline_margin', 0.0) or 0.0
                inc_margins.append(inc_m)
                base_margins.append(base_m)
                # count label: overall and nested entries carry the counts if present on node/child respectively
                src = node if cond_key == 'overall' else child
                inc_correct = src.get('incentive_correct_predictions') if isinstance(src, dict) else None
                inc_total = src.get('incentive_total_samples') if isinstance(src, dict) else None
                if isinstance(inc_correct, int) and isinstance(inc_total, int) and inc_total > 0:
                    labels_n.append(f"{inc_correct}/{inc_total}")
                    labels_pct.append(f"{(inc_correct / inc_total) * 100:.1f}%")
                else:
                    labels_n.append(None)
                    inc_val = (src.get('incentive_value') if isinstance(src, dict) else None)
                    if isinstance(inc_val, (int, float)):
                        labels_pct.append(f"{inc_val * 100:.1f}%")
                    else:
                        labels_pct.append(None)
                ovl = None
                if isinstance(v, (int, float)) and isinstance(inc_m, (int, float)) and isinstance(base_m, (int, float)):
                    i_low = v - inc_m
                    i_high = v + inc_m
                    b_low = -base_m
                    b_high = base_m
                    low = max(i_low, b_low)
                    high = min(i_high, b_high)
                    if high > low:
                        ovl = (low, high)
                overlaps.append(ovl)

            x_pos = x + (cond_idx * len(monitor_names) + monitor_idx) * width
            base_color = monitor_colors[monitor_idx]
            hatch = '.' if cond_name == 'RC' else '\\\\' if cond_name == 'RI' else None
            ax.bar(x_pos, vals, width, yerr=inc_margins, capsize=3, 
                   color=base_color, alpha=0.95, hatch=hatch,
                   label=f'{name} {cond_name}' if cond_idx == 0 else "")

            for xi, bmar in zip(x_pos, base_margins):
                if bmar and bmar > 0:
                    ax.vlines(x=xi, ymin=-bmar, ymax=bmar, colors=baseline_color, linestyles='-', linewidth=3, alpha=0.9)
            for xi, ovl in zip(x_pos, overlaps):
                if ovl is not None:
                    ax.vlines(x=xi, ymin=ovl[0], ymax=ovl[1], colors=overlap_colors[monitor_idx], linestyles='-', linewidth=8, alpha=0.5)
            for xi, v, e, pct, label in zip(x_pos, vals, inc_margins, labels_pct, labels_n):
                base_y = float(v) + float(e or 0)
                if pct:
                    ax.text(xi, base_y + 0.03, pct, ha='center', va='bottom', fontsize=8)
                if label:
                    ax.text(xi, base_y + 0.01, label, ha='center', va='bottom', fontsize=8)

    middle_positions = x + (len(monitor_names) * 3 - 1) * width / 2
    ax.set_xticks(middle_positions)
    ax.set_xticklabels(groups, rotation=0, ha='center', fontsize=12)
    # Dynamic y-limit: ensure full bars and baseline CI are visible
    max_extent = 0.25
    for cond_idx, (cond_name, cond_key) in enumerate(conditions):
        for monitor_idx, (name, comp) in enumerate(compare_data):
            ca = comp.get('conditional_accuracies', {}) if isinstance(comp, dict) else {}
            for gid in groups:
                node = ca.get(gid, {}) if isinstance(ca, dict) else {}
                if cond_key == 'overall':
                    v = node.get('diff', 0.0)
                    ci = node.get('ci', {}) if isinstance(node, dict) else {}
                else:
                    nested = node.get('nested', {}) if isinstance(node, dict) else {}
                    child = nested.get(cond_key, {}) if isinstance(nested, dict) else {}
                    v = child.get('diff', 0.0)
                    ci = child.get('ci', {}) if isinstance(child, dict) else {}
                im = ci.get('incentive_margin', 0.0) or 0.0
                bm = ci.get('baseline_margin', 0.0) or 0.0
                try:
                    local = max(abs(float(v)) + float(im), float(bm))
                    if np.isfinite(local):
                        max_extent = max(max_extent, local)
                except Exception:
                    pass
    limit = min(1.0, max_extent + 0.03)
    ax.set_ylim(-limit, limit)
    ax.set_yticks(np.arange(-limit, limit + 1e-6, 0.05))
    ax.set_ylabel('Δ Monitor accuracy (incentive − baseline)', fontsize=14)
    ax.grid(True, axis='y', linestyle='-', alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    zero_line = ax.axhline(y=0.0, color=baseline_color, linestyle='--', linewidth=2, alpha=0.8, label='Baseline (0)')
    baseline_ci_proxy = Line2D([0], [0], color=baseline_color, linestyle='-', linewidth=3, label='Baseline 95% CI')
    incentive_ci_proxy = Line2D([0], [0], color=[0,0,0], linestyle='-', label='Incentive 95% CI (error bars)')
    overlap_proxy = Line2D([0], [0], color=darker_color(palette[0], 0.6), linestyle='-', linewidth=8, alpha=0.5, label='CI overlap')
    # Keep legend mapping monitor colors and RC/RI patterns, plus CI proxies
    import matplotlib.patches as mpatches
    legend_elements = []
    for i, name in enumerate(monitor_names):
        legend_elements.append(mpatches.Patch(facecolor=monitor_colors[i], label=name))
    legend_elements.extend([
        mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch='...', linewidth=1.5, label='When reasoner correct'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch='\\\\', linewidth=1.5, label='When reasoner incorrect'),
    ])
    ci_handles = [zero_line, baseline_ci_proxy, incentive_ci_proxy, overlap_proxy]
    ax.legend(handles=legend_elements + ci_handles, loc='upper right', fontsize=12, handlelength=3.0, handletextpad=1.0, frameon=True, fancybox=True)
    plt.tight_layout()
    ensure_dir(out_dir)
    plt.savefig(out_dir / 'compare_monitor_accuracy_by_group.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create combined conditional monitor plots from YAML, summaries, or a predictions directory. Also generates compare (Δ) plots vs baseline with CI overlap visuals.')
    parser.add_argument('--dataset-name', type=str, default=None, help='Dataset name to use for output directory; defaults to detected from inputs.')
    parser.add_argument('--config', type=str, default=None, help='YAML config, same schema as func_corr/main/config.yaml.')
    parser.add_argument('--summaries', type=str, nargs='*', default=None, help='One or more monitor summary JSON paths.')
    parser.add_argument('--dir', type=str, default=None, help='Predictions directory containing *_monitor_predictions_summary.json files.')
    parser.add_argument('--baseline-dir', type=str, default='results/regular_RL/step_200/predictions', help='Baseline predictions directory to compare against.')
    parser.add_argument('--output-dir', type=str, default=None, help='Base directory to save plots; default is results/plots/<dataset_name>/conditional_monitor. If using --dir with --output-dir, saves directly in that folder.')
    args = parser.parse_args()

    runs: List[Tuple[str, List[Path]]] = []  # (run_name, [monitor_summary_paths])
    if args.config:
        cfg = load_yaml(Path(args.config))
        mon = (cfg.get('baseline') or {}).get('monitors') or []
        paths = [Path(m['accuracy_path']) for m in mon if isinstance(m, dict) and 'accuracy_path' in m]
        if paths:
            runs.append((cfg.get('baseline', {}).get('name', 'baseline'), paths))
        for run in cfg.get('runs', []):
            mon = (run or {}).get('monitors') or []
            paths = [Path(m['accuracy_path']) for m in mon if isinstance(m, dict) and 'accuracy_path' in m]
            if paths:
                runs.append((run.get('name', 'run'), paths))
    elif args.summaries:
        runs.append(('manual', [Path(p) for p in args.summaries]))
    elif args.dir:
        pred_dir = Path(args.dir)
        if not pred_dir.exists() or not pred_dir.is_dir():
            raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")
        # Collect known monitor summary files or any *_monitor_predictions_summary.json
        candidates = [
            pred_dir / 'gpt_4o_mini_monitor_predictions_summary.json',
            pred_dir / 'gpt_4o_monitor_predictions_summary.json',
        ]
        files = [p for p in candidates if p.exists()]
        if not files:
            files = list(pred_dir.glob('*_monitor_predictions_summary.json'))
        if not files:
            raise FileNotFoundError('No *_monitor_predictions_summary.json files found in the provided directory.')
        runs.append((pred_dir.name, files))
    else:
        raise ValueError('Provide --config, --summaries, or --dir.')

    # Determine dataset name
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = get_dataset_name_from_any([p for _, ps in runs for p in ps])
    base_out = Path(args.output_dir) if args.output_dir else (Path('results') / 'plots' / dataset_name / 'conditional_monitor')

    baseline_dir = Path(args.baseline_dir)
    write_direct_to_base = bool(args.dir) and bool(args.output_dir)

    for run_name, monitor_paths in runs:
        # Load all monitor data for this run
        monitor_data: List[Tuple[str, Dict[str, Any]]] = []
        compare_data: List[Tuple[str, Dict[str, Any]]] = []
        for monitor_path in monitor_paths:
            try:
                summary = load_json(monitor_path)
                monitor_name = monitor_path.stem.replace('_monitor_predictions_summary', '').replace('_', '-')
                monitor_data.append((monitor_name, summary))
                comp = get_compare_for_monitor(monitor_path, baseline_dir)
                if comp:
                    compare_data.append((monitor_name, comp))
            except Exception as e:
                print(f"Failed to load {monitor_path}: {e}")
                continue
        
        if not monitor_data:
            continue
        
        # Create combined plots for this run
        out_dir = base_out if write_direct_to_base else (base_out / run_name.replace(' ', '_'))
        
        plot_lv_combined(monitor_data, out_dir)
        plot_reasoner_combined(monitor_data, out_dir)
        plot_groups_combined(monitor_data, out_dir)

        if compare_data:
            plot_lv_combined_compare(monitor_data, compare_data, out_dir)
            plot_reasoner_combined_compare(monitor_data, compare_data, out_dir)
            plot_groups_combined_compare(monitor_data, compare_data, out_dir)
        
        print(f"Created combined conditional monitor plots for {run_name} with {len(monitor_data)} monitors. Compare plots: {'yes' if compare_data else 'no'}")


if __name__ == '__main__':
    main()
