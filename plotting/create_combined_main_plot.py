#!/usr/bin/env python3
"""
Create a combined monitorability plot for two datasets (e.g., Diamond + Function).
Bars are interleaved per dataset with color indicating the monitor and hatching indicating the dataset.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from matplotlib.patches import Patch


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_accuracy_from_json(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        accuracy = None
        ci_lower = None
        ci_upper = None
        ci_margin = None

        if "metrics" in data and isinstance(data["metrics"], dict):
            metrics = data["metrics"]
            if "measurement-wise_accuracy_overall" in metrics:
                mwao = metrics["measurement-wise_accuracy_overall"]
                if isinstance(mwao, dict) and "value" in mwao:
                    accuracy = mwao["value"]
                    ci = mwao.get("confidence_interval")
                    if isinstance(ci, list) and len(ci) >= 3:
                        ci_lower, ci_upper, ci_margin = ci[0], ci[1], ci[2]

        if accuracy is None and "measurement-wise_accuracy_overall" in data:
            val = data["measurement-wise_accuracy_overall"]
            if isinstance(val, dict) and "value" in val:
                accuracy = val["value"]
                ci = val.get("confidence_interval")
                if isinstance(ci, list) and len(ci) >= 3:
                    ci_lower, ci_upper, ci_margin = ci[0], ci[1], ci[2]
            elif isinstance(val, (int, float)):
                accuracy = float(val)

        if accuracy is None and "accuracy" in data:
            accuracy = data["accuracy"]
            ci_obj = data.get("accuracy_95_ci")
            if isinstance(ci_obj, dict):
                ci_lower = ci_obj.get("lower_bound")
                ci_upper = ci_obj.get("upper_bound")
                ci_margin = ci_obj.get("margin")

        return {
            "accuracy": accuracy,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_margin": ci_margin,
        }
    except Exception:
        return {
            "accuracy": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_margin": None,
        }


def extract_accuracy_data(config_path):
    config = load_config(config_path)
    data = []
    baseline_data = None
    setting = config.get("setting", "Unknown")

    if "baseline" in config:
        baseline = config["baseline"]
        reasoner_data = extract_accuracy_from_json(baseline["accuracy_path"])

        monitor_data = {}
        for monitor in baseline.get("monitors", []):
            monitor_name = monitor["name"]
            monitor_acc = extract_accuracy_from_json(monitor["accuracy_path"])
            monitor_data[monitor_name] = {
                **monitor_acc,
                "significant": bool(monitor.get("significant", False)),
            }

        if reasoner_data["accuracy"] is not None:
            baseline_data = {
                "name": baseline.get("name", "baseline"),
                "setting": setting,
                "reasoner_accuracy": reasoner_data["accuracy"],
                "reasoner_accuracy_ci": {
                    "lower": reasoner_data["ci_lower"],
                    "upper": reasoner_data["ci_upper"],
                    "margin": reasoner_data["ci_margin"],
                },
                "reasoner_significant": bool(baseline.get("reasoner_significant", False)),
                "monitors": monitor_data,
            }

    for run in config["runs"]:
        reasoner_data = extract_accuracy_from_json(run["accuracy_path"])
        monitor_data = {}
        for monitor in run["monitors"]:
            monitor_name = monitor["name"]
            monitor_acc = extract_accuracy_from_json(monitor["accuracy_path"])
            monitor_data[monitor_name] = {
                **monitor_acc,
                "significant": bool(monitor.get("significant", False)),
            }

        if reasoner_data["accuracy"] is not None:
            data.append(
                {
                    "name": run["name"],
                    "setting": setting,
                    "reasoner_accuracy": reasoner_data["accuracy"],
                    "reasoner_accuracy_ci": {
                        "lower": reasoner_data["ci_lower"],
                        "upper": reasoner_data["ci_upper"],
                        "margin": reasoner_data["ci_margin"],
                    },
                    "reasoner_significant": bool(run.get("reasoner_significant", False)),
                    "monitors": monitor_data,
                }
            )

    return data, baseline_data, config


def draw_significant_outline(bar, significant, linewidth=1.6):
    if bar is None:
        return
    if significant:
        bar.set_edgecolor("black")
        bar.set_linewidth(linewidth)
        bar.set_zorder(3)
    else:
        bar.set_edgecolor("none")


def draw_shared_boundary(ax, bottom, top, x_position, linewidth=1.6):
    if np.isnan(top):
        return
    ax.vlines(x_position, bottom, top, colors="black", linewidth=linewidth, zorder=4)


def create_combined_plot(dataset_a, dataset_b, output_dir: Path, save_as_pdf=False, font_size=10):
    """dataset_a/b: dict with keys data, baseline, config"""
    if not dataset_a["data"] or not dataset_b["data"]:
        print("One of the datasets has no runs; aborting.")
        return

    runs_a = [item["name"] for item in dataset_a["data"]]
    lookup_a = {item["name"]: item for item in dataset_a["data"]}
    lookup_b = {item["name"]: item for item in dataset_b["data"]}

    aligned_runs = [run for run in runs_a if run in lookup_b]
    if not aligned_runs:
        print("No overlapping run names between the two configs.")
        return

    colors = sns.color_palette("pastel", 3)
    color_map = {
        "gpt-4o-mini": colors[1],
        "gpt-4o": colors[0],
    }

    hatch_map = {
        dataset_a["setting"]: "",
        dataset_b["setting"]: "///",
    }

    output_subdir = output_dir / "plots" / f"combined_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    fig, ax_mon = plt.subplots(1, 1, figsize=(11, 3.0))

    x_positions = np.arange(len(aligned_runs))
    dataset_order = [dataset_a, dataset_b]

    # Monitorability plot
    group_width = 0.8
    sub_width = group_width / 4 * 0.9
    monitor_offsets = [
        -1.5 * sub_width,
        -0.5 * sub_width,
        0.5 * sub_width,
        1.5 * sub_width,
    ]

    for idx, run_name in enumerate(aligned_runs):
        run_items = [
            (dataset_a, lookup_a[run_name], "gpt-4o-mini"),
            (dataset_a, lookup_a[run_name], "gpt-4o"),
            (dataset_b, lookup_b[run_name], "gpt-4o-mini"),
            (dataset_b, lookup_b[run_name], "gpt-4o"),
        ]

        for offset_idx, (dataset, item, monitor_name) in enumerate(run_items):
            monitor_info = item["monitors"].get(monitor_name, {})
            val = monitor_info.get("accuracy")
            baseline_monitors = dataset["baseline"]["monitors"] if dataset["baseline"] else {}
            baseline_val = baseline_monitors.get(monitor_name, {}).get("accuracy") if baseline_monitors else None
            if val is not None and baseline_val is not None:
                val = val - baseline_val
            if val is None:
                val = np.nan
            error = monitor_info.get("ci_margin") or 0
            xpos = x_positions[idx] + monitor_offsets[offset_idx]
            bar = ax_mon.bar(
                xpos,
                val,
                sub_width,
                yerr=error,
                capsize=3,
                color=color_map.get(monitor_name, colors[0]),
                alpha=0.9,
                hatch=hatch_map[dataset["setting"]],
                zorder=3.5,
            )[0]
            draw_significant_outline(bar, monitor_info.get("significant", False))

        # draw shared boundaries between the adjacent bars if necessary
        pairings = [
            (monitor_offsets[0], monitor_offsets[1], lookup_a[run_name], dataset_a, "gpt-4o-mini", "gpt-4o"),
            (monitor_offsets[2], monitor_offsets[3], lookup_b[run_name], dataset_b, "gpt-4o-mini", "gpt-4o"),
        ]
        for left_offset, right_offset, item_ref, dataset_ref, mon_left, mon_right in pairings:
            baseline_monitors = dataset_ref["baseline"]["monitors"] if dataset_ref["baseline"] else {}
            left_info = item_ref["monitors"].get(mon_left, {})
            right_info = item_ref["monitors"].get(mon_right, {})
            left_val = left_info.get("accuracy")
            right_val = right_info.get("accuracy")
            left_base = baseline_monitors.get(mon_left, {}).get("accuracy") if baseline_monitors else None
            right_base = baseline_monitors.get(mon_right, {}).get("accuracy") if baseline_monitors else None
            left_sig = left_info.get("significant", False) and left_val is not None and left_base is not None
            right_sig = right_info.get("significant", False) and right_val is not None and right_base is not None
            if left_sig or right_sig:
                heights = []
                if left_sig:
                    heights.append(left_val - left_base)
                if right_sig:
                    heights.append(right_val - right_base)
                if heights:
                    max_height = max(heights)
                    min_height = min(heights)
                    draw_shared_boundary(
                        ax_mon,
                        min(0, min_height),
                        max(0, max_height),
                        x_positions[idx] + (left_offset + right_offset) / 2,
                        linewidth=1.6,
                    )

    ax_mon.set_ylabel("Monitorability change", fontsize=font_size)
    ax_mon.set_xticks(x_positions)
    ax_mon.set_xticklabels(aligned_runs, rotation=15, ha="center", fontsize=font_size)
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

    all_vals = []
    all_errs = []
    for ds in (dataset_a, dataset_b):
        for run in ds["data"]:
            for monitor_name in ("gpt-4o-mini", "gpt-4o"):
                monitor_info = run["monitors"].get(monitor_name, {})
                val = monitor_info.get("accuracy")
                baseline_monitors = ds["baseline"]["monitors"] if ds["baseline"] else {}
                baseline_val = baseline_monitors.get(monitor_name, {}).get("accuracy") if baseline_monitors else None
                if val is not None and baseline_val is not None:
                    all_vals.append(val - baseline_val)
                else:
                    all_vals.append(np.nan)
                all_errs.append(monitor_info.get("ci_margin") or 0)

    ymin_mon, ymax_mon = compute_y_lim(all_vals, all_errs)
    ax_mon.set_ylim(ymin_mon, ymax_mon)
    ax_mon.set_title("")
    ax_mon.set_axisbelow(True)
    ax_mon.grid(True, which="major", axis="y", alpha=0.8, linestyle="-", linewidth=0.8, zorder=0)
    ax_mon.grid(True, which="minor", axis="y", alpha=0.5, linestyle=":", linewidth=0.5, zorder=0)
    ax_mon.minorticks_on()
    ax_mon.spines["top"].set_visible(False)
    ax_mon.spines["right"].set_visible(False)
    ax_mon.spines["left"].set_linewidth(1.5)
    ax_mon.spines["bottom"].set_visible(False)
    ax_mon.tick_params(axis="x", which="both", length=0, width=0)
    ax_mon.axhline(0, color="black", linewidth=1.5, zorder=3)

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="", label=dataset_a["setting"]),
        Patch(facecolor="white", edgecolor="black", hatch="///", label=dataset_b["setting"]),
        Patch(facecolor=color_map["gpt-4o-mini"], edgecolor="none", label="GPT-4o mini"),
        Patch(facecolor=color_map["gpt-4o"], edgecolor="none", label="GPT-4o"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, fontsize=font_size, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    file_ext = ".pdf" if save_as_pdf else ".png"
    combined_path = output_subdir / f"combined_plot{file_ext}"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Combined legend is now embedded in the main figure, so we skip separate files
    combined_data = {
        "datasets": [dataset_a["setting"], dataset_b["setting"]],
        "runs": aligned_runs,
        "diamond_data": dataset_a["data"],
        "func_corr_data": dataset_b["data"],
    }
    with open(output_subdir / "combined_data.json", "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"Combined monitorability Δ plot saved to {combined_path}")


def prepare_dataset(config_path):
    data, baseline, config = extract_accuracy_data(config_path)
    return {
        "config_path": config_path,
        "setting": config.get("setting", Path(config_path).stem).title(),
        "data": data,
        "baseline": baseline,
        "font_size": config.get("font_size", 10),
        "save_as_pdf": config.get("save_as_pdf", False),
    }


def main():
    parser = argparse.ArgumentParser(description="Create combined monitorability plot for two datasets.")
    parser.add_argument("config_a", type=str, help="Path to first YAML config (e.g., Diamond).")
    parser.add_argument("config_b", type=str, help="Path to second YAML config (e.g., Function Correctness).")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save outputs.")
    args = parser.parse_args()

    config_a_path = Path(args.config_a)
    config_b_path = Path(args.config_b)
    if not config_a_path.exists() or not config_b_path.exists():
        print("One of the config paths does not exist. Aborting.")
        return

    dataset_a = prepare_dataset(config_a_path)
    dataset_b = prepare_dataset(config_b_path)

    # Use larger font size if either config requests it
    font_size = max(dataset_a["font_size"], dataset_b["font_size"])
    save_as_pdf = dataset_a["save_as_pdf"] or dataset_b["save_as_pdf"]

    create_combined_plot(dataset_a, dataset_b, Path(args.output_dir), save_as_pdf, font_size)


if __name__ == "__main__":
    main()

