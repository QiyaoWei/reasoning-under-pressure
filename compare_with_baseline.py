import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_interval_from_reasoner(ci_list: Optional[List[float]]) -> Optional[Tuple[float, float, float]]:
    if not ci_list or len(ci_list) < 3:
        return None
    lb, ub, margin = ci_list[0], ci_list[1], ci_list[2]
    return lb, ub, margin


def as_interval_from_monitor(ci_obj: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
    if not ci_obj:
        return None
    lb = ci_obj.get("lower_bound")
    ub = ci_obj.get("upper_bound")
    margin = ci_obj.get("margin")
    if lb is None or ub is None or margin is None:
        return None
    return lb, ub, margin


def ci_overlap(incentive: Optional[Tuple[float, float, float]], baseline: Optional[Tuple[float, float, float]]) -> Dict[str, Any]:
    if incentive is None or baseline is None:
        return {
            "incentive_margin": None,
            "baseline_margin": None,
            "overlap": None,
            "has_overlap": None,
        }
    i_lb, i_ub, i_margin = incentive
    b_lb, b_ub, b_margin = baseline
    overlap_len = max(0.0, min(i_ub, b_ub) - max(i_lb, b_lb))
    return {
        "incentive_margin": i_margin,
        "baseline_margin": b_margin,
        "overlap": overlap_len,
        "has_overlap": overlap_len > 0,
    }


def compare_reasoner(incentive_dir: str, baseline_dir: str, out_dir: str) -> Optional[str]:
    incentive_path = os.path.join(incentive_dir, "reasoner_accuracy_summary.json")
    baseline_path = os.path.join(baseline_dir, "reasoner_accuracy_summary.json")
    if not os.path.exists(incentive_path):
        return None
    if not os.path.exists(baseline_path):
        # Still write a stub noting missing baseline
        result = {
            "incentive_dir": incentive_dir,
            "baseline_dir": baseline_dir,
            "error": "baseline reasoner_accuracy_summary.json not found",
        }
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, "reasoner_accuracy_comparison.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return out_path

    inc = read_json(incentive_path)
    base = read_json(baseline_path)

    def extract_overall(entry: Dict[str, Any], metric_key: str) -> Tuple[Optional[float], Optional[Tuple[float, float, float]]]:
        node = entry.get(metric_key)
        if not isinstance(node, dict):
            return None, None
        val = node.get("value")
        ci = as_interval_from_reasoner(node.get("confidence_interval"))
        return val, ci

    metrics_overall = {}
    overall_keys = [
        "measurement-wise_accuracy_overall",
        "all_correct_accuracy_overall",
        "format_accuracy_overall",
    ]
    for key in overall_keys:
        inc_val, inc_ci = extract_overall(inc, key)
        base_val, base_ci = extract_overall(base, key)
        diff_val = None if inc_val is None or base_val is None else (inc_val - base_val)
        metrics_overall[key] = {
            "incentive_value": inc_val,
            "baseline_value": base_val,
            "diff": diff_val,
            "ci": ci_overlap(inc_ci, base_ci),
        }

    group_results: Dict[str, Any] = {}
    inc_groups = inc.get("group_metrics", {}) or {}
    base_groups = base.get("group_metrics", {}) or {}
    group_metric_names = [
        "measurement-wise_accuracy",
        "all_correct_accuracy",
        "format_accuracy",
    ]
    for group_name, inc_group in inc_groups.items():
        base_group = base_groups.get(group_name, {}) or {}
        per_group: Dict[str, Any] = {}
        for m in group_metric_names:
            inc_node = inc_group.get(m)
            base_node = base_group.get(m)
            inc_val = inc_node.get("value") if isinstance(inc_node, dict) else None
            base_val = base_node.get("value") if isinstance(base_node, dict) else None
            inc_ci = as_interval_from_reasoner(inc_node.get("confidence_interval")) if isinstance(inc_node, dict) else None
            base_ci = as_interval_from_reasoner(base_node.get("confidence_interval")) if isinstance(base_node, dict) else None
            diff_val = None if inc_val is None or base_val is None else (inc_val - base_val)

            entry: Dict[str, Any] = {
                "incentive_value": inc_val,
                "baseline_value": base_val,
                "diff": diff_val,
                "ci": ci_overlap(inc_ci, base_ci),
            }

            # Include counts when available
            if isinstance(inc_node, dict) and "correct_count" in inc_node:
                entry["incentive_correct_count"] = inc_node.get("correct_count")
            if isinstance(base_node, dict) and "correct_count" in base_node:
                entry["baseline_correct_count"] = base_node.get("correct_count")
            if "incentive_correct_count" in entry and "baseline_correct_count" in entry:
                entry["diff_correct_count"] = entry["incentive_correct_count"] - entry["baseline_correct_count"]

            if isinstance(inc_group, dict) and "n_samples" in inc_group:
                entry["incentive_total_samples"] = inc_group.get("n_samples")
            if isinstance(base_group, dict) and "n_samples" in base_group:
                entry["baseline_total_samples"] = base_group.get("n_samples")
            if "incentive_total_samples" in entry and "baseline_total_samples" in entry:
                entry["diff_total_samples"] = entry["incentive_total_samples"] - entry["baseline_total_samples"]

            per_group[m] = entry
        group_results[group_name] = per_group

    result = {
        "incentive_dir": incentive_dir,
        "baseline_dir": baseline_dir,
        "metrics_overall": metrics_overall,
        "group_metrics": group_results,
    }

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "reasoner_accuracy_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path


def compare_monitor_file(incentive_path: str, baseline_path: str, out_dir: str) -> Optional[str]:
    filename = os.path.basename(incentive_path)
    if not os.path.exists(baseline_path):
        result = {
            "incentive_file": incentive_path,
            "baseline_file": baseline_path,
            "error": "baseline monitor summary not found",
        }
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, filename.replace("_summary.json", "_comparison_with_baseline.json"))
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return out_path

    inc = read_json(incentive_path)
    base = read_json(baseline_path)

    def compare_accuracy(inc_acc: Optional[float], base_acc: Optional[float], inc_ci_obj: Optional[Dict[str, Any]], base_ci_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        diff_val = None if inc_acc is None or base_acc is None else (inc_acc - base_acc)
        return {
            "incentive_value": inc_acc,
            "baseline_value": base_acc,
            "diff": diff_val,
            "ci": ci_overlap(as_interval_from_monitor(inc_ci_obj), as_interval_from_monitor(base_ci_obj)),
        }

    overall = compare_accuracy(
        inc.get("accuracy"),
        base.get("accuracy"),
        inc.get("accuracy_95_ci"),
        base.get("accuracy_95_ci"),
    )
    # add counts
    overall["incentive_correct_predictions"] = inc.get("correct_predictions")
    overall["baseline_correct_predictions"] = base.get("correct_predictions")
    if overall.get("incentive_correct_predictions") is not None and overall.get("baseline_correct_predictions") is not None:
        overall["diff_correct_predictions"] = overall["incentive_correct_predictions"] - overall["baseline_correct_predictions"]
    overall["incentive_total_samples"] = inc.get("total_samples")
    overall["baseline_total_samples"] = base.get("total_samples")
    if overall.get("incentive_total_samples") is not None and overall.get("baseline_total_samples") is not None:
        overall["diff_total_samples"] = overall["incentive_total_samples"] - overall["baseline_total_samples"]

    conds_result: Dict[str, Any] = {}
    inc_conds = inc.get("conditional_accuracies", {}) or {}
    base_conds = base.get("conditional_accuracies", {}) or {}

    def compare_cond(name: str, inc_node: Dict[str, Any], base_node: Dict[str, Any]) -> Dict[str, Any]:
        entry = compare_accuracy(
            inc_node.get("accuracy"),
            base_node.get("accuracy"),
            inc_node.get("95_ci"),
            base_node.get("95_ci"),
        )
        entry["incentive_correct_predictions"] = inc_node.get("correct_predictions")
        entry["baseline_correct_predictions"] = base_node.get("correct_predictions")
        if entry.get("incentive_correct_predictions") is not None and entry.get("baseline_correct_predictions") is not None:
            entry["diff_correct_predictions"] = entry["incentive_correct_predictions"] - entry["baseline_correct_predictions"]
        entry["incentive_total_samples"] = inc_node.get("total_samples")
        entry["baseline_total_samples"] = base_node.get("total_samples")
        if entry.get("incentive_total_samples") is not None and entry.get("baseline_total_samples") is not None:
            entry["diff_total_samples"] = entry["incentive_total_samples"] - entry["baseline_total_samples"]

        # Nested conditions inside group buckets (e.g., when_reasoner_correct)
        nested: Dict[str, Any] = {}
        for k, v in inc_node.items():
            if k in {"accuracy", "correct_predictions", "total_samples", "95_ci"}:
                continue
            if isinstance(v, dict):
                base_v = base_node.get(k, {}) if isinstance(base_node, dict) else {}
                nested[k] = compare_cond(f"{name}.{k}", v, base_v if isinstance(base_v, dict) else {})
        if nested:
            entry["nested"] = nested
        return entry

    for cond_name, inc_node in inc_conds.items():
        base_node = base_conds.get(cond_name, {}) if isinstance(base_conds, dict) else {}
        if isinstance(inc_node, dict):
            conds_result[cond_name] = compare_cond(cond_name, inc_node, base_node if isinstance(base_node, dict) else {})

    result = {
        "incentive_file": incentive_path,
        "baseline_file": baseline_path,
        "overall": overall,
        "conditional_accuracies": conds_result,
    }

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename.replace("_summary.json", "_comparison_with_baseline.json"))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path


def compare_monitors(incentive_dir: str, baseline_dir: str, out_dir: str) -> List[str]:
    written: List[str] = []
    for name in os.listdir(incentive_dir):
        if not name.endswith("_monitor_predictions_summary.json"):
            continue
        inc_path = os.path.join(incentive_dir, name)
        base_path = os.path.join(baseline_dir, name)
        out = compare_monitor_file(inc_path, base_path, out_dir)
        if out:
            written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare incentive results with RL baseline and write JSON summaries.")
    parser.add_argument("incentive_dir", type=str, help="Path to the incentive predictions directory (e.g., results/diamond/penalize_verbosity_1e-7/.../predictions)")
    parser.add_argument("--baseline_dir", type=str, default="results/diamond/regular/predictions", help="Path to the baseline predictions directory")
    args = parser.parse_args()

    incentive_dir = args.incentive_dir
    baseline_dir = args.baseline_dir

    if not os.path.isdir(incentive_dir):
        raise FileNotFoundError(f"Incentive directory not found: {incentive_dir}")
    if not os.path.isdir(baseline_dir):
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    out_dir = os.path.join(incentive_dir, "compare_with_baseline")

    compare_reasoner(incentive_dir, baseline_dir, out_dir)
    compare_monitors(incentive_dir, baseline_dir, out_dir)


if __name__ == "__main__":
    main()


