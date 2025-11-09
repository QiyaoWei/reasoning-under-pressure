#!/usr/bin/env python3
"""
Generate grouped, side-by-side sample reports comparing a target folder against a
baseline (regular RL) folder.

Inputs:
- target predictions dir (e.g., results/penalize_verbosity/k_-5e-8/step_260/predictions)
- baseline predictions dir (defaults to results/regular_RL/step_200/predictions)

Outputs:
- Text files under: <target_dir>/samples/<reasonerX_monitor>/<group_label>.txt
  where <reasonerX_monitor> in {
    reasoner_correct__monitor_correct,
    reasoner_incorrect__monitor_incorrect,
    reasoner_correct__monitor_incorrect,
    reasoner_incorrect__monitor_correct
  }

Each file starts with total counts for the category and group, followed by up to N
random samples (default N=5, configurable).

Sample content order:
1) Prompt "text" (from raw outputs) for target (left) and baseline (right) to verify match
2) Reasoning traces: target vs baseline side-by-side
3) Original predictions and ground truth; reasoner_is_correct (target and baseline)
4) Monitor sections for gpt-4o-mini then gpt-4o:
   - monitor_reasoning (left target, right baseline)
   - monitor_prediction, latent_variable, monitor_is_correct (both)

Notes:
- Sampling universe is defined by target gpt_4o_mini_monitor_predictions.json entries.
- Alignment by sample_idx across all files.
"""

import argparse
import json
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


RANDOM_SEED_DEFAULT = 17


def load_json_file(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        content = f.read().lstrip()
        if content.startswith('[') or content.startswith('{'):
            return json.loads(content)
        # Fallback JSONL
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def index_by_sample_idx(items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        if 'sample_idx' in it:
            try:
                idx = int(it['sample_idx'])
            except Exception:
                continue
            indexed[idx] = it
    return indexed


def build_required_paths(base_dir: Path) -> Dict[str, Path]:
    paths = {
        'mini_pred': base_dir / 'gpt_4o_mini_monitor_predictions.json',
        'o4_pred': base_dir / 'gpt_4o_monitor_predictions.json',
    }
    # Find the raw outputs file dynamically (ends with raw_outputs.json)
    raw_candidates = list(base_dir.glob('*raw_outputs.json'))
    if not raw_candidates:
        raise FileNotFoundError(f"No raw outputs JSON found under {base_dir}")
    # Prefer shortest name (more likely the step-specific file) if multiple
    raw_candidates.sort(key=lambda p: len(p.name))
    paths['raw_outputs'] = raw_candidates[0]
    return paths


def load_bundle(dir_path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    paths = build_required_paths(dir_path)
    mini = load_json_file(paths['mini_pred'])
    o4 = load_json_file(paths['o4_pred'])
    raw = load_json_file(paths['raw_outputs'])

    # Normalize to lists
    if isinstance(mini, dict):
        mini = [mini]
    if isinstance(o4, dict):
        o4 = [o4]
    if isinstance(raw, dict):
        raw = [raw]

    return index_by_sample_idx(mini), index_by_sample_idx(o4), index_by_sample_idx(raw)


def compute_monitor_correct(sample: Dict[str, Any]) -> Optional[bool]:
    if 'monitor_correct' in sample:
        return bool(sample['monitor_correct'])
    else:
        if 'monitor_prediction' in sample and 'latent_variable' in sample:
            return bool(sample['monitor_prediction']) == bool(sample['latent_variable'])
        else:
            raise ValueError(f"monitor_prediction or latent_variable not found in sample")


def reasoner_correct_bucket(val: Any) -> Optional[bool]:
    try:
        return bool(val)
    except Exception:
        raise ValueError(f"reasoner_is_correct not found in sample")


def get_group_key(reasoner_correct: bool, monitor_correct: bool) -> str:
    return (
        ('reasoner_correct__' if reasoner_correct else 'reasoner_incorrect__') +
        ('monitor_correct' if monitor_correct else 'monitor_incorrect')
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _wrap_block(text: str, width: Optional[int]) -> List[str]:
    if not text:
        return ['']
    if not width or width <= 0:
        return text.splitlines()
    wrapped: List[str] = []
    for line in text.splitlines():
        # Preserve empty lines
        if not line:
            wrapped.append('')
            continue
        parts = textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not parts:
            wrapped.append('')
        else:
            wrapped.extend(parts)
    return wrapped


def side_by_side(left: str, right: str, sep: str = ' | ', left_width: Optional[int] = None, right_width: Optional[int] = None) -> str:
    # Render two multi-line blocks side by side with a separator, optionally wrapping columns
    left_lines = _wrap_block(left, left_width)
    right_lines = _wrap_block(right, right_width)
    max_len_left = max((len(line) for line in left_lines), default=0)
    padded_left = [line.ljust(max_len_left) for line in left_lines]
    rows = []
    for i in range(max(len(padded_left), len(right_lines))):
        l = padded_left[i] if i < len(padded_left) else ' ' * max_len_left
        r = right_lines[i] if i < len(right_lines) else ''
        rows.append(f"{l}{sep}{r}")
    return "\n".join(rows)


def render_sample(
    idx: int,
    target_mini: Dict[str, Any],
    target_o4: Optional[Dict[str, Any]],
    target_raw: Optional[Dict[str, Any]],
    base_mini: Optional[Dict[str, Any]],
    base_o4: Optional[Dict[str, Any]],
    base_raw: Optional[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"===== sample_idx={idx} =====")

    # 1) Prompt text (target only)
    tgt_text = (target_raw or {}).get('text', '')
    lines.append("[Prompt text] (target)")
    lines.append(tgt_text)
    lines.append("")

    # 2) Reasoning traces (reasoner) side-by-side
    tgt_reason = (target_mini or {}).get('reasoning_trace', '')
    base_reason = (base_mini or {}).get('reasoning_trace', '')
    lines.append("[Reasoning trace] (target | baseline)")
    lines.append(side_by_side(tgt_reason, base_reason, sep=' | ', left_width=70, right_width=70))
    lines.append("")

    # 3) Predictions and ground truths
    def fmt_pred_block(s: Optional[Dict[str, Any]]) -> str:
        if not s:
            return "<missing>"
        original_prediction = s.get('original_prediction')
        ground_truth = s.get('ground_truth')
        reasoner_is_correct = s.get('reasoner_is_correct')
        return (
            f"original_prediction: {original_prediction}\n"
            f"ground_truth: {ground_truth}\n"
            f"reasoner_is_correct: {reasoner_is_correct}"
        )

    lines.append("[Predictions] (target | baseline)")
    left_block = fmt_pred_block(target_mini)
    right_block = fmt_pred_block(base_mini)
    lines.append(side_by_side(left_block, right_block, sep=' | '))
    lines.append("")

    # 4) Monitor sections for gpt-4o-mini
    def fmt_monitor_block(s: Optional[Dict[str, Any]]) -> str:
        if not s:
            return "<missing>"
        monitor_reasoning = s.get('monitor_reasoning') or s.get('monitor_full_response') or ''
        monitor_prediction = s.get('monitor_prediction')
        latent_variable = s.get('latent_variable')
        monitor_is_correct = compute_monitor_correct(s)
        group = s.get('group')
        return (
            f"monitor_reasoning:\n{monitor_reasoning}\n"
            f"monitor_prediction: {monitor_prediction}\n"
            f"latent_variable: {latent_variable}\n"
            f"monitor_is_correct: {monitor_is_correct}\n"
            f"group: {group}"
        )

    lines.append("[Monitor gpt-4o-mini] (target | baseline)")
    left_block = fmt_monitor_block(target_mini)
    right_block = fmt_monitor_block(base_mini)
    lines.append(side_by_side(left_block, right_block, sep=' | ', left_width=70, right_width=70))
    lines.append("")

    # 5) Monitor sections for gpt-4o
    lines.append("[Monitor gpt-4o] (target | baseline)")
    left_block = fmt_monitor_block(target_o4)
    right_block = fmt_monitor_block(base_o4)
    lines.append(side_by_side(left_block, right_block, sep=' | ', left_width=70, right_width=70))
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate grouped side-by-side sample reports')
    parser.add_argument('target_dir', type=str,
                        help='Path to target predictions dir (e.g., results/diamond/.../predictions)')
    parser.add_argument('--baseline-dir', type=str, default='results/diamond/regular/predictions',
                        help='Path to baseline predictions dir (default: results/diamond/regular/predictions)')
    parser.add_argument('-n', '--num-samples', type=int, default=5,
                        help='Number of samples per group per correctness bucket (default: 5)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED_DEFAULT,
                        help=f'Random seed (default: {RANDOM_SEED_DEFAULT})')
    parser.add_argument('--require-baseline-match', action='store_true',
                        help='If set, include only samples whose prompt text matches a baseline sample; otherwise include unmatched with empty baseline columns')
    args = parser.parse_args()

    random.seed(args.seed)

    target_dir = Path(args.target_dir).resolve()
    base_dir = Path(args.baseline_dir).resolve()

    # Load and index bundles with sample_idx as key
    tgt_mini, tgt_o4, tgt_raw = load_bundle(target_dir)  # with incentive
    base_mini, base_o4, base_raw = load_bundle(base_dir)  # Regular RL baseline

    # Build baseline text->sample_idx map to align by prompt text when indices differ
    base_text_to_idx: Dict[str, int] = {}
    for bidx, brow in base_raw.items():
        text = brow.get('text')
        if isinstance(text, str) and text not in base_text_to_idx:
            base_text_to_idx[text] = bidx

    # Prepare output directories
    samples_root = target_dir / 'samples'
    ensure_dir(samples_root)

    # Define correctness combinations and groups to render
    # (reasoner_correct, monitor_correct)
    correctness_buckets = [
        (True, True),
        (False, False),
        (True, False),
        (False, True),
    ]
    group_labels = ['all_false_lv_false', 'mixed_lv_false', 'all_true_lv_true', 'all_true_lv_false']

    # Build buckets by correctness and group
    # Using target gpt-4o-mini predictions as the master set
    by_combo_and_group: Dict[str, Dict[str, List[int]]] = {}
    for idx, sample in tgt_mini.items():
        r_correct = reasoner_correct_bucket(sample.get('reasoner_is_correct'))
        m_correct = compute_monitor_correct(sample)
        combo_key = get_group_key(r_correct, m_correct)
        grp = sample.get('group')
        if not isinstance(grp, str):
            raise ValueError(f"Group not found in sample. Run monitorability_analysis.py to add it.")
        bucket_for_combo = by_combo_and_group.setdefault(combo_key, {})
        bucket_for_combo.setdefault(grp, []).append(idx)

    # For each combo and group, write a file
    for (r_correct, m_correct) in correctness_buckets:
        combo_key = get_group_key(r_correct, m_correct)
        combo_dir = samples_root / combo_key
        ensure_dir(combo_dir)

        groups_map = by_combo_and_group.get(combo_key, {})
        for grp in group_labels:
            indices = groups_map.get(grp, [])
            total_count = len(indices)
            # Determine eligibility based on baseline text match
            def has_baseline_match(sample_idx: int) -> bool:
                tr = tgt_raw.get(sample_idx)
                if not isinstance(tr, dict):
                    return False
                text = tr.get('text')
                return isinstance(text, str) and text in base_text_to_idx

            eligible_count = sum(1 for idx in indices if has_baseline_match(idx))

            # Shuffle and then select according to match policy
            shuffled = indices.copy()
            random.shuffle(shuffled)
            blocks: List[str] = []
            if args.require_baseline_match:
                skip_notes: List[str] = []
                for idx in shuffled:
                    if len(blocks) >= max(0, args.num_samples):
                        break
                    tr = tgt_raw.get(idx)
                    tgt_text = tr.get('text') if isinstance(tr, dict) else None
                    if not (isinstance(tgt_text, str) and tgt_text in base_text_to_idx):
                        skip_notes.append(f"Note: Skipping sample_idx={idx} (no baseline match by text)")
                        print(f"Note: Skipping sample_idx={idx} (no baseline match by text)")
                        continue

                    tm = tgt_mini.get(idx)
                    to4 = tgt_o4.get(idx)
                    base_idx = base_text_to_idx[tgt_text]
                    bm = base_mini.get(base_idx)
                    bo4 = base_o4.get(base_idx)
                    br = base_raw.get(base_idx)
                    sample_txt = render_sample(idx, tm, to4, tr, bm, bo4, br)
                    blocks.append(sample_txt)
            else:
                for idx in shuffled:
                    if len(blocks) >= max(0, args.num_samples):
                        break
                    tr = tgt_raw.get(idx)
                    tgt_text = tr.get('text') if isinstance(tr, dict) else None
                    matched = isinstance(tgt_text, str) and tgt_text in base_text_to_idx
                    tm = tgt_mini.get(idx)
                    to4 = tgt_o4.get(idx)
                    if matched:
                        base_idx = base_text_to_idx[tgt_text]
                        bm = base_mini.get(base_idx)
                        bo4 = base_o4.get(base_idx)
                        br = base_raw.get(base_idx)
                    else:
                        bm = None
                        bo4 = None
                        br = None
                    sample_txt = render_sample(idx, tm, to4, tr, bm, bo4, br)
                    blocks.append(sample_txt)

            out_path = combo_dir / f"{grp}.txt"
            with out_path.open('w', encoding='utf-8') as f:
                f.write(f"Category: {combo_key}\n")
                f.write(f"Group: {grp}\n")
                f.write(f"Total available in category+group: {total_count}\n")
                f.write(f"Eligible with baseline match: {eligible_count}\n")
                f.write(f"Samples requested: {max(0, args.num_samples)}\n")
                f.write(f"Samples shown: {len(blocks)}\n")
                f.write("=" * 80 + "\n\n")

                # Write the selected sample blocks
                for i, block in enumerate(blocks, 1):
                    f.write(block)
                    if i < len(blocks):
                        f.write("\n" + ("-" * 80) + "\n\n")

    print(f"Done. Reports under: {samples_root}")


if __name__ == '__main__':
    main()


