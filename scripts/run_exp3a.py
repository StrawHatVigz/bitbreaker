#!/usr/bin/env python3
"""
run_exp3a.py

Experiment 3a: Role Targeting -- Attention vs FFN

Goal: Determine whether attention or FFN components are structurally more
      vulnerable to bit-flip attacks. Fixes flip_count=10, varies role_filter.

Region per model (most vulnerable confirmed in Exp 2):
    FP16    → weights
    Q8_0    → block_scale
    Q4_0    → block_scale
    Q4_K_M  → per-condition (see note below)

Q4_K_M note:
    llama.cpp's Q4_K_M applies full K-quant (with super_scale_d) selectively.
    Qwen:  attn tensors → Q4_0 internally (block_scale only)
           ffn tensors  → Q4_K internally (super_scale_d available)
    Llama: both attn and ffn → Q4_K internally (super_scale_d available)
    Fix (Option B): normalize to block_scale for all attn, super_scale_d for
    all ffn. Fair within-model comparison, consistent across both model families.
    Confirmed via sanity check on model maps before running.

Bit selection (high_impact=True, consistent with Exp 2):
    FP16 / scale regions : exponent zone bits 10-14
    Q8_0 weights         : sign bit 7
    Nibble weights        : MSB bit 3

Output:
    experiments/results/fault_injection/exp3a_role_targeting/
        qwen_fp16/condition_attn_only/seed_0/flip_10_results.json
        qwen_fp16/condition_ffn_only/seed_0/flip_10_results.json
        ...
        exp3a_summary.json

Usage:
    python scripts/run_exp3a.py --platform grendel
    python scripts/run_exp3a.py --platform mac
    python scripts/run_exp3a.py --platform grendel --dry-run
"""

import argparse
import json
import platform as platform_lib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "fault_injection"))

from model_map import ModelMap
from fault_injector import FaultInjector

# ─────────────────────────────────────────────
# Experiment config
# ─────────────────────────────────────────────

FLIP_COUNT = 10  # fixed for Exp 3a/3b

RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "fault_injection" / "exp3a_role_targeting"
)

# (model_file, model_label, quant_label, region_filter)
# region_filter:
#   None or str  → same region for all conditions
#   dict         → maps condition_name → region (Q4_K_M special case)
MODELS = [
    ("qwen2.5-0.5b-instruct-fp16.gguf",   "qwen_fp16",  "FP16",   None),
    ("qwen2.5-0.5b-instruct-q8_0.gguf",   "qwen_q8",    "Q8_0",   "block_scale"),
    ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen_q4km",  "Q4_K_M", {"attn_only": "block_scale", "ffn_only": "super_scale_d"}),
    ("qwen2.5-0.5b-instruct-q4_0.gguf",   "qwen_q4",    "Q4_0",   "block_scale"),
    ("Llama-3.2-1B-Instruct-f16.gguf",    "llama_fp16", "FP16",   None),
    ("Llama-3.2-1B-Instruct-Q8_0.gguf",   "llama_q8",   "Q8_0",   "block_scale"),
    ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "llama_q4km", "Q4_K_M", {"attn_only": "super_scale_d", "ffn_only": "super_scale_d"}),
    ("Llama-3.2-1B-Instruct-Q4_0.gguf",   "llama_q4",   "Q4_0",   "block_scale"),
]

# Role conditions: (condition_name, role_filter passed to injector.run)
ROLE_CONDITIONS = [
    ("attn_only", ['attn']),
    ("ffn_only",  ['ffn']),
]

SEEDS_GRENDEL = [0, 1, 2]
SEEDS_MAC     = [0, 1]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def detect_platform() -> str:
    return 'mac' if platform_lib.system() == 'Darwin' else 'grendel'


def resolve_region(region_filter, condition_name: str):
    """
    Resolve region_filter for a given condition.
    Handles both flat str/None and per-condition dict (Q4_K_M case).
    """
    if isinstance(region_filter, dict):
        return region_filter[condition_name]
    return region_filter


def should_skip(output_path: Path) -> bool:
    """Skip completed runs (valid or total-corruption). Re-run errors."""
    if not output_path.exists():
        return False
    try:
        with open(output_path) as f:
            d = json.load(f)
        if d.get('error') is not None:
            return False
        return True
    except Exception:
        return False


def build_summary(all_results: list) -> dict:
    summary = {'experiment': 'exp3a_role_targeting', 'runs': []}
    for r in all_results:
        ppl       = (r.get('perplexity') or {}).get('ppl')
        tasks     = r.get('tasks') or {}
        arc_easy  = (tasks.get('arc_easy')  or {}).get('accuracy')
        hellaswag = (tasks.get('hellaswag') or {}).get('accuracy')
        summary['runs'].append({
            'model':      r.get('model_label'),
            'quant':      r.get('quant_label'),
            'condition':  r.get('condition'),
            'region':     r.get('region_filter'),
            'flip_count': r['flip_count'],
            'seed':       r['seed'],
            'ppl':        ppl,
            'arc_easy':   arc_easy,
            'hellaswag':  hellaswag,
            'error':      r.get('error'),
        })
    return summary


def print_summary(summary: dict):
    print(f"\n{'='*100}")
    print(f"  Exp 3a Summary -- Role Targeting: Attn vs FFN")
    print(f"{'='*100}")
    print(f"  {'Model':<14} {'Quant':<8} {'Condition':<12} {'Region':<16} {'Seed':>4} "
          f"{'PPL':>14} {'ARC':>8} {'Hella':>8}  {'Status':>6}")
    print(f"  {'-'*92}")

    prev_model = None
    for r in summary['runs']:
        if prev_model and r['model'] != prev_model:
            print()
        prev_model = r['model']

        ppl_str   = f"{r['ppl']:.3f}" if isinstance(r['ppl'], float) else "inf/nan"
        arc_str   = f"{r['arc_easy']*100:.1f}%"  if r['arc_easy']   else "N/A"
        hella_str = f"{r['hellaswag']*100:.1f}%" if r['hellaswag']  else "N/A"
        status    = "ERR" if r['error'] else "OK"
        region    = r.get('region') or 'weights'

        print(f"  {r['model']:<14} {r['quant']:<8} {r['condition']:<12} {region:<16} "
              f"{r['seed']:>4} {ppl_str:>14} {arc_str:>8} {hella_str:>8}  {status:>6}")

    print(f"{'='*100}\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='BitBreaker Exp 3a: Role Targeting -- Attn vs FFN'
    )
    parser.add_argument('--platform', choices=['mac', 'grendel'], default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    platform = args.platform or detect_platform()
    seeds    = SEEDS_MAC if platform == 'mac' else SEEDS_GRENDEL

    total_runs = len(MODELS) * len(ROLE_CONDITIONS) * len(seeds)

    print(f"\n{'='*60}")
    print(f"  BitBreaker -- Experiment 3a: Role Targeting")
    print(f"{'='*60}")
    print(f"  Platform    : {platform}")
    print(f"  Models      : {len(MODELS)}")
    print(f"  Conditions  : attn_only, ffn_only")
    print(f"  Flip count  : {FLIP_COUNT} (fixed)")
    print(f"  Seeds       : {seeds}")
    print(f"  Bit mode    : high-impact zone")
    print(f"  Benchmarks  : PPL + ARC-Easy + HellaSwag")
    print(f"  Total runs  : {total_runs}")
    print(f"  Output      : {RESULTS_DIR}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  [DRY RUN] Run matrix:")
        for model_file, label, quant, region_filter in MODELS:
            for cname, role in ROLE_CONDITIONS:
                region = resolve_region(region_filter, cname) or 'weights'
                print(f"    {label:<14} {quant:<8} condition={cname:<12} "
                      f"region={region:<16} seeds={seeds}")
        print(f"\n  Total: {total_runs} runs. No files modified. Exiting.")
        return

    all_results  = []
    model_number = 0
    t_start      = time.time()

    for model_file, model_label, quant_label, region_filter in MODELS:
        model_number += 1
        model_path = PROJECT_ROOT / "models" / model_file
        map_path   = (
            PROJECT_ROOT / "experiments" / "maps"
            / f"{Path(model_file).stem}_map.json"
        )

        print(f"\n{'='*60}")
        print(f"  [{model_number}/{len(MODELS)}] {model_label}  ({quant_label})")
        print(f"{'='*60}")

        if not model_path.exists():
            print(f"  SKIP: model not found at {model_path}")
            continue
        if not map_path.exists():
            print(f"  SKIP: map not found — run scripts/run_build_maps.py first")
            continue

        mm = ModelMap.load(str(map_path))
        print(f"  Loaded map: {len(mm.in_scope_tensors)} in-scope tensors, "
              f"{mm.n_layers} layers")

        injector = FaultInjector(
            model_map      = mm,
            project_root   = PROJECT_ROOT,
            n_gpu_layers   = 99,
            platform       = platform,
            run_perplexity = True,
            run_tasks      = True,
        )

        condition_runs = len(ROLE_CONDITIONS) * len(seeds)
        run_number     = 0

        for condition_name, role_filter in ROLE_CONDITIONS:
            region = resolve_region(region_filter, condition_name)
            output_dir = RESULTS_DIR / model_label / f"condition_{condition_name}"

            print(f"\n  Condition: {condition_name}  region={region or 'weights'}")

            for seed in seeds:
                run_number += 1
                print(f"\n  [{run_number}/{condition_runs}]  "
                      f"condition={condition_name}  seed={seed}")

                result_path = (
                    output_dir / f"seed_{seed}" / f"flip_{FLIP_COUNT}_results.json"
                )

                if should_skip(result_path):
                    print(f"    SKIP (already complete): {result_path.name}")
                    continue

                result = injector.run(
                    flip_count    = FLIP_COUNT,
                    seed          = seed,
                    region_filter = region,
                    role_filter   = role_filter,
                    layer_range   = None,
                    output_path   = result_path,
                    bit_position  = None,
                    high_impact   = True,
                )

                result['model_label'] = model_label
                result['quant_label'] = quant_label
                result['condition']   = condition_name
                result['region_filter'] = region
                # re-save with labels so rebuild scripts can use them
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                all_results.append(result)

    # ── Rebuild summary from disk (captures skipped runs from prior sessions) ──
    summary_path = RESULTS_DIR / "exp3a_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    all_disk_results = []
    for result_file in sorted(RESULTS_DIR.rglob('flip_*_results.json')):
        with open(result_file) as f:
            d = json.load(f)
        parts = result_file.parts
        all_disk_results.append({
            'model':      d.get('model_label') or parts[-4],
            'quant':      d.get('quant_label'),
            'condition':  d.get('condition') or parts[-3].replace('condition_', ''),
            'flip_count': d['flip_count'],
            'seed':       d['seed'],
            'ppl':        (d.get('perplexity') or {}).get('ppl'),
            'arc_easy':   ((d.get('tasks') or {}).get('arc_easy')  or {}).get('accuracy'),
            'hellaswag':  ((d.get('tasks') or {}).get('hellaswag') or {}).get('accuracy'),
            'error':      d.get('error'),
            'region':     d.get('region_filter'),
        })
    summary = {'experiment': 'exp3a_role_targeting', 'runs': all_disk_results}
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    elapsed = time.time() - t_start
    print_summary(summary)
    print(f"  Total time : {elapsed/60:.1f} min")
    print(f"  Summary    : {summary_path}")


if __name__ == '__main__':
    main()
