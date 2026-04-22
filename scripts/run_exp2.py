#!/usr/bin/env python3
"""
run_exp2.py

Experiment 2: Scale vs Weight Bits -- Damage Efficiency Comparison

Goal: Directly compare damage efficiency of flipping scale bits vs weight bits
at the same flip budget. Tests whether scale bits are disproportionately
dangerous per flip across quantization formats.

Setup:
    Models      : All 8
    Flip counts : [1, 5, 10]
    Seeds       : [0, 1, 2]
    Benchmarks  : All 3 -- PPL + ARC-Easy + HellaSwag (Tier 1)

Three conditions per model:
    A -- weight bits   : random weight byte, pseudo-random high-impact bit
    B -- block_scale   : block-level FP16 scale (Q8_0, Q4_0) or sub_scales (Q4_K_M)
    C -- super_scale_d : super-block FP16 scale (Q4_K_M, Q6_K only)

Bit selection for ALL flips (not just first):
    Scale regions  : pseudo-random from FP16 exponent zone (bits 10-14)
    FP16 weights   : pseudo-random from FP16 exponent zone (bits 10-14)
    INT8 weights   : bit 7 (sign bit -- only meaningful high-impact bit)
    Nibble weights : bit 3 (nibble MSB -- only meaningful high-impact bit)

FP16 models: condition A only (no scale bits exist)
Q4_K_M/Q6_K: all 3 conditions
Q8_0/Q4_0  : conditions A and B only (no super_scale_d)

Output:
    experiments/results/fault_injection/exp2_scale_vs_weight/
        qwen_fp16/condition_weight/seed_0/flip_1_results.json ...
        qwen_q8/condition_weight/...
        qwen_q8/condition_block_scale/...
        qwen_q4km/condition_weight/...
        qwen_q4km/condition_block_scale/...
        qwen_q4km/condition_super_scale/...
        ...
        exp2_summary.json

Usage:
    python scripts/run_exp2.py --platform grendel
    python scripts/run_exp2.py --platform mac
    python scripts/run_exp2.py --platform grendel --dry-run
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

FLIP_COUNTS = [1, 5, 10]
ROLE_FILTER = ['attn', 'ffn', 'output']

RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "fault_injection" / "exp2_scale_vs_weight"
)

# Full Grendel run -- all conditions, 3 seeds
MODELS_GRENDEL = [
    ("qwen2.5-0.5b-instruct-fp16.gguf",   "qwen_fp16",  "FP16",   [("weights", None)]),
    ("qwen2.5-0.5b-instruct-q8_0.gguf",   "qwen_q8",    "Q8_0",   [("weights", None), ("block_scale", "block_scale")]),
    ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen_q4km",  "Q4_K_M", [("weights", None), ("block_scale", "sub_scales"), ("super_scale", "super_scale_d")]),
    ("qwen2.5-0.5b-instruct-q4_0.gguf",   "qwen_q4",    "Q4_0",   [("weights", None), ("block_scale", "block_scale")]),
    ("Llama-3.2-1B-Instruct-f16.gguf",    "llama_fp16", "FP16",   [("weights", None)]),
    ("Llama-3.2-1B-Instruct-Q8_0.gguf",   "llama_q8",   "Q8_0",   [("weights", None), ("block_scale", "block_scale")]),
    ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "llama_q4km", "Q4_K_M", [("weights", None), ("block_scale", "sub_scales"), ("super_scale", "super_scale_d")]),
    ("Llama-3.2-1B-Instruct-Q4_0.gguf",   "llama_q4",   "Q4_0",   [("weights", None), ("block_scale", "block_scale")]),
]

# Mac subset -- skip immune conditions, 2 seeds only
# Weight bits confirmed immune on all quantized formats (Grendel).
# Only run conditions that showed degradation for cross-platform verification.
MODELS_MAC = [
    ("qwen2.5-0.5b-instruct-fp16.gguf",   "qwen_fp16",  "FP16",   [("weights", None)]),           # FP16 cross-platform baseline
    ("qwen2.5-0.5b-instruct-q8_0.gguf",   "qwen_q8",    "Q8_0",   [("block_scale", "block_scale")]),  # skip weights -- immune
    ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen_q4km",  "Q4_K_M", [("super_scale", "super_scale_d")]), # skip weights+block_scale -- immune
    ("qwen2.5-0.5b-instruct-q4_0.gguf",   "qwen_q4",    "Q4_0",   [("block_scale", "block_scale")]),
    ("Llama-3.2-1B-Instruct-f16.gguf",    "llama_fp16", "FP16",   [("weights", None)]),           # most interesting cross-platform
    ("Llama-3.2-1B-Instruct-Q8_0.gguf",   "llama_q8",   "Q8_0",   [("block_scale", "block_scale")]),
    ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "llama_q4km", "Q4_K_M", [("super_scale", "super_scale_d")]),
    ("Llama-3.2-1B-Instruct-Q4_0.gguf",   "llama_q4",   "Q4_0",   [("block_scale", "block_scale")]),
]

SEEDS_GRENDEL = [0, 1, 2]
SEEDS_MAC     = [0, 1]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def detect_platform() -> str:
    return 'mac' if platform_lib.system() == 'Darwin' else 'grendel'


def build_summary(all_results: list) -> dict:
    summary = {'experiment': 'exp2_scale_vs_weight', 'runs': []}
    for r in all_results:
        ppl       = (r.get('perplexity') or {}).get('ppl')
        tasks     = r.get('tasks') or {}
        arc_easy  = tasks.get('arc_easy') or {}
        hellaswag = tasks.get('hellaswag') or {}
        summary['runs'].append({
            'model':       r.get('model_label'),
            'quant':       r.get('quant_label'),
            'condition':   r.get('condition'),
            'flip_count':  r['flip_count'],
            'seed':        r['seed'],
            'ppl':         ppl,
            'arc_easy':    arc_easy.get('accuracy'),
            'hellaswag':   hellaswag.get('accuracy'),
            'error':       r.get('error'),
        })
    return summary


def print_summary(summary: dict):
    print(f"\n{'='*88}")
    print(f"  Exp 2 Summary -- Scale vs Weight Bits")
    print(f"{'='*88}")
    print(f"  {'Model':<14} {'Quant':<8} {'Condition':<14} {'Flips':>5} "
          f"{'Seed':>4} {'PPL':>12} {'ARC':>8} {'Hella':>8}  {'Status':>6}")
    print(f"  {'-'*80}")

    prev_model = None
    for r in summary['runs']:
        if prev_model and r['model'] != prev_model:
            print()
        prev_model = r['model']

        ppl_str   = f"{r['ppl']:.3f}"       if r['ppl']       else "N/A"
        arc_str   = f"{r['arc_easy']*100:.1f}%" if r['arc_easy'] else "N/A"
        hella_str = f"{r['hellaswag']*100:.1f}%" if r['hellaswag'] else "N/A"
        status    = "ERR" if r['error'] else "OK"

        print(f"  {r['model']:<14} {r['quant']:<8} {r['condition']:<14} "
              f"{r['flip_count']:>5} {r['seed']:>4} {ppl_str:>12} "
              f"{arc_str:>8} {hella_str:>8}  {status:>6}")

    print(f"{'='*88}\n")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def should_skip(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        with open(output_path) as f:
            d = json.load(f)
        if d.get('error') is not None:
            return False
        return True  # ppl=None is valid -- means total corruption
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='BitBreaker Exp 2: Scale vs Weight Bits'
    )
    parser.add_argument('--platform', choices=['mac', 'grendel'], default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    platform = args.platform or detect_platform()

    # Select model list and seed count based on platform
    models = MODELS_MAC     if platform == 'mac' else MODELS_GRENDEL
    seeds  = SEEDS_MAC      if platform == 'mac' else SEEDS_GRENDEL

    # Count total runs
    total_runs = sum(
        len(conditions) * len(FLIP_COUNTS) * len(seeds)
        for _, _, _, conditions in models
    )

    print(f"\n{'='*60}")
    print(f"  BitBreaker -- Experiment 2: Scale vs Weight Bits")
    print(f"{'='*60}")
    print(f"  Platform    : {platform}")
    print(f"  Models      : {len(models)} ({platform} subset)" if platform == "mac" else f"  Models      : {len(models)}")
    print(f"  Flip counts : {FLIP_COUNTS}")
    print(f"  Seeds       : {seeds}")
    print(f"  Bit mode    : high-impact zone (all flips)")
    print(f"  Benchmarks  : PPL + ARC-Easy + HellaSwag")
    print(f"  Total runs  : {total_runs}")
    print(f"  Output      : {RESULTS_DIR}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  [DRY RUN] Conditions per model:")
        for model_file, label, quant, conditions in models:
            print(f"    {label:<14} {quant:<8} conditions: {[c for c,_ in conditions]}")
        print("\n  No files will be modified. Exiting.")
        return

    all_results  = []
    model_number = 0
    t_start      = time.time()

    for model_file, model_label, quant_label, conditions in models:
        model_number += 1
        model_path = PROJECT_ROOT / "models" / model_file
        map_path   = (
            PROJECT_ROOT / "experiments" / "maps"
            / f"{Path(model_file).stem}_map.json"
        )

        print(f"\n{'='*60}")
        print(f"  [{model_number}/{len(models)}] {model_label}  ({quant_label})")
        print(f"  Conditions: {[c for c,_ in conditions]}")
        print(f"{'='*60}")

        if not model_path.exists():
            print(f"  SKIP: model not found")
            continue
        if not map_path.exists():
            print(f"  SKIP: map not found -- run scripts/run_build_maps.py first")
            continue

        mm = ModelMap.load(str(map_path))
        print(f"  Loaded map: {len(mm.in_scope_tensors)} in-scope tensors")

        injector = FaultInjector(
            model_map      = mm,
            project_root   = PROJECT_ROOT,
            n_gpu_layers   = 99,
            platform       = platform,
            run_perplexity = True,
            run_tasks      = True,   # Tier 1: all benchmarks
        )

        condition_runs = len(conditions) * len(FLIP_COUNTS) * len(seeds)
        run_number     = 0

        for condition_name, region_filter in conditions:
            output_dir = RESULTS_DIR / model_label / f"condition_{condition_name}"

            for seed in seeds:
                for flip_count in FLIP_COUNTS:
                    run_number += 1
                    print(f"\n  [{run_number}/{condition_runs}]  "
                          f"condition={condition_name}  seed={seed}  flips={flip_count}")

                    result_path = (
                        output_dir / f"seed_{seed}" / f"flip_{flip_count}_results.json"
                    )

                    if should_skip(result_path):
                        print(f"    SKIP (already complete): {result_path.name}")
                        continue

                    result = injector.run(
                        flip_count    = flip_count,
                        seed          = seed,
                        region_filter = region_filter,
                        role_filter   = ROLE_FILTER,
                        layer_range   = None,
                        output_path   = result_path,
                        bit_position  = None,
                        high_impact   = True,   # all flips use high-impact zone
                    )

                    result['model_label']  = model_label
                    result['quant_label']  = quant_label
                    result['condition']    = condition_name
                    # re-save with labels so rebuild scripts can use them
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    all_results.append(result)

    # ── Rebuild summary from disk (captures skipped runs from prior sessions) ──
    summary_path = RESULTS_DIR / "exp2_summary.json"
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
        })
    summary = {'experiment': 'exp2_scale_vs_weight', 'runs': all_disk_results}
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start
    print_summary(summary)
    print(f"  Total time : {elapsed/60:.1f} min")
    print(f"  Summary    : {summary_path}")


if __name__ == '__main__':
    main()
