#!/usr/bin/env python3
"""
run_exp1.py

Experiment 1: Random Flip Sweep -- Baseline Degradation Curves

Goal: Find the flip count at which random untargeted bit flips start degrading
each quantization format. Produces headline degradation curves (PPL vs flip count).

Setup:
    Models      : All 8 (Qwen 0.5B + Llama 1B x FP16, Q8_0, Q4_K_M, Q4_0)
    Flip counts : [25, 50, 100, 250, 500, 1000]
    Seeds       : [0, 1, 2]
    Region      : Weight bits only -- fully random (attn + ffn + output)
    Bit position: Fully random -- no targeting
    Benchmarks  : PPL only (Tier 2)

Total runs: 8 models x 6 counts x 3 seeds = 144 eval runs (~1.6hrs on Grendel)

Output:
    experiments/results/fault_injection/exp1_random_sweep/
        qwen_fp16/seed_0/flip_25_results.json  ...
        qwen_q8/...
        llama_fp16/...
        ...
        exp1_summary.json

Usage:
    python scripts/run_exp1.py --platform grendel
    python scripts/run_exp1.py --platform mac
    python scripts/run_exp1.py --platform grendel --dry-run
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

FLIP_COUNTS = [25, 50, 100, 250, 500, 1000]
SEEDS       = [0, 1, 2]
ROLE_FILTER = ['attn', 'ffn', 'output']

RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "fault_injection" / "exp1_random_sweep"
)

# model filename : label : quant_label
MODELS = [
    ("qwen2.5-0.5b-instruct-fp16.gguf",    "qwen_fp16",  "FP16"),
    ("qwen2.5-0.5b-instruct-q8_0.gguf",    "qwen_q8",    "Q8_0"),
    ("qwen2.5-0.5b-instruct-q4_k_m.gguf",  "qwen_q4km",  "Q4_K_M"),
    ("qwen2.5-0.5b-instruct-q4_0.gguf",    "qwen_q4",    "Q4_0"),
    ("Llama-3.2-1B-Instruct-f16.gguf",     "llama_fp16", "FP16"),
    ("Llama-3.2-1B-Instruct-Q8_0.gguf",    "llama_q8",   "Q8_0"),
    ("Llama-3.2-1B-Instruct-Q4_K_M.gguf",  "llama_q4km", "Q4_K_M"),
    ("Llama-3.2-1B-Instruct-Q4_0.gguf",    "llama_q4",   "Q4_0"),
]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def detect_platform() -> str:
    return 'mac' if platform_lib.system() == 'Darwin' else 'grendel'


def build_summary(all_results: list) -> dict:
    summary = {'experiment': 'exp1_random_sweep', 'runs': []}
    for r in all_results:
        ppl = (r.get('perplexity') or {}).get('ppl')
        summary['runs'].append({
            'model':      r.get('model_label'),
            'quant':      r.get('quant_label'),
            'flip_count': r['flip_count'],
            'seed':       r['seed'],
            'ppl':        ppl,
            'error':      r.get('error'),
        })
    return summary


def print_summary(summary: dict):
    print(f"\n{'='*72}")
    print(f"  Exp 1 Summary -- Random Flip Sweep")
    print(f"{'='*72}")
    print(f"  {'Model':<14} {'Quant':<8} {'Flips':>6} {'Seed':>4} {'PPL':>12}  {'Status':>6}")
    print(f"  {'-'*60}")

    prev_model = None
    for r in summary['runs']:
        if prev_model and r['model'] != prev_model:
            print()
        prev_model = r['model']

        ppl_str = f"{r['ppl']:.3f}" if r['ppl'] else "N/A"
        status  = "ERR" if r['error'] else "OK"
        print(f"  {r['model']:<14} {r['quant']:<8} {r['flip_count']:>6} "
              f"{r['seed']:>4} {ppl_str:>12}  {status:>6}")

    print(f"{'='*72}\n")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def should_skip(output_path: Path) -> bool:
    """
    Skip this run if result already exists, has no error, and has a valid PPL.
    Prevents redundant re-runs when resuming interrupted experiments.
    """
    if not output_path.exists():
        return False
    try:
        with open(output_path) as f:
            d = json.load(f)
        ppl = (d.get('perplexity') or {}).get('ppl')
        has_error = d.get('error') is not None
        if has_error:
            return False
        if ppl is None:
            return False
        return True
    except Exception:
        return False



def main():
    parser = argparse.ArgumentParser(description='BitBreaker Exp 1: Random Flip Sweep')
    parser.add_argument('--platform', choices=['mac', 'grendel'], default=None)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without touching model files')
    args = parser.parse_args()

    platform    = args.platform or detect_platform()
    total_runs  = len(MODELS) * len(FLIP_COUNTS) * len(SEEDS)

    print(f"\n{'='*60}")
    print(f"  BitBreaker -- Experiment 1: Random Flip Sweep")
    print(f"{'='*60}")
    print(f"  Platform    : {platform}")
    print(f"  Models      : {len(MODELS)}")
    print(f"  Flip counts : {FLIP_COUNTS}")
    print(f"  Seeds       : {SEEDS}")
    print(f"  Region      : weight bits (fully random)")
    print(f"  Benchmarks  : PPL only")
    print(f"  Total runs  : {total_runs}")
    print(f"  Output      : {RESULTS_DIR}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  [DRY RUN] No files will be modified. Exiting.")
        return

    all_results = []
    model_number = 0
    t_start = time.time()

    for model_file, model_label, quant_label in MODELS:
        model_number += 1
        model_path = PROJECT_ROOT / "models" / model_file
        map_path   = (
            PROJECT_ROOT / "experiments" / "maps"
            / f"{Path(model_file).stem}_map.json"
        )
        output_dir = RESULTS_DIR / model_label

        print(f"\n{'='*60}")
        print(f"  [{model_number}/{len(MODELS)}] {model_label}  ({quant_label})")
        print(f"{'='*60}")

        # Skip missing models gracefully
        if not model_path.exists():
            print(f"  SKIP: model not found: {model_path}")
            continue
        if not map_path.exists():
            print(f"  SKIP: map not found: {map_path}")
            print(f"        Run scripts/run_build_maps.py first")
            continue

        mm = ModelMap.load(str(map_path))
        print(f"  Loaded map: {len(mm.in_scope_tensors)} in-scope tensors")

        injector = FaultInjector(
            model_map      = mm,
            project_root   = PROJECT_ROOT,
            n_gpu_layers   = 99,
            platform       = platform,
            run_perplexity = True,
            run_tasks      = False,   # Tier 2: PPL only
        )

        run_number = 0
        model_runs = len(FLIP_COUNTS) * len(SEEDS)

        for seed in SEEDS:
            for flip_count in FLIP_COUNTS:
                run_number += 1
                print(f"\n  [{run_number}/{model_runs}]  seed={seed}  flips={flip_count}")

                result_path = output_dir / f"seed_{seed}" / f"flip_{flip_count}_results.json"

                if should_skip(result_path):
                    print(f"    SKIP (already complete): {result_path.name}")
                    continue

                result = injector.run(
                    flip_count    = flip_count,
                    seed          = seed,
                    region_filter = None,       # weight bits only
                    role_filter   = ROLE_FILTER,
                    layer_range   = None,
                    output_path   = result_path,
                    bit_position  = None,       # fully random
                )

                result['model_label'] = model_label
                result['quant_label'] = quant_label
                all_results.append(result)

    # ── Save and print summary ────────────────────────────────────────────────
    summary      = build_summary(all_results)
    summary_path = RESULTS_DIR / "exp1_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start
    print_summary(summary)
    print(f"  Total time : {elapsed/60:.1f} min")
    print(f"  Summary    : {summary_path}")


if __name__ == '__main__':
    main()