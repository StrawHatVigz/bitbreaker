#!/usr/bin/env python3
"""
run_calibration.py

Calibration experiment for BitBreaker.

Purpose: Validate whether PPL degradation reliably correlates with
task accuracy degradation at low flip counts, before committing to
the full tiered execution strategy.

Setup:
    Model       : Llama-3.2-1B-Instruct-Q4_K_M  (richest format — all region types)
    Flip counts : [1, 5, 10]
    Seeds       : [0, 1, 2]
    Region      : weight bits only (random pool: attn + ffn + output)
    Benchmarks  : PPL + ARC-Easy + HellaSwag (all 3)

Output structure:
    experiments/results/fault_injection/calibration/
        llama_q4km/
            seed_0/
                flip_1_results.json
                flip_5_results.json
                flip_10_results.json
            seed_1/
                ...
            seed_2/
                ...
        calibration_summary.json   ← aggregated table across all runs

Usage:
    # Auto-detect platform
    python scripts/run_calibration.py

    # Specify platform explicitly
    python scripts/run_calibration.py --platform grendel
    python scripts/run_calibration.py --platform mac

    # Dry run: print what would execute without touching model files
    python scripts/run_calibration.py --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Project root + imports ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "fault_injection"))

from model_map import ModelMap
from fault_injector import FaultInjector

# ─────────────────────────────────────────────
# Calibration config
# ─────────────────────────────────────────────

MODEL_FILE  = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_LABEL = "llama_q4km"
QUANT_LABEL = "Q4_K_M"

FLIP_COUNTS = [1, 5, 10]
SEEDS       = [0, 1, 2]

ROLE_FILTER   = ['attn', 'ffn', 'output']
REGION_FILTER = None    # None → weight bits (random pool)

RESULTS_BASE = (
    PROJECT_ROOT / "experiments" / "results" / "fault_injection" / "calibration"
)

# ─────────────────────────────────────────────
# Platform detection
# ─────────────────────────────────────────────

def detect_platform() -> str:
    import platform
    return 'mac' if platform.system() == 'Darwin' else 'grendel'

# ─────────────────────────────────────────────
# Summary builder
# ─────────────────────────────────────────────

def build_summary(all_results: list) -> dict:
    """Flatten all run results into a summary table keyed by (flip_count, seed)."""
    summary = {
        'model':   MODEL_LABEL,
        'quant':   QUANT_LABEL,
        'runs':    [],
    }

    for r in all_results:
        ppl       = r.get('perplexity') or {}
        tasks     = r.get('tasks') or {}
        arc_easy  = tasks.get('arc_easy') or {}
        hellaswag = tasks.get('hellaswag') or {}

        summary['runs'].append({
            'flip_count':  r['flip_count'],
            'seed':        r['seed'],
            'ppl':         ppl.get('ppl'),
            'ppl_err':     ppl.get('ppl_err'),
            'arc_easy':    arc_easy.get('accuracy'),
            'arc_n':       arc_easy.get('total'),
            'hellaswag':   hellaswag.get('accuracy'),
            'hellaswag_n': hellaswag.get('total'),
            'error':       r.get('error'),
        })

    return summary


def print_summary(summary: dict):
    """Print a readable summary table to stdout."""
    print(f"\n{'='*80}")
    print(f"  Calibration Summary — {summary['model']}  {summary['quant']}")
    print(f"{'='*80}")
    print(f"  {'Flips':>6}  {'Seed':>4}  {'PPL':>10}  {'ARC-Easy':>10}  {'HellaSwag':>10}  {'Status':>8}")
    print(f"  {'-'*62}")

    for r in summary['runs']:
        ppl_str   = f"{r['ppl']:.3f}"     if r['ppl']      else "N/A"
        arc_str   = f"{r['arc_easy']*100:.2f}%" if r['arc_easy'] else "N/A"
        hella_str = f"{r['hellaswag']*100:.2f}%" if r['hellaswag'] else "N/A"
        status    = "ERROR" if r['error'] else "OK"

        print(f"  {r['flip_count']:>6}  {r['seed']:>4}  {ppl_str:>10}  "
              f"{arc_str:>10}  {hella_str:>10}  {status:>8}")

    print(f"{'='*80}\n")


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
    parser = argparse.ArgumentParser(
        description='Run BitBreaker calibration experiment'
    )
    parser.add_argument(
        '--platform', choices=['mac', 'grendel'], default=None,
        help='Platform (auto-detected if not specified)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print experiment plan without touching model files'
    )
    parser.add_argument(
        '--region', default=None,
        choices=['weights', 'block_scale', 'sub_scales',
                 'super_scale_d', 'super_scale_dmin'],
        help='Region type to flip. Default: weights (random weight bits)'
    )
    parser.add_argument(
        '--flip-counts', nargs='+', type=int, default=None,
        help='Override flip counts. e.g. --flip-counts 1 5 10 50'
    )
    parser.add_argument(
        '--bit-position', type=int, default=None,
        help=(
            'Word-level bit index for the FIRST flip only. '
            'For FP16 scales: 14=exponent MSB (max damage), 15=sign, 9=mantissa MSB. '
            'Remaining flips use random bit positions. Default: all flips random.'
        )
    )
    args = parser.parse_args()

    platform     = args.platform or detect_platform()
    flip_counts  = args.flip_counts or FLIP_COUNTS
    region       = args.region     or REGION_FILTER   # None = weight bits
    bit_position = args.bit_position

    # ── Resolve paths ─────────────────────────────────────────────────────────
    model_path = PROJECT_ROOT / "models" / MODEL_FILE
    map_path   = PROJECT_ROOT / "experiments" / "maps" / f"{Path(MODEL_FILE).stem}_map.json"

    # Output dir encodes the region so targeted runs don't overwrite weight runs
    region_tag = region if region else "weights"
    bit_tag    = f"_bit{bit_position}" if bit_position is not None else ""
    output_dir = RESULTS_BASE / MODEL_LABEL / f"{region_tag}{bit_tag}"

    # ── Validate ──────────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not map_path.exists():
        print(f"ERROR: map not found: {map_path}", file=sys.stderr)
        print(f"       Run scripts/run_build_maps.py first", file=sys.stderr)
        sys.exit(1)

    # ── Print plan ────────────────────────────────────────────────────────────
    total_runs = len(flip_counts) * len(SEEDS)
    print(f"\n{'='*60}")
    print(f"  BitBreaker — Calibration Experiment")
    print(f"{'='*60}")
    print(f"  Model    : {MODEL_FILE}")
    print(f"  Quant    : {QUANT_LABEL}")
    print(f"  Platform : {platform}")
    print(f"  Flips    : {flip_counts}")
    print(f"  Seeds    : {SEEDS}")
    print(f"  Region   : {region_tag}")
    print(f"  Bit pos  : {bit_position if bit_position is not None else 'random'}")
    print(f"  Benchmarks: PPL + ARC-Easy + HellaSwag")
    print(f"  Total runs: {total_runs}")
    print(f"  Output   : {output_dir}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  [DRY RUN] No files will be modified. Exiting.")
        return

    # ── Load map ──────────────────────────────────────────────────────────────
    print(f"\n  Loading model map...")
    mm = ModelMap.load(str(map_path))
    print(f"  Loaded: {len(mm.in_scope_tensors)} in-scope tensors")

    # ── Build injector ────────────────────────────────────────────────────────
    injector = FaultInjector(
        model_map      = mm,
        project_root   = PROJECT_ROOT,
        n_gpu_layers   = 99,
        platform       = platform,
        run_perplexity = True,
        run_tasks      = True,
    )

    # ── Run experiments ───────────────────────────────────────────────────────
    all_results = []
    run_number  = 0
    t_start     = time.time()

    for seed in SEEDS:
        for flip_count in flip_counts:
            run_number += 1
            print(f"\n  [{run_number}/{total_runs}]  seed={seed}  flips={flip_count}")

            result_path = output_dir / f"seed_{seed}" / f"flip_{flip_count}_results.json"

            if should_skip(result_path):
                print(f"    SKIP (already complete): {result_path.name}")
                continue

            result = injector.run(
                flip_count    = flip_count,
                seed          = seed,
                region_filter = region,
                role_filter   = ROLE_FILTER,
                layer_range   = None,
                output_path   = result_path,
                bit_position  = bit_position,
            )

            all_results.append(result)

    # ── Save and print summary ────────────────────────────────────────────────
    summary      = build_summary(all_results)
    summary_path = RESULTS_BASE / f"calibration_summary_{region_tag}{bit_tag}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start
    print_summary(summary)
    print(f"  Total time : {elapsed/60:.1f} min")
    print(f"  Summary    : {summary_path}")
    print(f"  Run results: {output_dir}/seed_*/flip_*_results.json")


if __name__ == '__main__':
    main()