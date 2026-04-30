#!/usr/bin/env python3
"""
run_exp3b.py

Experiment 3b: Depth Targeting -- Early vs Late Layers

Goal: Determine whether faults in early transformer layers cause more damage
      than faults in late layers. Tests the "fault amplification through depth"
      hypothesis — early layer corruption propagates through all subsequent
      layers; late layer corruption only affects the final output projection.

Layer ranges (computed from ModelMap at runtime):
    early: first 25% of transformer blocks  → mm.get_early_layer_range()
    late:  last  25% of transformer blocks  → mm.get_late_layer_range()

Bit selection per format (high_impact=True, consistent with Exp 2/3a):
    FP16  weights     : exponent zone bits 10-14
    Q8_0  block_scale : exponent zone bits 10-14
    Q4_K_M super_scale: exponent zone bits 10-14
    Q4_0  block_scale : exponent zone bits 10-14

Design notes:
    - All roles (attn + ffn) targeted together — isolating role is Exp 3a's job
    - Flip count fixed at 10 to match Exp 3a for direct comparison
    - Layer ranges are printed at runtime so you can verify against model architecture
    - Grendel: 3 seeds; Mac: 2 seeds

Output:
    experiments/results/fault_injection/exp3b_depth_targeting/
        qwen_fp16/condition_early_layers/seed_0/flip_10_results.json ...
        qwen_fp16/condition_late_layers/...
        qwen_q8/...
        ...
        exp3b_summary.json

Usage:
    python scripts/run_exp3b.py --platform grendel
    python scripts/run_exp3b.py --platform mac
    python scripts/run_exp3b.py --platform grendel --dry-run
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

FLIP_COUNT  = 10  # fixed, matches Exp 3a for direct comparison
ROLE_FILTER = ['attn', 'ffn']  # all roles — isolating role is Exp 3a's job

RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "fault_injection" / "exp3b_depth_targeting"
)

# (model_file, model_label, quant_label, region_filter)
# region_filter: most vulnerable region confirmed in Exp 2
MODELS = [
    ("qwen2.5-0.5b-instruct-fp16.gguf",   "qwen_fp16",  "FP16",   None),
    ("qwen2.5-0.5b-instruct-q8_0.gguf",   "qwen_q8",    "Q8_0",   "block_scale"),
    ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen_q4km",  "Q4_K_M", "super_scale_d"),
    ("qwen2.5-0.5b-instruct-q4_0.gguf",   "qwen_q4",    "Q4_0",   "block_scale"),
    ("Llama-3.2-1B-Instruct-f16.gguf",    "llama_fp16", "FP16",   None),
    ("Llama-3.2-1B-Instruct-Q8_0.gguf",   "llama_q8",   "Q8_0",   "block_scale"),
    ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "llama_q4km", "Q4_K_M", "super_scale_d"),
    ("Llama-3.2-1B-Instruct-Q4_0.gguf",   "llama_q4",   "Q4_0",   "block_scale"),
]

SEEDS_GRENDEL = [0, 1, 2]
SEEDS_MAC     = [0, 1]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def detect_platform() -> str:
    return 'mac' if platform_lib.system() == 'Darwin' else 'linux'


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
    summary = {'experiment': 'exp3b_depth_targeting', 'runs': []}
    for r in all_results:
        ppl       = (r.get('perplexity') or {}).get('ppl')
        tasks     = r.get('tasks') or {}
        arc_easy  = (tasks.get('arc_easy')  or {}).get('accuracy')
        hellaswag = (tasks.get('hellaswag') or {}).get('accuracy')
        summary['runs'].append({
            'model':       r.get('model_label'),
            'quant':       r.get('quant_label'),
            'condition':   r.get('condition'),
            'layer_range': r.get('layer_range'),
            'flip_count':  r['flip_count'],
            'seed':        r['seed'],
            'ppl':         ppl,
            'arc_easy':    arc_easy,
            'hellaswag':   hellaswag,
            'error':       r.get('error'),
        })
    return summary


def print_summary(summary: dict):
    print(f"\n{'='*100}")
    print(f"  Exp 3b Summary -- Depth Targeting: Early vs Late Layers")
    print(f"{'='*100}")
    print(f"  {'Model':<14} {'Quant':<8} {'Condition':<14} {'Layers':>12} {'Seed':>4} "
          f"{'PPL':>14} {'ARC':>8} {'Hella':>8}  {'Status':>6}")
    print(f"  {'-'*92}")

    prev_model = None
    for r in summary['runs']:
        if prev_model and r['model'] != prev_model:
            print()
        prev_model = r['model']

        layer_str = str(r['layer_range']) if r['layer_range'] else "N/A"
        ppl_str   = f"{r['ppl']:.3f}" if isinstance(r['ppl'], float) else "inf/nan"
        arc_str   = f"{r['arc_easy']*100:.1f}%"  if r['arc_easy']   else "N/A"
        hella_str = f"{r['hellaswag']*100:.1f}%" if r['hellaswag']  else "N/A"
        status    = "ERR" if r['error'] else "OK"

        print(f"  {r['model']:<14} {r['quant']:<8} {r['condition']:<14} {layer_str:>12} "
              f"{r['seed']:>4} {ppl_str:>14} {arc_str:>8} {hella_str:>8}  {status:>6}")

    print(f"{'='*100}\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='BitBreaker Exp 3b: Depth Targeting -- Early vs Late Layers'
    )
    parser.add_argument('--platform', choices=['mac', 'grendel', 'linux'], default=None,
                        help="'mac'=Apple Silicon Metal, 'grendel'=NCSU GPU cluster, "
                             "'linux'=any other Linux/CUDA machine")
    parser.add_argument('--n-gpu-layers', type=int, default=99,
                        help='GPU layers to offload (default 99). Use 0 for CPU-only.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    platform = args.platform or detect_platform()
    seeds    = SEEDS_MAC if platform == 'mac' else SEEDS_GRENDEL

    total_runs = len(MODELS) * 3 * len(seeds)  # 3 depth conditions per model

    print(f"\n{'='*60}")
    print(f"  BitBreaker -- Experiment 3b: Depth Targeting")
    print(f"{'='*60}")
    print(f"  Platform    : {platform}")
    print(f"  Models      : {len(MODELS)}")
    print(f"  Conditions  : early_layers (first 25%), middle_layers (mid 50%), late_layers (last 25%)")
    print(f"  Flip count  : {FLIP_COUNT} (fixed)")
    print(f"  Seeds       : {seeds}")
    print(f"  Roles       : attn + ffn (all)")
    print(f"  Bit mode    : high-impact zone")
    print(f"  Benchmarks  : PPL + ARC-Easy + HellaSwag")
    print(f"  Total runs  : {total_runs}")
    print(f"  Output      : {RESULTS_DIR}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  [DRY RUN] Layer ranges per model (from ModelMap):")
        for model_file, label, quant, region in MODELS:
            map_path = (
                PROJECT_ROOT / "experiments" / "maps"
                / f"{Path(model_file).stem}_map.json"
            )
            if map_path.exists():
                mm = ModelMap.load(str(map_path))
                early = mm.get_early_layer_range()
                late  = mm.get_late_layer_range()
                middle = (early[1] + 1, late[0] - 1)
                print(f"    {label:<14} {quant:<8}  early={early}  middle={middle}  late={late}  "
                f"region={region or 'weights'}")
                
            else:
                print(f"    {label:<14} {quant:<8}  map not found")
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
        print(f"  Region: {region_filter or 'weights'}")
        print(f"{'='*60}")

        if not model_path.exists():
            print(f"  SKIP: model not found at {model_path}")
            continue
        if not map_path.exists():
            print(f"  SKIP: map not found — run scripts/run_build_maps.py first")
            continue

        mm          = ModelMap.load(str(map_path))
        early_range = mm.get_early_layer_range()
        late_range  = mm.get_late_layer_range()
        middle_range = (early_range[1] + 1, late_range[0] - 1)
        print(f"  Loaded map: {len(mm.in_scope_tensors)} in-scope tensors, "
              f"{mm.n_layers} layers")
        print(f"  Layer ranges: early={early_range}  late={late_range}")


        # Build conditions at runtime using actual layer ranges from this model's map
        depth_conditions = [
            ("early_layers", early_range),
            ("middle_layers", middle_range),
            ("late_layers",  late_range),
        ]

        injector = FaultInjector(
            model_map      = mm,
            project_root   = PROJECT_ROOT,
            n_gpu_layers   = args.n_gpu_layers,
            platform       = platform,
            run_perplexity = True,
            run_tasks      = True,
        )

        condition_runs = len(depth_conditions) * len(seeds)
        run_number     = 0

        for condition_name, layer_range in depth_conditions:
            output_dir = RESULTS_DIR / model_label / f"condition_{condition_name}"

            for seed in seeds:
                run_number += 1
                print(f"\n  [{run_number}/{condition_runs}]  "
                      f"condition={condition_name}  layers={layer_range}  seed={seed}")

                result_path = (
                    output_dir / f"seed_{seed}" / f"flip_{FLIP_COUNT}_results.json"
                )

                if should_skip(result_path):
                    print(f"    SKIP (already complete): {result_path.name}")
                    continue

                result = injector.run(
                    flip_count    = FLIP_COUNT,
                    seed          = seed,
                    region_filter = region_filter,
                    role_filter   = ROLE_FILTER,       # attn + ffn
                    layer_range   = layer_range,       # (lo, hi) for this model
                    output_path   = result_path,
                    bit_position  = None,
                    high_impact   = True,
                )

                result['model_label'] = model_label
                result['quant_label'] = quant_label
                result['condition']   = condition_name
                # re-save with labels so rebuild scripts can use them
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                all_results.append(result)

    # ── Rebuild summary from disk (captures skipped runs from prior sessions) ──
    summary_path = RESULTS_DIR / "exp3b_depth_targeting_summary.json"
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
            'layer_range': d.get('layer_range'),
            'flip_count': d['flip_count'],
            'seed':       d['seed'],
            'ppl':        (d.get('perplexity') or {}).get('ppl'),
            'arc_easy':   ((d.get('tasks') or {}).get('arc_easy')  or {}).get('accuracy'),
            'hellaswag':  ((d.get('tasks') or {}).get('hellaswag') or {}).get('accuracy'),
            'error':      d.get('error'),
        })
    summary = {'experiment': 'exp3b_depth_targeting', 'runs': all_disk_results}
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start
    print_summary(summary)
    print(f"  Total time : {elapsed/60:.1f} min")
    print(f"  Summary    : {summary_path}")


if __name__ == '__main__':
    main()
