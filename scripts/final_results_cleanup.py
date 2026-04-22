#!/usr/bin/env python3
"""
final_results_cleanup.py

Reads Exp 2, 3a, 3b summary JSONs and outputs formatted result tables
with a % worse than baseline column appended.

Usage:
    python final_results_cleanup.py --platform mac
    python final_results_cleanup.py --platform grendel
    python final_results_cleanup.py --platform mac --platform grendel

Output files written to ./results_tables/
    exp2_mac.txt, exp2_grendel.txt
    exp3a_mac.txt, exp3a_grendel.txt
    exp3b_mac.txt, exp3b_grendel.txt
"""

import argparse
import json
import math
from pathlib import Path

# ── Per-platform baselines (from actual baseline runs) ────────────────────────

BASELINES = {
    'mac': {
        'qwen_fp16':  15.8634,
        'qwen_q8':    15.9032,
        'qwen_q4km':  16.3498,
        'qwen_q4':    17.4864,
        'llama_fp16': 13.8805,
        'llama_q8':   13.8820,
        'llama_q4km': 14.3290,
        'llama_q4':   15.0513,
    },
    'grendel': {
        'qwen_fp16':  15.8667,
        'qwen_q8':    15.9529,
        'qwen_q4km':  16.4267,
        'qwen_q4':    17.5420,
        'llama_fp16': 13.8833,
        'llama_q8':   13.8902,
        'llama_q4km': 14.3588,
        'llama_q4':   15.0599,
    },
}

# ── Paths ─────────────────────────────────────────────────────────────────────

PLATFORM_BASES = {
    'mac': Path('/Users/viggy/Documents/Grad School/Sem_4/ECE 591 - SW HW Co-design'
                '/Project/bitbreaker/experiments/results/fault_injection'),
    'grendel': Path('/Users/viggy/Documents/Grad School/Sem_4/ECE 591 - SW HW Co-design'
                '/Project/bitbreaker/experiments/results_grendel/fault_injection')
}

EXP_FILES = {
    'exp1':  'exp1_random_sweep/exp1_summary.json',
    'exp2':  'exp2_scale_vs_weight/exp2_summary.json',
    'exp3a': 'exp3a_role_targeting/exp3a_summary.json',
    'exp3b': 'exp3b_depth_targeting/exp3b_depth_targeting_summary.json',
}

OUT_DIR = Path('results_tables')

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        raw = f.read()
    raw = raw.replace(': Infinity', ': 1e500').replace(':Infinity', ':1e500')
    return json.loads(raw)


def fmt_ppl(v) -> str:
    if v is None:
        return 'N/A'
    try:
        f = float(v)
        if math.isinf(f) or math.isnan(f) or f > 1e13:
            return 'inf/nan'
        if f > 1e6:
            return f'{f:.3e}'
        return f'{f:.3f}'
    except Exception:
        return 'N/A'


def fmt_pct_metric(v) -> str:
    if v is None:
        return 'N/A'
    return f'{float(v) * 100:.1f}%'


def pct_worse(ppl, model_key: str, platform: str) -> str:
    baseline = BASELINES[platform].get(model_key)
    if baseline is None or ppl is None:
        return 'N/A'
    try:
        f = float(ppl)
        if math.isinf(f) or math.isnan(f) or f > 1e13:
            return 'inf'
        pct = (f - baseline) / baseline * 100
        if pct > 1e6:
            return f'{pct:.2e}%'
        return f'{pct:+.1f}%'
    except Exception:
        return 'N/A'


def sep(width=110):
    return '  ' + '-' * width

def write_exp1(runs: list, platform: str, out_path: Path):
    lines = []
    lines.append('=' * 80)
    lines.append(f'  Exp 1 -- Random Weight-Bit Sweep  [{platform.upper()}]')
    lines.append('=' * 80)
    lines.append(
        f"  {'Model':<14} {'Quant':<8} {'Flips':>6} {'Seed':>4} "
        f"{'PPL':>14} {'% Worse':>14}  {'Status':<6}"
    )
    lines.append('  ' + '-' * 66)

    prev_model = None
    for r in sorted(runs, key=lambda x: (x['model'], x['flip_count'], x['seed'])):
        if prev_model and r['model'] != prev_model:
            lines.append('')
        prev_model = r['model']
        status = 'ERR' if r.get('error') else 'OK'
        lines.append(
            f"  {r['model']:<14} {r.get('quant') or '':<8} {r['flip_count']:>6} {r['seed']:>4} "
            f"{fmt_ppl(r['ppl']):>14} {pct_worse(r['ppl'], r['model'], platform):>14}  {status:<6}"
        )

    lines.append('=' * 80)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'  Written: {out_path}')

# ── Exp 2 table ───────────────────────────────────────────────────────────────

def write_exp2(runs: list, platform: str, out_path: Path):
    lines = []
    lines.append('=' * 114)
    lines.append(f'  Exp 2 -- Scale vs Weight Bits  [{platform.upper()}]')
    lines.append('=' * 114)
    lines.append(
        f"  {'Model':<14} {'Quant':<8} {'Condition':<14} {'Flips':>5} {'Seed':>4} "
        f"{'PPL':>14} {'% Worse':>14} {'ARC':>8} {'Hella':>8}  {'Status':<6}"
    )
    lines.append(sep())

    prev_model = None
    for r in sorted(runs, key=lambda x: (x['model'], x['flip_count'], x['seed'])):
        if prev_model and r['model'] != prev_model:
            lines.append('')
        prev_model = r['model']
        status = 'ERR' if r.get('error') else 'OK'
        lines.append(
            f"  {r['model']:<14} {r.get('quant') or '':<8} {r.get('condition') or '':<14} "
            f"{r['flip_count']:>5} {r['seed']:>4} "
            f"{fmt_ppl(r['ppl']):>14} {pct_worse(r['ppl'], r['model'], platform):>14} "
            f"{fmt_pct_metric(r.get('arc_easy')):>8} {fmt_pct_metric(r.get('hellaswag')):>8}  {status:<6}"
        )

    lines.append('=' * 114)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'  Written: {out_path}')


# ── Exp 3a table ──────────────────────────────────────────────────────────────

def write_exp3a(runs: list, platform: str, out_path: Path):
    lines = []
    lines.append('=' * 114)
    lines.append(f'  Exp 3a -- Role Targeting: Attn vs FFN  [{platform.upper()}]')
    lines.append('=' * 114)
    lines.append(
        f"  {'Model':<14} {'Quant':<8} {'Condition':<12} {'Region':<16} {'Seed':>4} "
        f"{'PPL':>14} {'% Worse':>14} {'ARC':>8} {'Hella':>8}  {'Status':<6}"
    )
    lines.append(sep())

    prev_model = None
    for r in sorted(runs, key=lambda x: (x['model'], x.get('condition', ''), x['seed'])):
        if prev_model and r['model'] != prev_model:
            lines.append('')
        prev_model = r['model']
        status = 'ERR' if r.get('error') else 'OK'
        region = r.get('region') or 'weights'
        lines.append(
            f"  {r['model']:<14} {r.get('quant') or '':<8} {r.get('condition') or '':<12} "
            f"{region:<16} {r['seed']:>4} "
            f"{fmt_ppl(r['ppl']):>14} {pct_worse(r['ppl'], r['model'], platform):>14} "
            f"{fmt_pct_metric(r.get('arc_easy')):>8} {fmt_pct_metric(r.get('hellaswag')):>8}  {status:<6}"
        )

    lines.append('=' * 114)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'  Written: {out_path}')


# ── Exp 3b table ──────────────────────────────────────────────────────────────

def write_exp3b(runs: list, platform: str, out_path: Path):
    lines = []
    lines.append('=' * 114)
    lines.append(f'  Exp 3b -- Depth Targeting: Early / Middle / Late  [{platform.upper()}]')
    lines.append('=' * 114)
    lines.append(
        f"  {'Model':<14} {'Quant':<8} {'Condition':<16} {'Layers':>12} {'Seed':>4} "
        f"{'PPL':>14} {'% Worse':>14} {'ARC':>8} {'Hella':>8}  {'Status':<6}"
    )
    lines.append(sep())

    prev_model = None
    for r in sorted(runs, key=lambda x: (x['model'], x.get('condition', ''), x['seed'])):
        if prev_model and r['model'] != prev_model:
            lines.append('')
        prev_model = r['model']
        status = 'ERR' if r.get('error') else 'OK'
        layer_str = str(r.get('layer_range') or 'N/A')
        lines.append(
            f"  {r['model']:<14} {r.get('quant') or '':<8} {r.get('condition') or '':<16} "
            f"{layer_str:>12} {r['seed']:>4} "
            f"{fmt_ppl(r['ppl']):>14} {pct_worse(r['ppl'], r['model'], platform):>14} "
            f"{fmt_pct_metric(r.get('arc_easy')):>8} {fmt_pct_metric(r.get('hellaswag')):>8}  {status:<6}"
        )

    lines.append('=' * 114)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'  Written: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', choices=['mac', 'grendel'],
                        action='append', dest='platforms', default=None)
    args = parser.parse_args()
    platforms = args.platforms or ['mac', 'grendel']

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    writers = {
        'exp1':  write_exp1,
        'exp2':  write_exp2,
        'exp3a': write_exp3a,
        'exp3b': write_exp3b,
    }

    for platform in platforms:
        base = PLATFORM_BASES[platform]
        print(f'\n[{platform.upper()}]  base: {base}')
        for exp, rel_path in EXP_FILES.items():
            json_path = base / rel_path
            if not json_path.exists():
                print(f'  SKIP {exp}: not found at {json_path}')
                continue
            data = load_json(json_path)
            runs = data.get('runs', [])
            out_path = OUT_DIR / f'{exp}_{platform}.txt'
            writers[exp](runs, platform, out_path)

    print(f'\nDone. Tables in ./{OUT_DIR}/')


if __name__ == '__main__':
    main()
