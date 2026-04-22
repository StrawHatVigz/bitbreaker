#!/usr/bin/env python3
"""
plot_exp2.py

Exp 2 degradation bar chart: scale-targeted flip count vs PPL
Layout: 2 rows (Qwen / Llama) x 4 cols (FP16 / Q8_0 / Q4_K_M / Q4_0)
Each subplot: grouped bars for flip=1,5,10 with seed 0 and seed 1 side by side
Log scale y-axis. Baseline reference line. Inf/null capped and annotated.

Usage:
    python plot_exp2.py --mac   <mac_summary.json>   --grendel <grendel_summary.json>
    python plot_exp2.py --mac   <mac_summary.json>              # mac only
    python plot_exp2.py         --grendel <grendel_summary.json>  # grendel only
"""

import argparse
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

CAP_VALUE   = 1e13          # PPL cap for inf/null — displayed with annotation
FLIP_COUNTS = [1, 5, 10]
MODELS      = ['qwen', 'llama']
QUANTS      = ['FP16', 'Q8_0', 'Q4_K_M', 'Q4_0']

# model label prefix → display name
MODEL_DISPLAY = {'qwen': 'Qwen 2.5-0.5B', 'llama': 'Llama 3.2-1B'}
QUANT_DISPLAY = {'FP16': 'FP16', 'Q8_0': 'Q8_0', 'Q4_K_M': 'Q4_K_M', 'Q4_0': 'Q4_0'}

# Baseline PPL per model×quant (clean inference, from project records)
BASELINES = {
    ('qwen',  'FP16'):   15.86,
    ('qwen',  'Q8_0'):   15.93,
    ('qwen',  'Q4_K_M'): 16.43,
    ('qwen',  'Q4_0'):   17.49,
    ('llama', 'FP16'):   13.88,
    ('llama', 'Q8_0'):   13.88,
    ('llama', 'Q4_K_M'): 14.33,
    ('llama', 'Q4_0'):   15.05,
}

SEED_COLORS  = ['#378ADD', '#D85A30']   # blue seed0, coral seed1
SEED_LABELS  = ['Seed 0', 'Seed 1']
BAR_WIDTH    = 0.35
GROUP_GAP    = 1.0   # x-distance between flip-count groups

# ── helpers ───────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        raw = f.read()
    # JSON spec doesn't allow Infinity — replace before parsing
    raw = raw.replace(': Infinity', ': 1e500').replace(':Infinity', ':1e500')
    return json.loads(raw)


def ppl_value(v):
    """Return float, capping inf/null at CAP_VALUE."""
    if v is None:
        return CAP_VALUE
    try:
        f = float(v)
        return CAP_VALUE if (math.isinf(f) or math.isnan(f) or f > CAP_VALUE) else f
    except Exception:
        return CAP_VALUE


def is_capped(v):
    if v is None:
        return True
    try:
        f = float(v)
        return math.isinf(f) or math.isnan(f) or f > CAP_VALUE
    except Exception:
        return True


def build_lookup(runs):
    """
    Returns dict: (model_prefix, quant, flip_count, seed) → ppl_raw
    model_prefix: 'qwen' or 'llama'
    """
    lut = {}
    for r in runs:
        model = r['model']       # e.g. 'llama_fp16', 'qwen_q4km'
        quant = r['quant']       # e.g. 'FP16', 'Q4_K_M'
        prefix = 'qwen' if model.startswith('qwen') else 'llama'
        key = (prefix, quant, r['flip_count'], r['seed'])
        lut[key] = r['ppl']
    return lut


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_platform(lut, platform_label, out_path):
    fig, axes = plt.subplots(
        2, 4,
        figsize=(16, 7),
        sharey=False,
    )
    fig.suptitle(
        f'Exp 2 — Scale-Targeted Bit-Flip Degradation  ({platform_label})',
        fontsize=13, fontweight='normal', y=1.01
    )

    for row, model in enumerate(MODELS):
        for col, quant in enumerate(QUANTS):
            ax = axes[row][col]

            baseline = BASELINES[(model, quant)]
            x_centers = np.arange(len(FLIP_COUNTS)) * GROUP_GAP

            for s_idx, seed in enumerate([0, 1]):
                x_pos = x_centers + (s_idx - 0.5) * BAR_WIDTH
                heights = []
                capped  = []
                for fc in FLIP_COUNTS:
                    raw = lut.get((model, quant, fc, seed))
                    heights.append(ppl_value(raw))
                    capped.append(is_capped(raw))

                bars = ax.bar(
                    x_pos, heights,
                    width=BAR_WIDTH,
                    color=SEED_COLORS[s_idx],
                    alpha=0.85,
                    label=SEED_LABELS[s_idx],
                    zorder=3,
                )

                # annotate capped bars
                for bar, cap in zip(bars, capped):
                    if cap:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.05,
                            '∞', ha='center', va='bottom',
                            fontsize=9, color='#444'
                        )

            # baseline reference line
            ax.axhline(baseline, color='#888', linewidth=1.0,
                       linestyle='--', zorder=2, label='Baseline')

            ax.set_yscale('log')
            ax.set_xticks(x_centers)
            ax.set_xticklabels([f'flip={fc}' for fc in FLIP_COUNTS], fontsize=9)
            ax.set_ylim(bottom=max(1, baseline * 0.5))
            ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)

            # titles
            if row == 0:
                ax.set_title(QUANT_DISPLAY[quant], fontsize=11, pad=6)
            if col == 0:
                ax.set_ylabel(
                    f'{MODEL_DISPLAY[model]}\nPPL (log scale)',
                    fontsize=9, labelpad=4
                )

    # shared legend
    handles = [
        mpatches.Patch(color=SEED_COLORS[0], label='Seed 0'),
        mpatches.Patch(color=SEED_COLORS[1], label='Seed 1'),
        plt.Line2D([0], [0], color='#888', linestyle='--', linewidth=1.0, label='Baseline'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mac',     type=Path, default=None)
    parser.add_argument('--grendel', type=Path, default=None)
    parser.add_argument('--outdir',  type=Path, default=Path('.'))
    args = parser.parse_args()

    if not args.mac and not args.grendel:
        parser.error('Provide at least one of --mac or --grendel')

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.mac:
        data = load(args.mac)
        lut  = build_lookup(data['runs'])
        plot_platform(lut, 'Apple M1', args.outdir / 'exp2_degradation_mac.png')

    if args.grendel:
        data = load(args.grendel)
        lut  = build_lookup(data['runs'])
        plot_platform(lut, 'NVIDIA L4 (Grendel)', args.outdir / 'exp2_degradation_grendel.png')


if __name__ == '__main__':
    main()
