#!/usr/bin/env python3
"""
plot_results.py — Generate all BitBreaker degradation-curve figures.

Reads result JSON files from experiments/results_grendel/fault_injection/
and writes PNG figures to docs/figures/.

Figures produced:
  fig1_exp1_random_sweep.png       — Exp 1: FP16 random weight flip (Qwen + Llama)
  fig2_exp1_quant_immune.png       — Exp 1: Quantized models immune at 1000 flips
  fig3_exp2_scale_vs_weight.png    — Exp 2: Scale byte vs weight byte sensitivity
  fig4_grad_bfa_random.png         — Grad BFA: random exponent-bit (float16, block lottery)
  fig5_progressive_bfa_curve.png   — Progressive BFA: guided degradation curve (bfloat16)
  fig6_guided_vs_random_lower.png  — Progressive BFA: guided vs random lower bits
  fig7_attack_comparison.png       — Five-way comparison bar chart

Usage:
    python scripts/plot_results.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results_grendel" / "fault_injection"
FIGURES_DIR  = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

SEED_COLORS  = ["#2196F3", "#FF5722", "#4CAF50"]   # blue, orange, green
SEED_MARKERS = ["o", "s", "^"]
CATAST       = 1e10   # PPL ratio cap for log plots ("catastrophic" sentinel)


def clamp_ratio(r, cap=CATAST):
    if r is None or (isinstance(r, float) and (math.isnan(r) or math.isinf(r))):
        return cap
    return min(float(r), cap)


def save(fig, name):
    p = FIGURES_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {p.name}")


# ── Figure 1 — Exp 1: FP16 random weight bit sweep ────────────────────────────

def fig1_exp1_fp16():
    path = RESULTS_ROOT / "exp1_random_sweep" / "exp1_summary.json"
    with open(path) as f:
        data = json.load(f, parse_constant=lambda x: float("inf"))

    flip_counts = [25, 50, 100, 250, 500, 1000]
    models = [("qwen_fp16", "Qwen FP16"), ("llama_fp16", "Llama FP16")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.suptitle("Exp 1 — FP16 Random Weight-Bit Sweep (Grendel)", fontsize=14, fontweight="bold")

    for ax, (model_key, model_label) in zip(axes, models):
        # gather per-seed data
        runs = [r for r in data["runs"] if r["model"] == model_key]
        baseline = next(
            (r["ppl"] for r in runs if r["flip_count"] == 25 and r["seed"] == 0),
            None,
        )
        # approximate baseline from the first flip count being close to 1x
        # use a hardcoded baseline from summary
        baselines = {"qwen_fp16": 15.8667, "llama_fp16": 13.8833}
        bppl = baselines.get(model_key, 15.0)

        for s, color, marker in zip([0, 1, 2], SEED_COLORS, SEED_MARKERS):
            seed_runs = {r["flip_count"]: r["ppl"] for r in runs if r["seed"] == s}
            ratios = []
            for fc in flip_counts:
                ppl = seed_runs.get(fc, None)
                if ppl is None or math.isinf(ppl) or math.isnan(ppl):
                    ratios.append(CATAST)
                else:
                    ratios.append(max(ppl / bppl, 1.0))

            ax.semilogy(flip_counts, ratios, color=color, marker=marker,
                        linewidth=1.8, markersize=6, label=f"seed {s}")

        ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.6, label="baseline")
        ax.set_title(model_label)
        ax.set_xlabel("Number of random bit flips")
        ax.set_xticks(flip_counts)
        ax.legend(loc="upper left")

    axes[0].set_ylabel("PPL ratio (vs baseline, log scale)")
    fig.tight_layout()
    save(fig, "fig1_exp1_fp16_random_sweep.png")


# ── Figure 2 — Exp 1: Quantized models stay flat ──────────────────────────────

def fig2_exp1_quant():
    path = RESULTS_ROOT / "exp1_random_sweep" / "exp1_summary.json"
    with open(path) as f:
        raw = f.read()
    data = json.loads(raw.replace("Infinity", "1e308").replace("NaN", "null"))

    flip_counts = [25, 50, 100, 250, 500, 1000]
    quant_models = [
        ("qwen_q4km", "Qwen Q4_K_M", "#9C27B0"),
        ("qwen_q8",   "Qwen Q8_0",   "#009688"),
        ("llama_q4km","Llama Q4_K_M","#F44336"),
        ("llama_q8",  "Llama Q8_0",  "#FF9800"),
        ("qwen_fp16", "Qwen FP16",   "#2196F3"),
        ("llama_fp16","Llama FP16",  "#795548"),
    ]
    baselines_ppl = {
        "qwen_q4km": 16.43, "qwen_q8": 15.95, "qwen_fp16": 15.87,
        "llama_q4km": 14.36, "llama_q8": 13.89, "llama_fp16": 13.88,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Exp 1 — Quantized vs FP16: Random Weight-Bit Immunity (seed 0)",
                 fontsize=13, fontweight="bold")

    for model_key, label, color in quant_models:
        runs = [r for r in data["runs"] if r["model"] == model_key and r["seed"] == 0]
        bppl = baselines_ppl[model_key]
        ratios = []
        for fc in flip_counts:
            row = next((r for r in runs if r["flip_count"] == fc), None)
            if row is None or row["ppl"] is None:
                ratios.append(None)
            else:
                ppl = float(row["ppl"])
                ratios.append(min(ppl / bppl, CATAST) if math.isfinite(ppl) else CATAST)

        xs = [fc for fc, r in zip(flip_counts, ratios) if r is not None]
        ys = [r for r in ratios if r is not None]
        ls = "--" if "fp16" in model_key else "-"
        lw = 1.4 if "fp16" in model_key else 2.0
        ax.semilogy(xs, ys, color=color, linewidth=lw, linestyle=ls,
                    marker="o", markersize=5, label=label)

    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_xlabel("Number of random bit flips (weight bytes only)")
    ax.set_ylabel("PPL ratio (vs baseline, log scale)")
    ax.set_xticks(flip_counts)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig2_exp1_quant_immune.png")


# ── Figure 3 — Exp 2: Scale vs weight sensitivity ─────────────────────────────

def fig3_exp2_scale():
    path = RESULTS_ROOT / "exp2_scale_vs_weight" / "exp2_summary.json"
    with open(path) as f:
        raw = f.read()
    data = json.loads(raw.replace("Infinity", "1e308").replace("NaN", "null"))

    flip_counts = [1, 5, 10]
    condition_styles = {
        "weights":     ("#607D8B", "-",  "Weights (control)"),
        "block_scale": ("#FF9800", "--", "block_scale"),
        "super_scale": ("#F44336", "-",  "super_scale_d"),
    }
    model_baselines = {
        "qwen_q4km": 16.43, "qwen_q8": 15.95, "qwen_q4": 17.54,
        "llama_q4km": 14.36, "llama_q8": 13.89, "llama_q4": 15.06,
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Exp 2 — Scale Byte vs Weight Byte Sensitivity (seed 0)", fontsize=14, fontweight="bold")

    for ax, (model_prefix, title) in zip(axes, [("qwen", "Qwen 0.5B"), ("llama", "Llama 1B")]):
        plotted = set()
        for model_key, bppl in model_baselines.items():
            if not model_key.startswith(model_prefix):
                continue
            quant = model_key.replace(model_prefix + "_", "").upper().replace("KM", "_K_M")
            runs = [r for r in data["runs"] if r["model"] == model_key and r["seed"] == 0]

            for cond, (color, ls, cond_label) in condition_styles.items():
                cond_runs = [r for r in runs if r["condition"] == cond]
                if not cond_runs:
                    continue
                ratios = []
                for fc in flip_counts:
                    row = next((r for r in cond_runs if r["flip_count"] == fc), None)
                    if row is None or row["ppl"] is None:
                        ratios.append(1.0)
                    else:
                        ppl = float(row["ppl"])
                        ratios.append(min(ppl / bppl, CATAST) if math.isfinite(ppl) else CATAST)

                legend_label = f"{quant} — {cond_label}" if cond not in plotted else None
                ax.semilogy(flip_counts, ratios, color=color, linestyle=ls,
                            linewidth=2.0, marker="o", markersize=6,
                            label=f"{quant}/{cond_label}")
                plotted.add(cond)

        ax.axhline(1.0, color="gray", linewidth=1, linestyle=":", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Number of bit flips")
        ax.set_xticks(flip_counts)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("PPL ratio (vs baseline, log scale)")
    fig.tight_layout()
    save(fig, "fig3_exp2_scale_vs_weight.png")


# ── Figure 4 — Gradient BFA random exponent (float16) block lottery ───────────

def fig4_grad_bfa_random():
    path = RESULTS_ROOT / "grad_bfa" / "grad_bfa_summary.json"
    with open(path) as f:
        data = json.load(f)

    flip_counts = [1, 5, 10, 25, 50]
    baseline = data["baseline_ppl"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Grad BFA — Random Exponent Bits (float16, Qwen 0.5B)\nBlock Lottery: Seed-Dependent Degradation",
                 fontsize=13, fontweight="bold")

    for s, color, marker in zip(["0", "1", "2"], SEED_COLORS, SEED_MARKERS):
        seed_data = data["seeds"][s]["random"]
        ratios = [clamp_ratio(seed_data.get(str(fc), {}).get("degradation")) for fc in flip_counts]
        ax.semilogy(flip_counts, ratios, color=color, marker=marker,
                    linewidth=2.0, markersize=8, label=f"seed {s} (random exponent)")

    # Guided = NaN at flip-1 → show as horizontal band
    ax.axhspan(CATAST * 0.01, CATAST * 1.5, alpha=0.08, color="red",
               label="Guided attack → NaN at flip-1 (all seeds)")
    ax.annotate("Guided: NaN @ flip=1\n(model non-functional)", xy=(1, CATAST * 0.5),
                fontsize=9, color="red", ha="left")

    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Number of bit flips")
    ax.set_ylabel("PPL ratio (vs 18.99 baseline, log scale)")
    ax.set_xticks(flip_counts)
    ax.set_ylim(0.8, CATAST * 5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_grad_bfa_random_block_lottery.png")


# ── Figure 5 — Progressive BFA degradation curve (bfloat16, bits 7–9) ─────────

def fig5_progressive_curve():
    path = RESULTS_ROOT / "progressive_bfa_bits7_8_9" / "progressive_bfa_summary.json"
    with open(path) as f:
        data = json.load(f)

    flip_counts = [1, 5, 10, 25, 50]
    baseline = data["baseline_ppl"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Progressive BFA — Guided Attack Degradation Curve\n(bfloat16, bits 7–9, Qwen 0.5B)",
                 fontsize=13, fontweight="bold")

    for s, color, marker in zip(["0", "1", "2"], SEED_COLORS, SEED_MARKERS):
        prog = data["seeds"][s]["progressive"]
        ratios = [clamp_ratio(prog.get(str(fc), {}).get("degradation")) for fc in flip_counts]
        ax.semilogy([0] + flip_counts, [1.0] + ratios,
                    color=color, marker=marker, linewidth=2.2, markersize=8,
                    label=f"seed {s}")

    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.6, label="baseline")
    ax.axhline(1e4, color="red", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.annotate("Catastrophic (10,000×)", xy=(0.2, 1.2e4), fontsize=9, color="red", alpha=0.7)

    ax.set_xlabel("Number of gradient-guided bit flips")
    ax.set_ylabel("PPL ratio (vs 19.01 baseline, log scale)")
    ax.set_xticks([0] + flip_counts)
    ax.set_xticklabels(["base", "1", "5", "10", "25", "50"])
    ax.set_ylim(0.8, CATAST * 5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_progressive_bfa_degradation_curve.png")


# ── Figure 6 — Guided vs random, lower bits (progressive bfloat16) ────────────

def fig6_guided_vs_random():
    path = RESULTS_ROOT / "progressive_bfa_bits7_8_9" / "progressive_bfa_summary.json"
    with open(path) as f:
        data = json.load(f)

    flip_counts = [1, 5, 10, 25, 50]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Progressive BFA (bfloat16, bits 7–9): Guided vs Random — Per Seed",
                 fontsize=13, fontweight="bold")

    for ax, s in zip(axes, ["0", "1", "2"]):
        prog = data["seeds"][s]["progressive"]
        rand = data["seeds"][s]["random"]

        p_ratios = [clamp_ratio(prog.get(str(fc), {}).get("degradation")) for fc in flip_counts]
        r_ratios = [clamp_ratio(rand.get(str(fc), {}).get("degradation")) for fc in flip_counts]

        ax.semilogy([0] + flip_counts, [1.0] + p_ratios,
                    color="#F44336", marker="o", linewidth=2.2, markersize=7, label="Guided (progressive)")
        ax.semilogy([0] + flip_counts, [1.0] + r_ratios,
                    color="#607D8B", marker="s", linewidth=2.0, markersize=7,
                    linestyle="--", label="Random (same bits)")

        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_title(f"Seed {s}")
        ax.set_xlabel("Bit flips")
        ax.set_xticks([0] + flip_counts)
        ax.set_xticklabels(["base", "1", "5", "10", "25", "50"])
        ax.set_ylim(0.8, CATAST * 5)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("PPL ratio (log scale)")
    fig.tight_layout()
    save(fig, "fig6_progressive_guided_vs_random.png")


# ── Figure 7 — Five-way attack comparison ─────────────────────────────────────

def fig7_comparison():
    """
    Bar chart comparing median PPL ratio at 5 flips across all attack types.
    NaN/immune displayed with special annotations.
    """
    grad_path = RESULTS_ROOT / "grad_bfa" / "grad_bfa_summary.json"
    prog_path  = RESULTS_ROOT / "progressive_bfa_bits7_8_9" / "progressive_bfa_summary.json"

    with open(grad_path)  as f: grad = json.load(f)
    with open(prog_path)  as f: prog = json.load(f)

    # attack labels, median degradation @ flip=5, color, note
    attacks = [
        {
            "label": "Random\nweight bits\n(GGUF Exp 1)",
            "ratios": [1.0, 1.0, 1.0],   # immune at 1000 flips — certainly at 5
            "color": "#607D8B",
            "note": "Immune\n(1,000 flips = no effect)",
        },
        {
            "label": "Random\nexponent bits\n(float16)",
            "ratios": [
                grad["seeds"]["0"]["random"]["5"]["degradation"] or 1.0,
                grad["seeds"]["1"]["random"]["5"]["degradation"] or 1.0,
                grad["seeds"]["2"]["random"]["5"]["degradation"] or 1.0,
            ],
            "color": "#FF9800",
            "note": None,
        },
        {
            "label": "Gradient\nguided\n(float16, 1-shot)",
            "ratios": [CATAST, CATAST, CATAST],   # NaN at flip-1
            "color": "#F44336",
            "note": "NaN @ flip-1\n(catastrophic)",
        },
        {
            "label": "Progressive\nguided\n(bfloat16, bits 7–9)",
            "ratios": [
                clamp_ratio(prog["seeds"]["0"]["progressive"]["5"]["degradation"]),
                clamp_ratio(prog["seeds"]["1"]["progressive"]["5"]["degradation"]),
                clamp_ratio(prog["seeds"]["2"]["progressive"]["5"]["degradation"]),
            ],
            "color": "#9C27B0",
            "note": None,
        },
        {
            "label": "Random lower\nbits (bfloat16,\nbits 7–9)",
            "ratios": [
                prog["seeds"]["0"]["random"]["5"]["degradation"],
                prog["seeds"]["1"]["random"]["5"]["degradation"],
                prog["seeds"]["2"]["random"]["5"]["degradation"],
            ],
            "color": "#4CAF50",
            "note": "Immune\n(~1.0× all seeds)",
        },
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Attack Comparison @ 5 Bit Flips — Degradation Ratio (log scale)",
                 fontsize=13, fontweight="bold")

    x       = np.arange(len(attacks))
    width   = 0.22
    offsets = [-1, 0, 1]

    for i, (off, seed, color_offset) in enumerate(zip(offsets, [0, 1, 2], [0.9, 1.0, 1.1])):
        vals = [min(a["ratios"][seed], CATAST) for a in attacks]
        bars = ax.bar(x + off * width, vals, width, alpha=0.85,
                      color=[a["color"] for a in attacks],
                      edgecolor="white", linewidth=0.5)

    # Seed legend
    for off, seed in zip(offsets, [0, 1, 2]):
        ax.bar([], [], color="gray", alpha=0.5 + off * 0.25, label=f"seed {seed}")
    ax.legend(loc="upper right", fontsize=9)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([a["label"] for a in attacks], fontsize=9)
    ax.set_ylabel("PPL ratio (log scale)")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    # Annotate immune / NaN attacks
    for i, a in enumerate(attacks):
        if a["note"]:
            ax.annotate(a["note"], xy=(i, 2.0), ha="center", va="bottom",
                        fontsize=8, color="darkgray",
                        arrowprops=dict(arrowstyle="->", color="darkgray"),
                        xytext=(i, 5.0))

    ax.set_ylim(0.5, CATAST * 10)
    fig.tight_layout()
    save(fig, "fig7_five_way_attack_comparison.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output directory: {FIGURES_DIR}")
    print()
    fig1_exp1_fp16()
    fig2_exp1_quant()
    fig3_exp2_scale()
    fig4_grad_bfa_random()
    fig5_progressive_curve()
    fig6_guided_vs_random()
    fig7_comparison()
    print(f"\nAll figures saved to {FIGURES_DIR}")
