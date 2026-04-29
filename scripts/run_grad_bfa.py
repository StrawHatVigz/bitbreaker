#!/usr/bin/env python3
"""
run_grad_bfa.py — Gradient-guided Bit-Flip Attack experiment for BitBreaker.

Implements the progressive BFA algorithm (Rakin et al., ICCV 2019) adapted
for float16 transformer LLMs loaded via HuggingFace Transformers.

Model   : Qwen 2.5-0.5B-Instruct (FP16, PyTorch / HuggingFace)
Attack  : gradient-guided bit selection targeting FP16 exponent bits (10–14)
          — same bit zone as high_impact=True in the existing GGUF experiments
Flip counts : [1, 5, 10, 25, 50]
Seeds       : [0, 1, 2]   (seed controls calibration batch offset)
Metrics     : PPL on wikitext2 test set, ctx=512 (matching GGUF baseline method)

This experiment adds a "random" comparison condition at the same flip counts
and bit zone so that guided vs. unguided performance can be compared directly
within a single script run.

Why this complements the GGUF experiments:
  Exp1 (GGUF) showed random weight-bit flips leave quantised models unharmed.
  This script asks: does an optimal white-box attacker targeting FP16 weight bits
  outperform random selection?  If yes, by how much — and is it still weaker
  than targeting scale bytes (Exp2)?

Output structure:
    experiments/results_grendel/fault_injection/grad_bfa/
        seed_0/
            guided_flip_1_results.json
            guided_flip_5_results.json
            ...
            random_flip_1_results.json
            ...
        seed_1/ ...
        seed_2/ ...
        grad_bfa_summary.json

Usage:
    # Full run (guided + random, all seeds + flip counts)
    /tmp/rrgundam/bfa_env/bin/python scripts/run_grad_bfa.py

    # Guided only, quick test
    /tmp/rrgundam/bfa_env/bin/python scripts/run_grad_bfa.py --mode guided --flip-counts 1 5

    # Dry run: show what would execute without touching the model
    /tmp/rrgundam/bfa_env/bin/python scripts/run_grad_bfa.py --dry-run

    # Force re-run even if result files already exist
    /tmp/rrgundam/bfa_env/bin/python scripts/run_grad_bfa.py --force
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()   # suppress tokenizer sequence-length warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "fault_injection"))

from gradient_bfa import (
    EXPONENT_BITS,
    apply_flips,
    compute_bit_scores,
    restore_flips,
)

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH  = PROJECT_ROOT / "models" / "hf" / "Qwen2.5-0.5B-Instruct"
WIKITEXT    = PROJECT_ROOT / "configs" / "wikitext2_test.txt"
RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "results_grendel"
    / "fault_injection" / "grad_bfa"
)

FLIP_COUNTS  = [1, 5, 10, 25, 50]
SEEDS        = [0, 1, 2]
CALIB_TOKENS = 512    # tokens used to compute gradients (calibration batch)
EVAL_CTX     = 512    # context window for PPL  — matches GGUF baseline
DEVICE       = "cuda:0"


# ── PPL evaluation ─────────────────────────────────────────────────────────────

def compute_ppl(
    model:     torch.nn.Module,
    tokenizer,
    text:      str,
    ctx:       int = EVAL_CTX,
) -> float:
    """
    Non-overlapping sliding-window perplexity on plain text.
    Matches the llama-perplexity -c 512 methodology used in the GGUF experiments.
    """
    model.eval()
    enc      = tokenizer(text, return_tensors="pt").input_ids[0]   # [T]
    total_nll    = 0.0
    total_tokens = 0

    for start in range(0, len(enc) - 1, ctx):
        end   = min(start + ctx + 1, len(enc))
        chunk = enc[start:end]
        if len(chunk) < 2:
            break
        input_ids = chunk[:-1].unsqueeze(0).to(DEVICE)  # [1, L]
        labels    = chunk[1:].unsqueeze(0).to(DEVICE)   # [1, L]

        with torch.no_grad():
            logits = model(input_ids).logits              # [1, L, vocab]
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )
        total_nll    += nll.item()
        total_tokens += labels.numel()

    return round(math.exp(total_nll / total_tokens), 4)


# ── Calibration batch ──────────────────────────────────────────────────────────

def get_calib_batch(tokenizer, text: str, seed: int, n_tokens: int) -> torch.Tensor:
    """
    Sample a contiguous n_tokens chunk from text at a seeded random offset.
    Returned as [1, n_tokens] on DEVICE.
    """
    enc       = tokenizer(text, return_tensors="pt").input_ids[0]
    rng       = random.Random(seed)
    max_start = len(enc) - n_tokens - 1
    start     = rng.randint(0, max_start)
    return enc[start : start + n_tokens].unsqueeze(0).to(DEVICE)


# ── Random-flip baseline ────────────────────────────────────────────────────────

def random_flip_candidates(
    model:      torch.nn.Module,
    n_flips:    int,
    seed:       int,
    bits:       List[int] = EXPONENT_BITS,
) -> List[Tuple[str, int, int, float]]:
    """
    Select `n_flips` (layer, flat_index, bit) triples uniformly at random
    from eligible weight matrices and the given bit zone.
    Returns them in the same (name, flat_idx, bit, 0.0) format as compute_bit_scores.
    """
    rng      = random.Random(seed + 9999)   # different seed space from calib batch
    eligible = [
        (name, param)
        for name, param in model.named_parameters()
        if param.dim() == 2
        and not any(kw in name.lower() for kw in ("embed", "norm", "bias", "lm_head", "head"))
    ]

    flips: List[Tuple[str, int, int, float]] = []
    while len(flips) < n_flips:
        name, param = rng.choice(eligible)
        flat_idx    = rng.randrange(param.numel())
        bit         = rng.choice(bits)
        flips.append((name, flat_idx, bit, 0.0))

    return flips


# ── Result builder ─────────────────────────────────────────────────────────────

def build_result(
    mode:         str,
    n_flips:      int,
    seed:         int,
    ppl:          float,
    baseline_ppl: float,
    flip_list:    List[Tuple[str, int, int, float]],
    elapsed_s:    float,
) -> dict:
    return {
        "model":        str(MODEL_PATH.name),
        "quant_label":  "FP16",
        "platform":     "grendel",
        "mode":         mode,              # "guided" or "random"
        "flip_count":   n_flips,
        "seed":         seed,
        "calib_tokens": CALIB_TOKENS,
        "bits_targeted": EXPONENT_BITS,
        "baseline_ppl": baseline_ppl,
        "ppl":          ppl,
        "degradation":  round(ppl / baseline_ppl, 4),
        "flips": [
            {
                "layer":               t[0],
                "flat_idx":            t[1],
                "bit":                 t[2],
                "delta_loss_approx":   t[3],
            }
            for t in flip_list
        ],
        "eval_time_s": round(elapsed_s, 2),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer …  {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading model (float16) on {DEVICE} …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    vram_gb  = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"  {n_params:.0f}M params  |  {vram_gb:.2f} GB VRAM")

    wikitext = WIKITEXT.read_text(encoding="utf-8")

    # ── Baseline PPL ──────────────────────────────────────────────────────────
    print("\nComputing baseline PPL …")
    t0           = time.time()
    baseline_ppl = compute_ppl(model, tokenizer, wikitext)
    print(f"  Baseline PPL = {baseline_ppl:.4f}   ({time.time()-t0:.1f}s)")

    summary = {
        "experiment":   "grad_bfa",
        "model":        MODEL_PATH.name,
        "baseline_ppl": baseline_ppl,
        "seeds":        {},
    }

    # ── Per-seed loop ─────────────────────────────────────────────────────────
    for seed in args.seeds:
        print(f"\n{'='*64}")
        print(f"SEED {seed}")
        print(f"{'='*64}")

        seed_dir = RESULTS_DIR / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        seed_summary: Dict = {"baseline_ppl": baseline_ppl, "guided": {}, "random": {}}
        summary["seeds"][str(seed)] = seed_summary

        # ── Gradient scores (computed once per seed) ──────────────────────────
        guided_scores = None
        if "guided" in args.mode:
            calib_ids = get_calib_batch(tokenizer, wikitext, seed, CALIB_TOKENS)
            print(f"  Computing gradient scores  (calib={CALIB_TOKENS} tok, bits={EXPONENT_BITS}) …")
            t0            = time.time()
            guided_scores = compute_bit_scores(model, calib_ids, bits=EXPONENT_BITS)
            elapsed_score = time.time() - t0
            print(f"  {len(guided_scores):,} candidate flips found  ({elapsed_score:.1f}s)")
            print("  Top-5 candidates:")
            for nm, idx, bit, dl in guided_scores[:5]:
                layer_short = ".".join(nm.split(".")[-3:])
                print(f"    {layer_short}[{idx}]  bit{bit}  Δloss≈{dl:.5f}")

        # ── Per-flip-count loop ───────────────────────────────────────────────
        for n_flips in args.flip_counts:

            # ── Guided ───────────────────────────────────────────────────────
            if "guided" in args.mode:
                out_path = seed_dir / f"guided_flip_{n_flips}_results.json"

                if out_path.exists() and not args.force:
                    print(f"  [guided] flip={n_flips:>3}  already done — skipping")
                elif args.dry_run:
                    print(f"  [dry-run guided] flip={n_flips:>3}")
                else:
                    top_n     = guided_scores[:n_flips]
                    originals = apply_flips(model, top_n)
                    t0        = time.time()
                    ppl       = compute_ppl(model, tokenizer, wikitext)
                    elapsed   = time.time() - t0
                    restore_flips(model, originals)

                    result = build_result("guided", n_flips, seed, ppl,
                                          baseline_ppl, top_n, elapsed)
                    out_path.write_text(json.dumps(result, indent=2))
                    seed_summary["guided"][str(n_flips)] = {
                        "ppl": ppl, "degradation": result["degradation"]
                    }
                    print(
                        f"  [guided] flip={n_flips:>3}  PPL={ppl:>14.4f}"
                        f"  ({result['degradation']:.2f}x baseline)  [{elapsed:.1f}s]"
                    )

            # ── Random ───────────────────────────────────────────────────────
            if "random" in args.mode:
                out_path = seed_dir / f"random_flip_{n_flips}_results.json"

                if out_path.exists() and not args.force:
                    print(f"  [random] flip={n_flips:>3}  already done — skipping")
                elif args.dry_run:
                    print(f"  [dry-run random] flip={n_flips:>3}")
                else:
                    rand_flips = random_flip_candidates(model, n_flips, seed)
                    originals  = apply_flips(model, rand_flips)
                    t0         = time.time()
                    ppl        = compute_ppl(model, tokenizer, wikitext)
                    elapsed    = time.time() - t0
                    restore_flips(model, originals)

                    result = build_result("random", n_flips, seed, ppl,
                                          baseline_ppl, rand_flips, elapsed)
                    out_path.write_text(json.dumps(result, indent=2))
                    seed_summary["random"][str(n_flips)] = {
                        "ppl": ppl, "degradation": result["degradation"]
                    }
                    print(
                        f"  [random] flip={n_flips:>3}  PPL={ppl:>14.4f}"
                        f"  ({result['degradation']:.2f}x baseline)  [{elapsed:.1f}s]"
                    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / "grad_bfa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {summary_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gradient-guided BFA on Qwen 0.5B FP16")
    p.add_argument(
        "--mode",
        nargs="+",
        choices=["guided", "random"],
        default=["guided", "random"],
        help="Which attack mode(s) to run  (default: both)",
    )
    p.add_argument(
        "--flip-counts",
        nargs="+",
        type=int,
        default=FLIP_COUNTS,
        metavar="N",
        help="Flip counts to test  (default: 1 5 10 25 50)",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to run  (default: 0 1 2)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if result files already exist",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
