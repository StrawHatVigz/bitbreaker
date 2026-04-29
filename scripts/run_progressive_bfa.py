#!/usr/bin/env python3
"""
run_progressive_bfa.py — Proper progressive BFA experiment for BitBreaker.

Implements the Rakin et al. ICCV 2019 progressive BFA with bfloat16 precision,
contrasted with the one-shot gradient attack in run_grad_bfa.py.

KEY DIFFERENCES from run_grad_bfa.py:
  - Model loaded in BFLOAT16 (not float16):
      bfloat16 has 8 exponent bits (same range as float32, max ~3.4×10^38).
      A flipped weight may amplify by ~65536× but stays finite — the model
      degrades gradually instead of NaN-ing at flip-1.  This gives a proper
      degradation curve for analysis.

  - PROGRESSIVE gradient recomputation (not one-shot):
      run_grad_bfa.py computes gradients once then applies top-N simultaneously.
      This script recomputes gradients after EVERY flip (the actual Rakin paper
      algorithm).  This means flip-5 gradients reflect a model already corrupted
      by flips 1–4 — later flips compound earlier damage.

EXPERIMENTAL DESIGN:
  Model      : Qwen 2.5-0.5B-Instruct, loaded as bfloat16
  Attack     : progressive guided (1 gradient pass per flip step)
  Comparison : random exponent-bit flips at same flip counts (applied all at once)
  Flip counts: checkpoint measurements at [1, 5, 10, 25, 50] steps
               (efficient: run 50 progressive steps, measure PPL at each checkpoint)
  Seeds      : [0, 1, 2]
  Bits       : 10–14  (top 5 exponent bits, same zone as GGUF high_impact=True)

MODEL SCOPE NOTE:
  This experiment targets FP16/BF16 PyTorch models (gradient BFA requires
  autograd).  Running it on GGUF quantized models would require a GGUF-aware
  dequantization layer in PyTorch (not available in llama.cpp).  For now:
    ✓ Qwen 2.5-0.5B-Instruct (FP16/BF16, already downloaded)
    ○ Llama 3.2-1B-Instruct (FP16/BF16) — add --model llama if HF weights downloaded
  The meaningful comparison to Exp 1–3b is the cross-attack table in the paper
  (random weight, random exponent, guided, scale-byte), not running progressive
  BFA on all 8 model×quant combinations.

OUTPUT STRUCTURE:
    experiments/results_grendel/fault_injection/progressive_bfa/
        seed_0/
            progressive_results.json   — degradation at each checkpoint step
            random_flip_1_results.json
            random_flip_5_results.json
            ...
        seed_1/ ...
        seed_2/ ...
        progressive_bfa_summary.json

USAGE:
    # Full run (all seeds, progressive + random)
    /tmp/rrgundam/bfa_env/bin/python scripts/run_progressive_bfa.py

    # Quick test — one seed, first two checkpoints only
    /tmp/rrgundam/bfa_env/bin/python scripts/run_progressive_bfa.py \\
        --seeds 0 --checkpoints 1 5

    # Dry run — print plan without running
    /tmp/rrgundam/bfa_env/bin/python scripts/run_progressive_bfa.py --dry-run

    # Force re-run existing results
    /tmp/rrgundam/bfa_env/bin/python scripts/run_progressive_bfa.py --force

RUNTIME ESTIMATE (Grendel L4):
    Per seed: 50 gradient passes (~2s each) + 5 PPL evaluations (~30s each)
             ≈ 100 + 150 = ~4 min/seed  →  ~12 min total (3 seeds)
    Random baseline: negligible (no gradient passes)
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "fault_injection"))

from progressive_bfa import (
    EXPONENT_BITS_BF16,
    restore_progressive,
    run_progressive_attack,
)

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH   = PROJECT_ROOT / "models" / "hf" / "Qwen2.5-0.5B-Instruct"
WIKITEXT     = PROJECT_ROOT / "configs" / "wikitext2_test.txt"
RESULTS_DIR  = (
    PROJECT_ROOT / "experiments" / "results_grendel"
    / "fault_injection" / "progressive_bfa"
)

CHECKPOINTS  = [1, 5, 10, 25, 50]    # PPL is measured at these flip counts
MAX_FLIPS    = max(CHECKPOINTS)
SEEDS        = [0, 1, 2]
CALIB_TOKENS = 512
EVAL_CTX     = 512
DEVICE       = "cuda:0"
BITS         = EXPONENT_BITS_BF16    # bits 10-14  (default; overridden by --bits)


# ── PPL evaluation ─────────────────────────────────────────────────────────────

def compute_ppl(model, tokenizer, text: str, ctx: int = EVAL_CTX) -> float:
    model.eval()
    enc          = tokenizer(text, return_tensors="pt").input_ids[0]
    total_nll    = 0.0
    total_tokens = 0

    for start in range(0, len(enc) - 1, ctx):
        end   = min(start + ctx + 1, len(enc))
        chunk = enc[start:end]
        if len(chunk) < 2:
            break
        input_ids = chunk[:-1].unsqueeze(0).to(DEVICE)
        labels    = chunk[1:].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids).logits
            nll    = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )
        total_nll    += nll.item()
        total_tokens += labels.numel()

    if total_tokens == 0:
        return float("inf")
    return round(math.exp(total_nll / total_tokens), 4)


def get_calib_batch(tokenizer, text: str, seed: int, n_tokens: int) -> torch.Tensor:
    enc       = tokenizer(text, return_tensors="pt").input_ids[0]
    rng       = random.Random(seed)
    max_start = len(enc) - n_tokens - 1
    start     = rng.randint(0, max_start)
    return enc[start : start + n_tokens].unsqueeze(0).to(DEVICE)


def random_flip_candidates(model, n_flips: int, seed: int, bits: list) -> list:
    """Uniform random (layer, flat_idx, bit) from eligible weight matrices."""
    rng      = random.Random(seed + 9999)
    eligible = [
        (name, param)
        for name, param in model.named_parameters()
        if param.dim() == 2
        and not any(kw in name.lower() for kw in ("embed", "norm", "bias", "lm_head", "head"))
    ]
    flips = []
    while len(flips) < n_flips:
        name, param = rng.choice(eligible)
        flat_idx    = rng.randrange(param.numel())
        bit         = rng.choice(bits)
        flips.append((name, flat_idx, bit, 0.0))
    return flips


def apply_random_flips(model, flip_list) -> dict:
    """Apply all random flips at once; return originals for restore."""
    originals = {}
    params    = dict(model.named_parameters())
    for name, flat_idx, bit, _ in flip_list:
        key  = (name, flat_idx)
        view = params[name].data.view(-1)
        if key not in originals:
            originals[key] = view[flat_idx].clone()
        mask    = torch.tensor(1 << bit, dtype=torch.int16, device=params[name].device)
        w_int   = view[flat_idx].view(torch.int16)
        flipped = (w_int ^ mask).view(params[name].dtype)
        view[flat_idx] = flipped
    return originals


def restore_random_flips(model, originals: dict) -> None:
    params = dict(model.named_parameters())
    for (name, flat_idx), orig in originals.items():
        params[name].data.view(-1)[flat_idx] = orig


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    bits = args.bits if args.bits else BITS
    # Output dir encodes the bit zone so different runs don't clobber each other
    bit_tag    = f"bits{'_'.join(str(b) for b in bits)}"
    results_dir = (
        PROJECT_ROOT / "experiments" / "results_grendel"
        / "fault_injection" / f"progressive_bfa_{bit_tag}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(args.checkpoints)
    max_flips   = max(checkpoints)

    print(f"Loading tokenizer …  {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading model (bfloat16) on {DEVICE} …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    vram_gb  = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"  {n_params:.0f}M params  |  {vram_gb:.2f} GB VRAM  |  dtype=bfloat16")

    wikitext = WIKITEXT.read_text(encoding="utf-8")

    print("\nComputing baseline PPL (bfloat16) …")
    t0           = time.time()
    baseline_ppl = compute_ppl(model, tokenizer, wikitext)
    print(f"  Baseline PPL = {baseline_ppl:.4f}   ({time.time()-t0:.1f}s)")

    print(f"  Bit zone: {bits}")
    summary: Dict = {
        "experiment":   "progressive_bfa",
        "model":        MODEL_PATH.name,
        "dtype":        "bfloat16",
        "bits":         bits,
        "checkpoints":  checkpoints,
        "baseline_ppl": baseline_ppl,
        "seeds":        {},
    }

    for seed in args.seeds:
        print(f"\n{'='*64}")
        print(f"SEED {seed}")
        print(f"{'='*64}")

        seed_dir = results_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        seed_summary: Dict = {
            "baseline_ppl": baseline_ppl,
            "progressive":  {},
            "random":       {},
        }
        summary["seeds"][str(seed)] = seed_summary

        # ── Progressive guided attack ─────────────────────────────────────────
        prog_out = seed_dir / "progressive_results.json"

        if prog_out.exists() and not args.force:
            print("  [progressive] already done — skipping")
            try:
                saved = json.loads(prog_out.read_text())
                seed_summary["progressive"] = saved.get("checkpoints", {})
            except Exception:
                pass
        elif args.dry_run:
            print(f"  [dry-run progressive] {max_flips} steps, checkpoints at {checkpoints}")
        else:
            calib_ids = get_calib_batch(tokenizer, wikitext, seed, CALIB_TOKENS)

            print(f"  Running progressive attack ({max_flips} gradient steps) …")
            t_start = time.time()

            flip_history, originals = run_progressive_attack(
                model, calib_ids, max_flips, bits
            )

            # flip_history is ordered: (step, name, flat_idx, bit, dl)
            # After all flips are applied we need to evaluate PPL at each checkpoint.
            # Strategy: flips are already applied cumulatively. We need to UNDO back
            # to each checkpoint level. Easier: re-run with incremental restore.
            # Actually the model now has max_flips flips applied — undo them all first,
            # then re-apply incrementally to each checkpoint.

            # Restore to clean state
            restore_progressive(model, originals)

            checkpoint_results = {}
            applied_so_far     = []

            for target in checkpoints:
                # Apply flips up to `target` steps
                steps_to_apply = flip_history[:target]
                for _, name, flat_idx, bit, _ in steps_to_apply[len(applied_so_far):]:
                    params = dict(model.named_parameters())
                    mask   = torch.tensor(1 << bit, dtype=torch.int16, device=params[name].device)
                    view   = params[name].data.view(-1)
                    w_int  = view[flat_idx].view(torch.int16)
                    view[flat_idx] = (w_int ^ mask).view(params[name].dtype)
                    applied_so_far.append((name, flat_idx, bit))

                t0  = time.time()
                ppl = compute_ppl(model, tokenizer, wikitext)
                elapsed = time.time() - t0
                deg = round(ppl / baseline_ppl, 4)
                checkpoint_results[str(target)] = {
                    "ppl": ppl, "degradation": deg
                }
                print(
                    f"  [progressive] flip={target:>3}  PPL={ppl:>16.4f}"
                    f"  ({deg:.4f}x baseline)  [{elapsed:.1f}s]"
                )

            # Restore model to clean state
            for name, flat_idx, bit in applied_so_far:
                # Flip again to restore (XOR is its own inverse)
                params = dict(model.named_parameters())
                mask   = torch.tensor(1 << bit, dtype=torch.int16, device=params[name].device)
                view   = params[name].data.view(-1)
                w_int  = view[flat_idx].view(torch.int16)
                view[flat_idx] = (w_int ^ mask).view(params[name].dtype)

            t_total = time.time() - t_start
            result  = {
                "model":        MODEL_PATH.name,
                "dtype":        "bfloat16",
                "platform":     "grendel",
                "mode":         "progressive_guided",
                "seed":         seed,
                "calib_tokens": CALIB_TOKENS,
                "bits":         bits,
                "baseline_ppl": baseline_ppl,
                "checkpoints":  checkpoint_results,
                "flip_history": [
                    {
                        "step":              s,
                        "layer":             n,
                        "flat_idx":          i,
                        "bit":               b,
                        "delta_loss_approx": d,
                    }
                    for s, n, i, b, d in flip_history
                ],
                "total_time_s": round(t_total, 2),
            }
            prog_out.write_text(json.dumps(result, indent=2))
            seed_summary["progressive"] = checkpoint_results

        # ── Random baseline ───────────────────────────────────────────────────
        for n_flips in checkpoints:
            out_path = seed_dir / f"random_flip_{n_flips}_results.json"

            if out_path.exists() and not args.force:
                print(f"  [random] flip={n_flips:>3}  already done — skipping")
                continue

            if args.dry_run:
                print(f"  [dry-run random] flip={n_flips:>3}")
                continue

            rand_flips = random_flip_candidates(model, n_flips, seed, bits)
            orig       = apply_random_flips(model, rand_flips)
            t0         = time.time()
            ppl        = compute_ppl(model, tokenizer, wikitext)
            elapsed    = time.time() - t0
            restore_random_flips(model, orig)

            deg    = round(ppl / baseline_ppl, 4)
            result = {
                "model":        MODEL_PATH.name,
                "dtype":        "bfloat16",
                "platform":     "grendel",
                "mode":         "random",
                "flip_count":   n_flips,
                "seed":         seed,
                "bits":         bits,
                "baseline_ppl": baseline_ppl,
                "ppl":          ppl,
                "degradation":  deg,
                "flips": [
                    {"layer": t[0], "flat_idx": t[1], "bit": t[2]}
                    for t in rand_flips
                ],
                "eval_time_s": round(elapsed, 2),
            }
            out_path.write_text(json.dumps(result, indent=2))
            seed_summary["random"][str(n_flips)] = {"ppl": ppl, "degradation": deg}
            print(
                f"  [random]      flip={n_flips:>3}  PPL={ppl:>16.4f}"
                f"  ({deg:.4f}x baseline)  [{elapsed:.1f}s]"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    summ_path = results_dir / "progressive_bfa_summary.json"
    summ_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {summ_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Progressive BFA (bfloat16) on Qwen 0.5B"
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=CHECKPOINTS,
        metavar="N",
        help="Flip counts at which to measure PPL  (default: 1 5 10 25 50)",
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
        help="Print what would run without executing",
    )
    p.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=None,
        metavar="B",
        help=(
            "Bit positions to target (default: 10 11 12 13 14 — high exponent).\n"
            "Use 7 8 9 for lower exponent bits (bfloat16 lower zone — avoids NaN,\n"
            "gives a gradual degradation curve instead of kill-switch at flip-1)."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if result files exist",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
