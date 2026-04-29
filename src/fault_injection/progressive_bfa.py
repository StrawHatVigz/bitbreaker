#!/usr/bin/env python3
"""
progressive_bfa.py — Proper Rakin et al. progressive BFA for float16/bfloat16 LLMs.

Unlike gradient_bfa.py (one-shot: score once, apply N flips), this implements
the progressive loop from the original paper:

    for step in 1 .. N:
        1. FRESH forward + backward pass on current (already-flipped) model
        2. Score every (weight, bit) triple with Taylor approx
        3. Pick the globally best candidate
        4. Apply that one flip to the model  (permanently for this attack run)

The key difference: gradients are re-evaluated on the already-corrupted model
at every step, so later flips compound the damage of earlier ones.

Why bfloat16 instead of float16:
    bfloat16 has the same 8-bit exponent as float32, giving a max representable
    value of ~3.4×10^38.  A single exponent-bit flip amplifies a weight by up to
    ~65536× but the result stays finite (bfloat16 max is not exceeded for typical
    weight magnitudes).  This lets us observe a gradual degradation curve rather
    than an immediate NaN at flip-1 — which is the correct behaviour for studying
    progressive attacks.

Bit layout:
    float16  — bit 15 sign | bits 14-10 exponent (5-bit) | bits 9-0 mantissa
    bfloat16 — bit 15 sign | bits 14-7  exponent (8-bit) | bits 6-0 mantissa

    Both are 16-bit, so the same int16 bit-manipulation trick applies.
    Targeted zone (bits 10-14) covers the TOP 5 exponent bits in both dtypes —
    same zone as high_impact=True in the GGUF FaultInjector.
"""

import torch
from typing import Dict, List, Optional, Tuple

# ── Bit-range constants ────────────────────────────────────────────────────────

EXPONENT_BITS_FP16 = list(range(10, 15))   # 5-bit exponent, full range
EXPONENT_BITS_BF16 = list(range(10, 15))   # top 5 of bfloat16's 8-bit exponent
ALL_BITS_FP16      = list(range(16))
ALL_BITS_BF16      = list(range(16))


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_weight_matrix(name: str, param: torch.Tensor) -> bool:
    """True for 2-D weight matrices in attention / FFN layers."""
    if param.dim() != 2:
        return False
    skip = ("embed", "norm", "bias", "lm_head", "head")
    return not any(kw in name.lower() for kw in skip)


def _delta_w_for_bit(
    w_flat: torch.Tensor,
    bit: int,
) -> torch.Tensor:
    """
    Return (w_flipped − w) in float32 for every element of w_flat.

    Works for both float16 and bfloat16 — detects dtype from w_flat.
    Elements whose flipped value is non-finite get delta = 0.
    """
    dtype  = w_flat.dtype                                          # fp16 or bf16
    mask   = torch.tensor(1 << bit, dtype=torch.int16, device=w_flat.device)
    w_int  = w_flat.view(torch.int16)
    f_int  = (w_int ^ mask)
    w_flip = f_int.view(dtype)                                     # back to fp dtype
    delta  = w_flip.float() - w_flat.float()
    delta[~torch.isfinite(w_flip)] = 0.0
    return delta                                                   # float32 [N]


def _apply_single_flip(
    params:   Dict[str, torch.nn.Parameter],
    name:     str,
    flat_idx: int,
    bit:      int,
) -> torch.Tensor:
    """
    Flip bit `bit` of weight at flat index `flat_idx` in layer `name`.
    Returns the ORIGINAL value (for restore).
    """
    param     = params[name]
    view      = param.data.view(-1)
    original  = view[flat_idx].clone()
    mask      = torch.tensor(1 << bit, dtype=torch.int16, device=param.device)
    w_int     = view[flat_idx].view(torch.int16)
    flipped   = (w_int ^ mask).view(param.dtype)
    view[flat_idx] = flipped
    return original


# ── Public API ─────────────────────────────────────────────────────────────────

def find_best_flip(
    model:          torch.nn.Module,
    input_ids:      torch.Tensor,
    bits:           List[int],
    top_k:          int = 500,
    skip_positions: Optional[set] = None,
) -> Optional[Tuple[str, int, int, float]]:
    """
    Single forward+backward on the CURRENT model state  →  best (name, flat_idx, bit, dl).

    Used inside the progressive loop: called once per flip step so gradients
    reflect the already-corrupted model.  Returns None if no positive-delta
    candidate is found.

    skip_positions: set of (name, flat_idx) pairs already flipped — excluded
    from consideration to prevent the oscillation where a weight is flipped
    back and forth between two states (standard Rakin et al. no-re-flip rule).
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    with torch.enable_grad():
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()

    skip = skip_positions or set()
    best: Optional[Tuple[str, int, int, float]] = None

    for name, param in model.named_parameters():
        if not _is_weight_matrix(name, param):
            continue
        if param.grad is None:
            continue

        w_flat = param.data.view(-1)
        g_flat = param.grad.detach().view(-1).float()

        dl_cols = [g_flat * _delta_w_for_bit(w_flat, b) for b in bits]
        dl_mat  = torch.stack(dl_cols, dim=1)           # [N, B]

        best_dl, best_bit_local = dl_mat.max(dim=1)     # [N]
        pos_mask = best_dl > 0
        if not pos_mask.any():
            continue

        pos_dl  = best_dl[pos_mask]
        pos_idx = pos_mask.nonzero(as_tuple=True)[0]
        pos_bit = best_bit_local[pos_mask]

        k = min(top_k, pos_dl.numel())
        topk_dl, topk_local = pos_dl.topk(k)

        for i in range(k):
            dl  = topk_dl[i].item()
            idx = pos_idx[topk_local[i]].item()
            if (name, idx) in skip:
                continue
            bit = bits[pos_bit[topk_local[i]].item()]
            if best is None or dl > best[3]:
                best = (name, idx, bit, dl)

    model.zero_grad()
    return best


def run_progressive_attack(
    model:       torch.nn.Module,
    calib_ids:   torch.Tensor,
    max_flips:   int,
    bits:        List[int],
    top_k:       int = 500,
) -> Tuple[List[Tuple[int, str, int, int, float]], Dict[Tuple[str, int], torch.Tensor]]:
    """
    Run the progressive BFA loop for up to `max_flips` steps.

    At each step:
      1. Recompute gradients on the currently-corrupted model
      2. Pick the globally best (layer, weight, bit) triple
      3. Apply it permanently (model is modified in-place)

    Returns:
        flip_history : list of (step, name, flat_idx, bit, delta_loss)
                       one entry per flip applied
        originals    : {(name, flat_idx): original_fp_value} for restore_progressive()

    The model is left in the corrupted state.
    Call restore_progressive(model, originals) to undo all flips.
    """
    params         = dict(model.named_parameters())
    originals:     Dict[Tuple[str, int], torch.Tensor] = {}
    flip_history:  List[Tuple[int, str, int, int, float]] = []
    flipped_set:   set = set()   # (name, flat_idx) already flipped — no re-flip rule

    for step in range(1, max_flips + 1):
        candidate = find_best_flip(
            model, calib_ids, bits, top_k=top_k, skip_positions=flipped_set
        )

        if candidate is None:
            print(f"  [progressive] step {step}: no unflipped positive-delta candidate, stopping early")
            break

        name, flat_idx, bit, dl = candidate
        key = (name, flat_idx)
        flipped_set.add(key)

        if key not in originals:
            originals[key] = params[name].data.view(-1)[flat_idx].clone()

        _apply_single_flip(params, name, flat_idx, bit)
        flip_history.append((step, name, flat_idx, bit, dl))

        layer_short = ".".join(name.split(".")[-3:])
        print(f"  step {step:>3}  {layer_short}[{flat_idx}]  bit{bit}  Δloss≈{dl:.5f}")

    return flip_history, originals


def restore_progressive(
    model:     torch.nn.Module,
    originals: Dict[Tuple[str, int], torch.Tensor],
) -> None:
    """Undo all flips applied by run_progressive_attack."""
    params = dict(model.named_parameters())
    for (name, flat_idx), orig_val in originals.items():
        params[name].data.view(-1)[flat_idx] = orig_val
