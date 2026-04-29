#!/usr/bin/env python3
"""
gradient_bfa.py — Gradient-guided Bit-Flip Attack for float16 transformer models.

Algorithm (Rakin et al., ICCV 2019 — adapted for LLMs):
  1. Forward pass on a calibration batch  →  cross-entropy loss
  2. Backward pass  →  ∂Loss/∂w_ij for every weight
  3. For each weight w_ij with gradient g_ij, for each candidate bit k:
         delta_loss ≈ g_ij * (flip_bit(w_ij, k) − w_ij)   [first-order Taylor]
  4. Rank all (layer, flat_index, bit) triples by delta_loss descending
  5. Return top-N for the caller to apply / restore

Targets:  attention + FFN 2-D weight matrices only.
Skips:    embed_tokens, lm_head, layernorm, biases  (1-D or explicitly excluded).

FP16 bit layout:
    bit 15        → sign
    bits 14–10    → exponent  (5 bits) — highest magnitude impact per flip
    bits 9–0      → mantissa (10 bits)
"""

import torch
from typing import Dict, List, Tuple

# ── Bit-range constants ────────────────────────────────────────────────────────

ALL_BITS      = list(range(16))
EXPONENT_BITS = list(range(10, 15))   # bits 10–14  (same zone as high_impact=True in FaultInjector)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_weight_matrix(name: str, param: torch.Tensor) -> bool:
    """True for 2-D weight matrices in attention / FFN layers."""
    if param.dim() != 2:
        return False
    skip = ("embed", "norm", "bias", "lm_head", "head")
    return not any(kw in name.lower() for kw in skip)


def _delta_w_for_bit(w_flat: torch.Tensor, bit: int) -> torch.Tensor:
    """
    Return (w_flipped − w) in float32 for every element of w_flat (float16).
    Elements whose flipped value is NaN or Inf get delta = 0 (no effect).
    """
    mask      = torch.tensor(1 << bit, dtype=torch.int16, device=w_flat.device)
    w_int     = w_flat.view(torch.int16)
    f_int     = (w_int ^ mask)
    w_flipped = f_int.view(torch.float16).float()
    delta     = w_flipped - w_flat.float()
    delta[~torch.isfinite(w_flipped)] = 0.0
    return delta   # float32 [N]


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_bit_scores(
    model:           torch.nn.Module,
    input_ids:       torch.Tensor,
    bits:            List[int] = EXPONENT_BITS,
    top_k_per_layer: int       = 500,
) -> List[Tuple[str, int, int, float]]:
    """
    Single forward+backward pass  →  ranked list of bit-flip candidates.

    For each eligible weight tensor the function:
      a) computes delta_loss = g * delta_w  for every (weight, bit) pair
      b) picks the best bit per weight  (the one that maximises delta_loss)
      c) keeps the top `top_k_per_layer` positive-delta entries from that tensor

    Returns a list sorted by delta_loss descending:
        [(layer_name, flat_index, bit, delta_loss), ...]

    Only flips that *increase* loss (delta_loss > 0) are included.

    Args:
        model           : float16 model on GPU, parameters must allow grad
        input_ids       : [1, seq_len] token ids on the same device as model
        bits            : bit positions to evaluate (default: exponent bits 10-14)
        top_k_per_layer : max candidates to keep per weight tensor
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    with torch.enable_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()

    candidates: List[Tuple[str, int, int, float]] = []

    for name, param in model.named_parameters():
        if not _is_weight_matrix(name, param):
            continue
        if param.grad is None:
            continue

        w_flat = param.data.view(-1)                    # float16 [N]
        g_flat = param.grad.detach().view(-1).float()   # float32 [N]

        # Build [N, B] delta_loss matrix (one column per bit position)
        dl_cols = [g_flat * _delta_w_for_bit(w_flat, b) for b in bits]
        dl_mat  = torch.stack(dl_cols, dim=1)           # float32 [N, B]

        # Per-weight: keep only the best bit
        best_dl, best_bit_local = dl_mat.max(dim=1)    # [N]

        pos_mask = best_dl > 0
        if not pos_mask.any():
            continue

        pos_dl        = best_dl[pos_mask]
        pos_idx       = pos_mask.nonzero(as_tuple=True)[0]
        pos_bit_local = best_bit_local[pos_mask]

        k          = min(top_k_per_layer, pos_dl.numel())
        topk_dl, topk_local = pos_dl.topk(k)

        for i in range(k):
            flat_idx = pos_idx[topk_local[i]].item()
            bit      = bits[pos_bit_local[topk_local[i]].item()]
            dl       = topk_dl[i].item()
            candidates.append((name, flat_idx, bit, dl))

    model.zero_grad()

    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates


def apply_flips(
    model:     torch.nn.Module,
    flip_list: List[Tuple[str, int, int, float]],
) -> Dict[Tuple[str, int], torch.Tensor]:
    """
    Apply bit flips to model weight parameters in-place.

    Returns originals dict  {(layer_name, flat_index): original_fp16_value}
    so flips can be fully reversed by restore_flips().

    If the same (layer, flat_index) appears more than once in flip_list the
    first occurrence wins for the saved original value — restoring will always
    return to the pre-flip state.
    """
    originals: Dict[Tuple[str, int], torch.Tensor] = {}
    params = dict(model.named_parameters())

    for name, flat_idx, bit, _ in flip_list:
        key   = (name, flat_idx)
        param = params[name]
        view  = param.data.view(-1)

        if key not in originals:
            originals[key] = view[flat_idx].clone()

        mask    = torch.tensor(1 << bit, dtype=torch.int16, device=param.device)
        w_int   = view[flat_idx].view(torch.int16)
        flipped = (w_int ^ mask).view(torch.float16)
        view[flat_idx] = flipped

    return originals


def restore_flips(
    model:     torch.nn.Module,
    originals: Dict[Tuple[str, int], torch.Tensor],
) -> None:
    """Restore model to its exact pre-flip state using originals from apply_flips()."""
    params = dict(model.named_parameters())
    for (name, flat_idx), orig_val in originals.items():
        params[name].data.view(-1)[flat_idx] = orig_val
