# Gradient-Guided Bit-Flip Attack — Complete Results

**Project:** BitBreaker (ECE 591, NCSU)
**Model:** Qwen 2.5-0.5B-Instruct
**Platform:** Grendel (NVIDIA L4 GPU, 22 GB VRAM)
**Date:** April 2026

---

## 1. Background and Motivation

The existing GGUF experiments (Exp 1–3b) established that:
- Random weight-bit flips leave quantized models completely immune at 1,000 flips.
- Scale-byte flips cause catastrophic failure in 1–10 flips.

The gradient-guided BFA experiments ask: **what does an optimal white-box attacker achieve
on the raw FP16 weight tensor?** This extends the experimental story beyond the GGUF format
into the PyTorch / HuggingFace model space, using the Rakin et al. ICCV 2019 algorithm.

---

## 2. Experiment A: One-Shot Gradient BFA (float16)

**Script:** `scripts/run_grad_bfa.py`
**Module:** `src/fault_injection/gradient_bfa.py`
**Model precision:** float16
**Algorithm:** One-shot — compute gradients once, rank all (weight, bit) candidates by
first-order Taylor approximation `Δloss ≈ g · Δw`, apply top-N simultaneously.
**Bit zone:** Exponent bits 10–14 (same zone as `high_impact=True` in GGUF FaultInjector)
**Comparison:** Random exponent-bit flips at the same flip counts and bit zone.
**Seeds:** 0, 1, 2 (controls calibration batch offset, 512 tokens from WikiText-2)
**Flip counts:** 1, 5, 10, 25, 50

### 2.1 Baseline

| Model | PPL (float16, PyTorch) |
|-------|------------------------|
| Qwen 2.5-0.5B-Instruct | **18.99** |

### 2.2 Guided Attack Results

All three seeds: **NaN at flip-1** across all flip counts.

| Seed | Flip=1 | Flip=5 | Flip=10 | Flip=25 | Flip=50 |
|------|--------|--------|---------|---------|---------|
| 0 | NaN | NaN | NaN | NaN | NaN |
| 1 | NaN | NaN | NaN | NaN | NaN |
| 2 | NaN | NaN | NaN | NaN | NaN |

**Mechanism:** The gradient identifies `mlp.down_proj.weight[302135] bit14` (seeds 0, 1) or
`mlp.gate_proj.weight[2702398] bit14` (seed 2) as the optimal flip. Flipping bit-14 (the MSB
of the float16 5-bit exponent) amplifies the weight by ~65,536× (e.g., 0.625 → 40,960).
This pushes FFN outputs past float16 max (65,504) → **Inf activations → NaN in LayerNorm /
softmax → propagates through all subsequent layers**. Model is non-functional at exactly 1 flip.

### 2.3 Random Exponent-Bit Baseline Results (same bit zone, no gradient guidance)

PPL ratios relative to 18.99 baseline:

| Seed | Flip=1 | Flip=5 | Flip=10 | Flip=25 | Flip=50 |
|------|--------|--------|---------|---------|---------|
| 0 | **1.00×** | **3.54×** | **24.9×** | **42,522×** | **1,327,045×** |
| 1 | 1.00× | 1.00× | 1.04× | **2,647×** | **702,594×** |
| 2 | 1.00× | 1.20× | 1.20× | **106×** | **49,626×** |

**Observation:** The block lottery is clearly visible. Seed 0 hits a critical block early
(catastrophic at flip=5). Seed 1 survives 5 flips with no effect, then degrades gradually
before catastrophic failure at flip=25. Seed 2 shows a mid-range curve. **Gradient guidance
eliminates this lottery**: the optimizer always finds the worst bit immediately.

### 2.4 Top Gradient Candidates (per seed)

| Seed | Rank 1 | Rank 2 | Rank 3 |
|------|--------|--------|--------|
| 0 | `mlp.down_proj.weight[302135]` bit14 Δloss≈24,080 | `mlp.down_proj.weight[305054]` bit14 Δloss≈7,167 | `mlp.down_proj.weight[305003]` bit14 Δloss≈5,648 |
| 1 | `mlp.down_proj.weight[302135]` bit14 Δloss≈12,210 | `mlp.down_proj.weight[2773293]` bit14 Δloss≈4,223 | `mlp.down_proj.weight[842285]` bit14 Δloss≈3,842 |
| 2 | `mlp.gate_proj.weight[2702398]` bit14 Δloss≈3,840 | `mlp.gate_proj.weight[2013374]` bit14 Δloss≈3,768 | `mlp.up_proj.weight[2013374]` bit14 Δloss≈3,645 |

All top candidates are **bit-14 flips in FFN layers** across all seeds.

---

## 3. Experiment B: Progressive BFA — High Bits (bfloat16, bits 10–14)

**Script:** `scripts/run_progressive_bfa.py --bits 10 11 12 13 14` (default run)
**Results dir:** `experiments/results_grendel/fault_injection/progressive_bfa_bits10_11_12_13_14/`
**Model precision:** bfloat16
**Algorithm:** Progressive — recompute gradients after EVERY flip (proper Rakin et al. loop).
No-re-flip rule: once a `(layer, flat_index)` is flipped, it is excluded from future steps.
**Motivation for bfloat16:** bfloat16 has an 8-bit exponent (same range as float32, max ~3.4×10^38)
vs float16's 5-bit exponent (max 65,504). The goal was to prevent immediate NaN by allowing
the larger dynamic range to absorb flips.

### 3.1 Results

Still **NaN at flip-1** across all 3 seeds.

| Seed | Flip=1 | Flip=5 | All further |
|------|--------|--------|-------------|
| 0 | NaN | NaN | NaN |
| 1 | NaN | NaN | NaN |
| 2 | NaN | NaN | NaN |

**Why bfloat16 did not help:**
Bit-14 in bfloat16 is the **MSB of the 8-bit bfloat16 exponent**. Flipping it on a typical
weight value (e.g., 0.5 → ~1.7×10^38) produces a number within bfloat16's representable range
but still enormous. When this weight participates in a matrix multiply with any non-tiny
activation, the output overflows float32 accumulation or bfloat16 intermediate storage →
Inf → NaN. The gradient optimizer is dtype-independent: it always finds the same "kill-switch"
bit regardless of whether the model is float16 or bfloat16.

**Finding:** Gradient-guided BFA catastrophic failure at 1 flip is not a float16 artifact —
it is a property of the gradient ranking algorithm finding the globally worst exponent-bit flip.

---

## 4. Experiment C: Progressive BFA — Lower Bits (bfloat16, bits 7–9)

**Script:** `scripts/run_progressive_bfa.py --bits 7 8 9`
**Results dir:** `experiments/results_grendel/fault_injection/progressive_bfa_bits7_8_9/`
**Model precision:** bfloat16
**Algorithm:** Progressive with no-re-flip rule (same as Exp B, fixed oscillation bug).
**Bit zone:** Bits 7–9 — the **3 lowest exponent bits** in bfloat16's 8-bit exponent.
Maximum amplification from a single flip: ~2^4 = 16× (bit-9 flip changes exponent by ±8 steps,
which is ±2^3 = 8× on the weight magnitude). This is large enough to damage the model but
small enough not to cause immediate overflow at flip-1.

### 4.1 Baseline

| Model | PPL (bfloat16, PyTorch) |
|-------|------------------------|
| Qwen 2.5-0.5B-Instruct | **19.01** |

Note: bfloat16 baseline (19.01) ≈ float16 baseline (18.99). The 0.1% difference is
floating-point arithmetic non-determinism from the different matmul accumulation paths.

### 4.2 Progressive Guided Attack — Degradation Curve

PPL values and ratios vs 19.01 baseline:

| Flip count | Seed 0 PPL | Seed 0 ratio | Seed 1 PPL | Seed 1 ratio | Seed 2 PPL | Seed 2 ratio |
|-----------|-----------|-------------|-----------|-------------|-----------|-------------|
| **0 (baseline)** | 19.01 | 1.00× | 19.01 | 1.00× | 19.01 | 1.00× |
| **1** | 97.10 | **5.11×** | 217.96 | **11.47×** | 20.00 | **1.05×** |
| **5** | 519,218 | **27,312×** | 16,809,531 | **884,205×** | 374,833,814 | **19,716,784×** |
| **10** | 1.82×10^14 | 9.6T× | 675,927 | 35,555× | 1.28×10^15 | 67T× |
| **25** | 2.26×10^17 | astronomical | 9.07×10^13 | 4.77T× | 3.45×10^37 | astronomical |
| **50** | 6.94×10^20 | astronomical | 2.44×10^19 | astronomical | 2.85×10^38 | astronomical |

**Key finding:** The no-re-flip fix allows the progressive loop to run through all 50 steps
with unique weights each time. Seed 0 and seed 2 show monotonically increasing damage.
Seed 1 shows a non-monotone dip at flip-10 (675K× < flip-5's 884K×) — this is expected:
once the model is catastrophically broken at flip-5, PPL measurements become noisy and
non-monotone (all values are astronomically large regardless).

**The true degradation curve is flip=0 → flip=1 → flip=5:**
- Seed 2 gives the cleanest story: 1.00× → 1.05× → catastrophic.
- Seed 1: 1.00× → 11.47× → catastrophic.
- Seed 0: 1.00× → 5.11× → catastrophic.

The progressive attack **always causes catastrophic failure by flip-5** across all seeds.

### 4.3 Step-by-step Flip History (first 10 steps, Seed 1)

| Step | Layer | Flat index | Bit | Δloss approx |
|------|-------|-----------|-----|-------------|
| 1 | `mlp.down_proj.weight` | 2773293 | 9 | 0.95 |
| 2 | `self_attn.k_proj.weight` | 82494 | 9 | 54.79 |
| 3 | `self_attn.q_proj.weight` | 627694 | 9 | 127.05 |
| 4 | `self_attn.k_proj.weight` | 108478 | 9 | 31.99 |
| 5 | `self_attn.v_proj.weight` | 74042 | 9 | 26.99 |
| 6 | `mlp.up_proj.weight` | 601786 | 9 | 29.84 |
| 7 | `self_attn.q_proj.weight` | 312454 | 9 | 68.08 |
| 8 | `mlp.down_proj.weight` | 3142815 | 9 | 83.48 |
| 9 | `self_attn.k_proj.weight` | 25734 | 9 | 67.87 |
| 10 | `mlp.gate_proj.weight` | 856638 | 9 | 12.15 |

The delta_loss values grow rapidly (0.95 → 127 in 3 steps) because each flip compounds on
the already-corrupted model. Steps 2–9 show the progressive recomputation working correctly —
gradients computed on the partially-corrupted model predict ever-larger damage from each
successive flip.

### 4.4 Random Lower-Bit Baseline (same bit zone, no guidance)

All seeds, all flip counts: **~1.000× baseline** (completely immune).

| Seed | Flip=1 | Flip=5 | Flip=10 | Flip=25 | Flip=50 |
|------|--------|--------|---------|---------|---------|
| 0 | 0.9996× | 0.9998× | 1.0004× | 0.9997× | 0.9990× |
| 1 | 1.0000× | 0.9999× | 0.9997× | 1.0000× | 0.9996× |
| 2 | 1.0001× | 0.9997× | 0.9995× | 0.9998× | 0.9998× |

**50 random lower-bit flips = zero measurable effect.** The gradient guidance is entirely
responsible for the attack's effectiveness. Without it, bits 7–9 are harmless.

---

## 5. Cross-Experiment Comparison Table

| Attack | Precision | Bit zone | Access required | Flips to catastrophe | Deterministic? |
|--------|-----------|----------|----------------|---------------------|----------------|
| Random weight bits (GGUF Exp 1) | Q4_K_M / Q8_0 | all bits | None | Immune at 1,000 | N/A |
| **Gradient-guided, high bits** | float16 | bits 10–14 (exponent MSBs) | White-box (gradients) | **1 flip — NaN every seed** | **Yes** |
| **Gradient-guided, high bits** | bfloat16 | bits 10–14 | White-box | **1 flip — NaN every seed** | **Yes** |
| Random exponent bits (float16) | float16 | bits 10–14 | None | 5–25 flips (seed-dependent) | No (block lottery) |
| Random lower bits (bfloat16) | bfloat16 | bits 7–9 | None | **Immune at 50 flips** | N/A |
| **Progressive guided, lower bits** | bfloat16 | bits 7–9 | White-box (gradients) | **5 flips — all seeds** | Near-deterministic |
| Scale bytes — block_scale (GGUF Exp 2) | Q4_K_M | scale metadata | Format knowledge only | 1–5 flips | No (block lottery) |
| Scale bytes — super_scale_d (GGUF Exp 2) | Q4_K_M | scale metadata | Format knowledge only | 1–3 flips | Near-deterministic |

---

## 6. Key Findings

### 6.1 Gradient guidance is dtype-independent
Both float16 and bfloat16 gradient-guided attacks cause immediate catastrophic failure at
1 flip. The optimizer identifies the same type of target: the MSB of the exponent field.
The exact weight index differs by seed (because different calibration batches → different
gradient landscapes) but the **outcome is the same**.

### 6.2 Lower exponent bits: gradient guidance turns an immune zone into an attack
Bits 7–9 in bfloat16 are completely harmless when flipped randomly (50 flips, no effect).
But with gradient guidance, 5 flips to those same bits cause catastrophic failure in all
seeds. This demonstrates that the danger is not in the bits themselves but in the guidance.
**A randomly distributed Rowhammer attack hitting bits 7–9 would do nothing. A targeted
attacker with model access would break the model in 5 flips.**

### 6.3 Progressive recomputation amplifies damage rapidly
In Experiment C (lower bits), the delta_loss predicted by the Taylor approximation grows
from ~1 at step 1 to ~127 at step 3 and ~610 at step 21. Each flip makes the corrupted
model more sensitive to the next flip, creating a compounding cascade. This compounding
effect explains why catastrophic failure occurs at flip=5 even with individually moderate
(≤16×) weight changes.

### 6.4 The scale-byte attack achieves comparable damage without model access
The progressive guided attack requires white-box access to gradients. The GGUF scale-byte
attack (Exp 2) achieves comparable destruction (catastrophic failure in 1–10 flips) with
only knowledge of the file format. For a realistic Rowhammer / storage-tampering attacker,
the scale-byte attack is the more practical threat.

---

## 7. Implementation Notes

### Files created

| File | Purpose |
|------|---------|
| `src/fault_injection/gradient_bfa.py` | One-shot gradient BFA core library |
| `src/fault_injection/progressive_bfa.py` | Progressive BFA core library (proper Rakin loop + no-re-flip rule) |
| `scripts/run_grad_bfa.py` | Runner for one-shot float16 experiment |
| `scripts/run_progressive_bfa.py` | Runner for progressive bfloat16 experiment; supports `--bits` to select zone |

### Environment
All PyTorch experiments ran in a conda env at `/tmp/rrgundam/bfa_env` (local disk, not home NFS).
The env uses CUDA 12.x with PyTorch 2.x and HuggingFace Transformers.

### Model
`models/hf/Qwen2.5-0.5B-Instruct/` — downloaded from HuggingFace bartowski namespace.
494M parameters. VRAM usage: 0.92 GB (bfloat16/float16).

### PPL computation
Non-overlapping 512-token windows on WikiText-2 test set, matching `llama-perplexity -c 512`
methodology used in GGUF experiments. Context is 512 to match.

### No-re-flip rule
After any `(layer_name, flat_index)` is flipped, it is excluded from all subsequent gradient
scoring steps. This prevents the oscillation observed in the naive implementation where
a weight is repeatedly flipped back and forth once the model is catastrophically broken.

---

## 8. Result Files

```
experiments/results_grendel/fault_injection/
├── grad_bfa/
│   ├── grad_bfa_summary.json           # All seeds, guided + random, float16
│   ├── seed_0/
│   │   ├── guided_flip_{1,5,10,25,50}_results.json
│   │   └── random_flip_{1,5,10,25,50}_results.json
│   ├── seed_1/ ...
│   └── seed_2/ ...
├── progressive_bfa/                    # High bits (10–14), NaN results
│   ├── progressive_bfa_summary.json
│   └── seed_{0,1,2}/progressive_results.json + random_flip_*_results.json
└── progressive_bfa_bits7_8_9/         # Lower bits (7–9), degradation curve
    ├── progressive_bfa_summary.json
    └── seed_{0,1,2}/progressive_results.json + random_flip_*_results.json
```
