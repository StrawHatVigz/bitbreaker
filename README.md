# BitBreaker: How Quantization Shapes Bit-Flip Vulnerability in Edge LLMs

**ECE 591 — Hardware-Software Co-Design | NCSU**  
**Team:** Ratan Gundami, Vignesh Sankararaman

A systematic fault injection security analysis measuring how GGUF quantization level (FP16, Q8\_0, Q4\_K\_M, Q4\_0) affects bit-flip vulnerability in small LLMs (Qwen 2.5-0.5B, Llama 3.2-1B) on edge hardware.

---

## Repository Structure

```
bitbreaker/
├── configs/
│   └── wikitext2_test.txt          WikiText-2 test split (perplexity dataset)
│
├── docs/
│   └── figures/                    Result figures and the script that generates them for the bfa experiments.
│       ├── generate.py
│       └── fig1_*.png  …  fig7_*.png
│   └── grad_bfa_results.md         Markdown summary of gradient-guided BFA findings
│
├── experiments/
│   ├── maps/                       Pre-built GGUF byte-range maps for all 8 models
│   │   └── <model>_map.json        Used by fault injector to locate weight/scale bytes
│   │
│   └── results/                    Mac (M1) experiment outputs
│       ├── baseline/               Clean model baselines
│       │   ├── perplexity/         llama-perplexity logs (.log)
│       │   └── tasks/              ARC-Easy + HellaSwag results (.json + .log)
│       └── fault_injection/
│           ├── calibration/        Early calibration runs (llama Q4_K_M)
│           ├── exp1_random_sweep/  Experiment 1: random flip degradation curves
│           ├── exp2_scale_vs_weight/ Experiment 2: scale vs weight bit damage
│           ├── exp3a_role_targeting/ Experiment 3a: attention vs FFN vulnerability
│           └── exp3b_depth_targeting/ Experiment 3b: early/middle/late layer targeting
│
│   └── results_grendel/            GPU cluster experiment outputs
│       ├── baseline/               (same structure as results/)
│       └── fault_injection/
│           ├── exp1_random_sweep/
│           ├── exp2_scale_vs_weight/
│           ├── exp3a_role_targeting/
│           ├── exp3b_depth_targeting/
│           ├── grad_bfa/           Gradient-guided BFA results
│           ├── progressive_bfa/    Progressive BFA (cumulative flips)
│           └── progressive_bfa_bits7_8_9/  Bit-position sensitivity sweep
│
├── figures/                        Cross-platform comparison figures
│   ├── exp2_degradation_grendel.png
│   └── exp2_degradation_mac.png
│
├── results_tables/                 Human-readable summary tables (.txt)
│   ├── exp1_grendel.txt
│   ├── exp2_grendel.txt  exp2_mac.txt
│   ├── exp3a_grendel.txt  exp3a_mac.txt
│   └── exp3b_grendel.txt  exp3b_mac.txt
│
├── scripts/                        Entry-point scripts — run experiments from here
│   ├── run_baseline_perplexity.sh  Baseline perplexity on all 8 models
│   ├── run_baseline_tasks.sh       Baseline ARC-Easy + HellaSwag on all 8 models
│   ├── run_build_maps.sh           Build GGUF byte-range maps (run once before exps)
│   ├── run_build_maps.py           Python wrapper for map building
│   ├── run_exp1.py                 Experiment 1: random flip sweep
│   ├── run_exp2.py                 Experiment 2: scale vs weight bits
│   ├── run_exp3a.py                Experiment 3a: attention vs FFN targeting
│   ├── run_exp3b.py                Experiment 3b: layer depth targeting
│   ├── run_grad_bfa.py             Gradient-guided BFA  — CUDA only (requires PyTorch + GPU)
│   ├── run_progressive_bfa.py      Progressive BFA with cumulative flip tracking  — CUDA only
│   ├── run_calibration.py          Early calibration run (Q4_K_M)
│   ├── parse_reports.py            Parse result JSONs into summary tables
│   ├── plot_results.py             Generate figures from result JSON files
│   ├── plot_exp2.py                Exp 2 specific figure generation
│   └── final_results_cleanup.py    Post-processing / cleanup utilities
│
└── src/                            Core library — imported by scripts/
    ├── evaluation/
    │   └── evaluate_tasks.py       Custom ARC-Easy + HellaSwag evaluator
    │                               (uses llama-cpp-python, avoids lm_eval compat issues)
    └── fault_injection/
        ├── build_model_map.py      Parse GGUF binary format → byte-range map
        ├── model_map.py            ModelMap class: load maps, sample weight/scale bytes
        ├── gguf_tensor_mapper.py   Low-level GGUF tensor offset resolver
        ├── fault_injector.py       Core engine: inject → benchmark → restore cycle
        ├── gradient_bfa.py         Gradient-guided bit flip targeting (BFA methodology)
        ├── progressive_bfa.py      Progressive BFA: cumulative flips with checkpoints
        └── sanity_check_map.py     Verify a generated map against the GGUF file
```

---

## Quickstart: Can You Reproduce This?

Yes — the fault injection framework is pure Python + a compiled llama.cpp binary. No special hardware is required to run it, though a GPU (CUDA or Apple Metal) is needed to reproduce the experiments in a reasonable time.

### Platform Requirements

| Platform | Notes |
|---|---|
| **Apple M1/M2/M3 MacBook Pro** | Primary development platform; Metal GPU acceleration |
| **NVIDIA L4 GPU (Linux)** | Full experiment suite including grad BFA |
| **Any Linux + CUDA GPU** | Pass `--platform linux`; set `BITBREAKER_N_GPU_LAYERS` to match your VRAM |
| **CPU only** | Slow but functional for small-scale validation; set `BITBREAKER_N_GPU_LAYERS=0` |

### Dependencies

- Python 3.11 (conda environment recommended)
- `llama-cpp-python` built with Metal (`CMAKE_ARGS="-DGGML_METAL=ON"`) or CUDA (`-DGGML_CUDA=ON`)
- llama.cpp binary built from source (for `llama-perplexity`)
- `numpy scipy pandas matplotlib seaborn tqdm rich huggingface_hub datasets transformers`

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/StrawHatVigz/bitbreaker.git
cd bitbreaker

conda create -n bitbreaker python=3.11 -y
conda activate bitbreaker
```

### 2. Install Python dependencies

```bash
# Mac (Apple Silicon)
CMAKE_ARGS="-DGGML_METAL=ON" pip install llama-cpp-python --no-cache-dir

# Linux/CUDA
# CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --no-cache-dir

pip install numpy scipy pandas matplotlib seaborn tqdm rich
pip install huggingface_hub datasets transformers tenacity
```

### 3. Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Mac
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
# Linux/CUDA
# cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
cd ..
```

### 4. Download the 8 GGUF models

```bash
conda activate bitbreaker
mkdir -p models
python -c "
from huggingface_hub import hf_hub_download
models = [
    ('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-fp16.gguf'),
    ('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-q8_0.gguf'),
    ('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-q4_k_m.gguf'),
    ('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct-q4_0.gguf'),
    ('bartowski/Llama-3.2-1B-Instruct-GGUF', 'Llama-3.2-1B-Instruct-f16.gguf'),
    ('bartowski/Llama-3.2-1B-Instruct-GGUF', 'Llama-3.2-1B-Instruct-Q8_0.gguf'),
    ('bartowski/Llama-3.2-1B-Instruct-GGUF', 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'),
    ('bartowski/Llama-3.2-1B-Instruct-GGUF', 'Llama-3.2-1B-Instruct-Q4_0.gguf'),
]
for repo, f in models:
    hf_hub_download(repo_id=repo, filename=f, local_dir='models')
    print(f'Downloaded {f}')
"
```

> **Note:** Llama models require accepting Meta's license at https://huggingface.co/meta-llama/Llama-3.2-1B before downloading. Run `huggingface-cli login` first.

### 5. Generate WikiText-2 and cache datasets

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
open('configs/wikitext2_test.txt', 'w').write('\n'.join(ds['text']))
load_dataset('allenai/ai2_arc', 'ARC-Easy', split='test')
load_dataset('Rowan/hellaswag', split='validation')
print('Done.')
"
```

### 6. Build model maps (one-time)

```bash
bash scripts/run_build_maps.sh
```

This parses the GGUF binary format for all 8 models and writes byte-range maps to `experiments/maps/`. Pre-built maps are already committed to the repo, so this step is only needed if you add new models.

---

## Running Experiments

All experiment scripts accept `--platform` and `--n-gpu-layers`. Platform auto-detects based on OS (Mac or Linux), but can be overridden.

```bash
# Baselines
bash scripts/run_baseline_perplexity.sh
bash scripts/run_baseline_tasks.sh

# Fault injection experiments
# --platform is optional — auto-detects Mac or Linux if omitted.
# all experiment logic is identical. Pass --platform linux on any non-Mac machine.
python scripts/run_exp1.py          # random flip sweep
python scripts/run_exp2.py          # scale vs weight bits
python scripts/run_exp3a.py         # attention vs FFN
python scripts/run_exp3b.py         # layer depth targeting

# Advanced — CUDA GPU only (uses PyTorch autograd, not llama.cpp)
# These will not run on Mac or CPU-only machines
python scripts/run_grad_bfa.py
python scripts/run_progressive_bfa.py


# Dry-run to preview experiment plan without touching model files
python scripts/run_exp1.py --dry-run
```

Results are written to `experiments/results/fault_injection/<exp_name>/`.

---

## Experiment Overview

| Script | Experiment | What it measures |
|---|---|---|
| `run_exp1.py` | Random Flip Sweep | Perplexity vs flip count (25–1000) across all 8 models. Produces headline degradation curves. |
| `run_exp2.py` | Scale vs Weight Bits | Damage efficiency: flipping scale bytes vs weight bytes at the same budget. Tests whether Q4_K_M super-scale bits are disproportionately dangerous. |
| `run_exp3a.py` | Role Targeting | Attention vs FFN: which component is structurally more vulnerable? |
| `run_exp3b.py` | Depth Targeting | Early vs middle vs late layers: does layer position affect fault sensitivity? |
| `run_grad_bfa.py` | Gradient-Guided BFA | Targeted attack following Rakin et al. BFA methodology; guided vs random comparison. **CUDA GPU required.** |
| `run_progressive_bfa.py` | Progressive BFA | Cumulative flip tracking with per-checkpoint perplexity; finds the "tipping point" flip count. **CUDA GPU required.** |

---

## Key Files for Understanding the Framework

1. **`src/fault_injection/fault_injector.py`** — The core engine. Implements the inject → benchmark → restore cycle. Every experiment uses this.

2. **`src/fault_injection/model_map.py`** — The `ModelMap` class. Loads a pre-built JSON map and exposes `get_random_weight_byte()` and `get_random_scale_byte()` for sampling flip targets.

3. **`src/fault_injection/build_model_map.py`** — How we parse the GGUF binary format to know where weight bytes live vs metadata and scale factors.

4. **`src/evaluation/evaluate_tasks.py`** — Custom ARC-Easy + HellaSwag evaluator. Written instead of `lm_eval` due to a logprobs API incompatibility between lm_eval's GGUF backend and current llama.cpp.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `llama-perplexity: not found` | llama.cpp not built | Run cmake build in Step 3 |
| `from llama_cpp import Llama` fails | Wrong Python env | `conda activate bitbreaker` |
| Tasks running at 1–2 it/s | llama-cpp-python built CPU-only | Reinstall with `CMAKE_ARGS="-DGGML_METAL=ON"` |
| `map not found` error in exp scripts | Maps not built | `bash scripts/run_build_maps.sh` |
| `conda: command not found` | conda not in PATH | `source ~/miniconda3/etc/profile.d/conda.sh` |
| PPL = `nan` or `inf` after flips | Model fully corrupted by flips | Expected at high flip counts — working as intended |
