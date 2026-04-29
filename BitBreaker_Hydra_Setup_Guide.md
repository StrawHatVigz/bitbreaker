# BitBreaker — Hydra GPU Server Setup Guide

**For:** New team members setting up the BitBreaker project on NCSU Hydra  
**Cluster:** hydra.ece.ncsu.edu (NVIDIA L4 GPU, CUDA 13.2, RHEL 8)  
**Time required:** ~45 minutes (mostly waiting for installs)

---

## Before You Start

You need:
- NCSU Unity ID with Hydra access
- HuggingFace account (free) with Meta Llama license accepted
- The 8 model GGUF files (transfer from teammate or download fresh)

---

## Step 1 — SSH into Hydra

```bash
ssh <your_unity_id>@hydra.ece.ncsu.edu
```

> **Note:** Hydra has multiple nodes (hydra20, hydra25, hydra27 etc). You may land on a different node each session — this is fine. Everything is installed on the network drive (`~/`) which is accessible from all nodes.

Verify GPU is available:
```bash
nvidia-smi
```

You should see the NVIDIA L4 with ~23GB VRAM. If you don't see it, you're on the wrong server — make sure you're on hydra, not grendel login node.

---

## Step 2 — Create Project Folder Structure

```bash
mkdir -p ~/bitbreaker/{models,src/{fault_injection,evaluation,utils},experiments/results,scripts,configs}
```

Sanity check:
```bash
find ~/bitbreaker -type d
```

Should show 9 directories.

---

## Step 3 — Install Miniconda

> **Warning:** Install to `~/miniconda3`, NOT `/tmp`. The `/tmp` directory is local to each node and gets cleared. Your home directory persists across all nodes and sessions.

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
```

> **This will take 10-15 minutes** — your home directory is on a network drive (NFS). This is a one-time cost.

Once it finishes:
```bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"
echo 'eval "$(~/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda --version
```

Should print conda version. From now on, conda is available automatically in every new terminal.

---

## Step 4 — Create Python Environment

```bash
conda create -n bitbreaker python=3.11 -y
conda activate bitbreaker
python --version
```

Must say `Python 3.11.x`. Do not use system Python (3.6) or any other version.

---

## Step 5 — Build llama.cpp with CUDA

```bash
# cmake is available system-wide on hydra, no need to install
cmake --version   # should show 3.26+

cd ~/bitbreaker
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

Build takes 5-10 minutes.

Sanity check — verify CUDA was detected:
```bash
~/bitbreaker/llama.cpp/build/bin/llama-cli --version 2>&1 | head -4
```

You must see:
```
ggml_cuda_init: found 1 CUDA devices
Device 0: NVIDIA L4
```

If you don't see CUDA devices, the build failed to detect CUDA. Stop and ask teammate.

---

## Step 6 — Install Python Dependencies

```bash
conda activate bitbreaker

# CRITICAL: Must include CMAKE_ARGS for CUDA acceleration
# Without this flag, llama-cpp-python installs CPU-only (10x slower)
CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --no-cache-dir

# This build takes ~10 minutes (compiling CUDA kernels)

pip install numpy scipy pandas matplotlib seaborn tqdm rich
pip install huggingface_hub transformers datasets
pip install tenacity
```

Sanity check:
```bash
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
```

---

## Step 7 — Get the Model Files

You need 8 GGUF files in `~/bitbreaker/models/`. Either:

### Option A: SFTP from teammate's Mac (faster on campus network)

On your local machine:
```bash
sftp <your_unity_id>@hydra.ece.ncsu.edu
cd bitbreaker/models
lcd /path/to/local/bitbreaker/models
mput *
```

### Option B: Download directly on Hydra

```bash
conda activate bitbreaker
cd ~/bitbreaker/models

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

for repo, filename in models:
    print(f'Downloading {filename}...')
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir='.')
    print(f'Done: {path}')
"
```

> **Note:** Llama models require a HuggingFace account and Meta license acceptance. Go to huggingface.co/meta-llama/Llama-3.2-1B and accept the license before downloading.

Sanity check — verify all 8 files:
```bash
ls -lh ~/bitbreaker/models/
```

Expected sizes:
```
Llama-3.2-1B-Instruct-Q4_0.gguf    ~737M
Llama-3.2-1B-Instruct-Q4_K_M.gguf  ~770M
Llama-3.2-1B-Instruct-Q8_0.gguf    ~1.2G
Llama-3.2-1B-Instruct-f16.gguf     ~2.3G
qwen2.5-0.5b-instruct-fp16.gguf    ~1.2G
qwen2.5-0.5b-instruct-q4_0.gguf    ~409M
qwen2.5-0.5b-instruct-q4_k_m.gguf  ~469M
qwen2.5-0.5b-instruct-q8_0.gguf    ~644M
```

If any file is suspiciously small, it's a partial download — delete and re-download that file.

---

## Step 8 — Transfer Project Files from Teammate

Via SFTP from teammate's Mac, transfer these files:

```
configs/wikitext2_test.txt              → ~/bitbreaker/configs/
src/evaluation/evaluate_tasks.py        → ~/bitbreaker/src/evaluation/
scripts/run_baseline_perplexity.sh      → ~/bitbreaker/scripts/
scripts/run_baseline_tasks.sh           → ~/bitbreaker/scripts/
```

---

## Step 9 — Fix Scripts for Hydra

The scripts have Mac paths hardcoded. Fix them:

```bash
# Replace Mac paths with Hydra paths
OLD="/Users\/viggy\/Documents\/Grad School\/Sem_4\/ECE 591 - SW HW Co-design\/Project\/bitbreaker"
NEW="\/mnt\/ncsudrive\/v\/<your_unity_id>\/bitbreaker"

sed -i "s|$OLD|$NEW|g" ~/bitbreaker/scripts/run_baseline_perplexity.sh
sed -i "s|$OLD|$NEW|g" ~/bitbreaker/scripts/run_baseline_tasks.sh

# Fix conda path
sed -i 's|source /opt/miniconda3/etc/profile.d/conda.sh|source ~/miniconda3/etc/profile.d/conda.sh|g' ~/bitbreaker/scripts/run_baseline_tasks.sh

# Fix shell (scripts use zsh, Hydra uses bash)
sed -i 's|#!/bin/zsh|#!/bin/bash|' ~/bitbreaker/scripts/run_baseline_perplexity.sh
sed -i 's|#!/bin/zsh|#!/bin/bash|' ~/bitbreaker/scripts/run_baseline_tasks.sh

# Make executable
chmod +x ~/bitbreaker/scripts/run_baseline_perplexity.sh
chmod +x ~/bitbreaker/scripts/run_baseline_tasks.sh
```

> **Replace `<your_unity_id>` with your actual Unity ID in the NEW variable above.**

Verify paths look correct:
```bash
head -10 ~/bitbreaker/scripts/run_baseline_perplexity.sh
```

---

## Step 10 — Create Results Directories

```bash
mkdir -p ~/bitbreaker/experiments/results/baseline/perplexity
mkdir -p ~/bitbreaker/experiments/results/baseline/tasks
```

---

## Step 11 — Cache Datasets

```bash
conda activate bitbreaker
python -c "
from datasets import load_dataset
print('Caching ARC-Easy...')
load_dataset('allenai/ai2_arc', 'ARC-Easy', split='test')
print('Caching HellaSwag...')
load_dataset('Rowan/hellaswag', split='validation')
print('Done. Both datasets cached locally.')
"
```

---

## Step 12 — Final Sanity Check

Run inference on one model to confirm GPU is working:

```bash
~/bitbreaker/llama.cpp/build/bin/llama-cli \
  -m ~/bitbreaker/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p "The capital of France is" \
  -n 5 \
  --no-display-prompt \
  -ngl 99
```

You should see `CUDA 0` in the output and get a coherent completion like "Paris".

---

## Step 13 — Run Baseline Experiments

Open two terminals and run both scripts simultaneously:

**Terminal 1 — Perplexity (fast, ~15 min total):**
```bash
conda activate bitbreaker
cd ~/bitbreaker
./scripts/run_baseline_perplexity.sh
```

**Terminal 2 — Task evaluation (slower, ~1 hour total):**
```bash
conda activate bitbreaker
cd ~/bitbreaker
./scripts/run_baseline_tasks.sh
```

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `nvidia-smi: command not found` | On wrong server | SSH to hydra, not grendel login |
| `conda: command not found` | bashrc not loaded | Run `eval "$(~/miniconda3/bin/conda shell.bash hook)"` |
| `cannot execute: required file not found` | Wrong line endings or shebang | Run the sed fixes in Step 9 |
| Tasks running at 1-2 it/s | llama-cpp-python built without CUDA | Reinstall with `CMAKE_ARGS="-DGGML_CUDA=ON"` |
| Miniconda install hangs | NFS is slow | Wait 15 minutes, it will finish |
| `/tmp` files missing | Landed on different hydra node | Everything should be in `~/`, not `/tmp` |

---

## Expected Results

After both scripts finish you should have:

```
experiments/results/baseline/
├── perplexity/
│   ├── qwen_fp16_perplexity.log    PPL ~15.87
│   ├── qwen_q8_perplexity.log      PPL ~15.95
│   ├── qwen_q4km_perplexity.log    PPL ~16.43
│   ├── qwen_q4_perplexity.log      PPL ~17.54
│   ├── llama_fp16_perplexity.log   PPL ~13.88
│   ├── llama_q8_perplexity.log     PPL ~13.89
│   ├── llama_q4km_perplexity.log   PPL ~14.36
│   └── llama_q4_perplexity.log     PPL ~15.06
└── tasks/
    └── <model>_tasks.json          ARC-Easy + HellaSwag accuracy per model
```

If your numbers are wildly different from the reference PPL values above, something went wrong in setup.

---

## Appendix — Model Configs: What We Use and Why

This appendix explains the 8 model files in full detail — what they are, where they come from, and how to download them from scratch if needed.

---

### The 2 Model Families

**Qwen2.5-0.5B-Instruct**
- Made by: Alibaba
- Parameters: 500 million
- HuggingFace repo: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- Why we use it: Small enough to run thousands of fault injection experiments quickly on both M1 and L4

**Llama-3.2-1B-Instruct**
- Made by: Meta
- Parameters: 1 billion
- HuggingFace repo: `bartowski/Llama-3.2-1B-Instruct-GGUF`
- Why bartowski and not Meta directly: Meta only hosts original PyTorch weights, not GGUF format. Bartowski is a trusted community converter widely used in the llama.cpp ecosystem.
- Why we use it: Larger than Qwen, different architecture — gives us cross-model comparison

> **Important:** Llama models require accepting Meta's license on HuggingFace before downloading. Go to https://huggingface.co/meta-llama/Llama-3.2-1B, log in, and click "Agree and access repository". Takes 30 seconds. Without this, the download will return a 403 error.

---

### The 4 Quantization Levels

Each model is downloaded at 4 different precisions. These are the same model weights stored at different bit widths:

| Format | Bits per weight | Description | Approx size (0.5B / 1B) |
|---|---|---|---|
| FP16 | 16 | Full precision, no compression | 1.2GB / 2.3GB |
| Q8_0 | 8 | 8-bit integer, virtually lossless | 644MB / 1.2GB |
| Q4_K_M | 4 | 4-bit K-quant, smart per-layer allocation | 469MB / 770MB |
| Q4_0 | 4 | 4-bit simple, cruder encoding | 409MB / 737MB |

**Why these 4 specifically?**
- FP16 is the baseline — closest to the original trained model
- Q8_0 tests whether 8-bit quantization increases vulnerability
- Q4_K_M is the most popular edge deployment format (llama.cpp default)
- Q4_0 is the most aggressive compression — expected to be most vulnerable

---

### The 8 Files

```
models/
├── qwen2.5-0.5b-instruct-fp16.gguf         Qwen FP16
├── qwen2.5-0.5b-instruct-q8_0.gguf         Qwen Q8
├── qwen2.5-0.5b-instruct-q4_k_m.gguf       Qwen Q4_K_M
├── qwen2.5-0.5b-instruct-q4_0.gguf         Qwen Q4_0
├── Llama-3.2-1B-Instruct-f16.gguf          Llama FP16
├── Llama-3.2-1B-Instruct-Q8_0.gguf         Llama Q8
├── Llama-3.2-1B-Instruct-Q4_K_M.gguf       Llama Q4_K_M
└── Llama-3.2-1B-Instruct-Q4_0.gguf         Llama Q4_0
```

Total disk space required: ~7.5GB

---

### Downloading From Scratch

**Step 1 — HuggingFace login**

Create a free account at huggingface.co if you don't have one. Then:

```bash
conda activate bitbreaker
pip install huggingface_hub   # already installed if you followed main guide
hf auth login
# paste your HuggingFace token when prompted
# get token from: huggingface.co/settings/tokens (Read access is enough)
```

Accept Meta's license at: https://huggingface.co/meta-llama/Llama-3.2-1B

**Step 2 — Verify available files on HuggingFace**

Before downloading, confirm exact filenames (these can change as repos update):

```bash
python -c "
from huggingface_hub import list_repo_files
print('=== Qwen2.5-0.5B ===')
for f in list_repo_files('Qwen/Qwen2.5-0.5B-Instruct-GGUF'):
    if f.endswith('.gguf'):
        print(f)
print()
print('=== Llama-3.2-1B ===')
for f in list_repo_files('bartowski/Llama-3.2-1B-Instruct-GGUF'):
    if f.endswith('.gguf'):
        print(f)
"
```

**Step 3 — Download all 8 files**

```bash
cd ~/bitbreaker/models

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

for repo, filename in models:
    print(f'Downloading {filename}...')
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir='.')
    print(f'Done: {path}')
    print()

print('All 8 models downloaded.')
"
```

Downloads run sequentially. Total download time on campus network: ~10-15 minutes.

**Step 4 — Verify all 8 files**

```bash
ls -lh ~/bitbreaker/models/*.gguf | awk '{print $5, $9}'
```

If any file is under 100MB it downloaded incorrectly. Delete it and re-run the download script for that file only.

---

### Setting Up Evaluation Datasets and Scripts

There are two separate evaluation pipelines — one for perplexity, one for task accuracy. They use completely different tools.

---

#### Pipeline 1 — Perplexity (WikiText-2)

**What it is:** WikiText-2 is a collection of clean Wikipedia articles used to measure how confused a model is. Lower perplexity = healthier model.

**Tool used:** `llama-perplexity` — a CLI binary built from llama.cpp source. No Python involved.

**Setup — prepare the dataset file:**

```bash
conda activate bitbreaker
cd ~/bitbreaker

python -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
text = '\n'.join(ds['text'])
with open('configs/wikitext2_test.txt', 'w') as f:
    f.write(text)
print(f'Saved {len(text):,} characters to configs/wikitext2_test.txt')
"
```

Sanity check:
```bash
wc -l configs/wikitext2_test.txt
# Should be ~7000 lines
```

**How to run perplexity on one model manually:**
```bash
~/bitbreaker/llama.cpp/build/bin/llama-perplexity \
  -m ~/bitbreaker/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -f ~/bitbreaker/configs/wikitext2_test.txt \
  -ngl 99 \
  --ctx-size 512
```

Final output looks like:
```
Final estimate: PPL = 16.34 +/- 0.12
```

---

#### Pipeline 2 — Task Accuracy (ARC-Easy + HellaSwag)

**What they are:**
- **ARC-Easy:** 2,376 elementary science multiple choice questions (4 options each)
- **HellaSwag:** 10,042 commonsense reasoning questions (4 sentence completions each)

**Tool used:** `evaluate_tasks.py` — a custom Python script using llama-cpp-python directly. We wrote this ourselves because lm_eval (the standard harness) has a version incompatibility with current llama.cpp.

**Why we didn't use lm_eval:** lm_eval's gguf backend requires a running llama.cpp server and expects a specific logprobs response format that current llama.cpp no longer returns. Rather than fighting version pinning, we wrote a lightweight evaluator (~100 lines) that loads the model directly and scores multiple choice options using log probabilities. Same methodology, no compatibility issues.

**Setup — get the evaluate_tasks.py script:**

Transfer from teammate via SFTP (already covered in Step 8 of main guide), or copy it to:
```
~/bitbreaker/src/evaluation/evaluate_tasks.py
```

**Setup — cache the datasets:**

```bash
conda activate bitbreaker

python -c "
from datasets import load_dataset
print('Caching ARC-Easy...')
load_dataset('allenai/ai2_arc', 'ARC-Easy', split='test')
print('Caching HellaSwag...')
load_dataset('Rowan/hellaswag', split='validation')
print('Both cached. Will load from disk instantly in future runs.')
"
```

> **Important:** Do this once before running any experiments. After caching, every subsequent run loads from ~/.cache/huggingface/datasets/ instantly with no network needed.

**How to run task evaluation on one model manually:**
```bash
conda activate bitbreaker

python ~/bitbreaker/src/evaluation/evaluate_tasks.py \
  --model ~/bitbreaker/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tasks arc_easy hellaswag \
  --output ~/bitbreaker/experiments/results/baseline/tasks/qwen_q4km_tasks.json \
  --label qwen_q4km \
  --n-gpu-layers 99 \
  --num-samples 2500
```

Arguments explained:
- `--model` — path to the GGUF file
- `--tasks` — which benchmarks to run (arc_easy, hellaswag, or both)
- `--output` — where to save the JSON results file
- `--label` — identifier for this run, goes into the JSON
- `--n-gpu-layers 99` — offload all layers to GPU (always use this on Hydra)
- `--num-samples 2500` — cap at 2500 questions per task (ARC-Easy only has 2376 so runs in full; HellaSwag capped at 2500 out of 10042)

**Quick test with 20 samples** (runs in ~30 seconds to verify pipeline works):
```bash
python ~/bitbreaker/src/evaluation/evaluate_tasks.py \
  --model ~/bitbreaker/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tasks arc_easy \
  --output /tmp/test_output.json \
  --label test \
  --n-gpu-layers 99 \
  --num-samples 20
```

Expected output:
```
ARC-Easy accuracy: 0.65xx (13/20)
```

If you see GPU layers loading and accuracy around 60-70%, the pipeline is working correctly.

**Output format:** Each run produces a JSON file with per-question results:
```json
{
  "label": "qwen_q4km",
  "model_path": "...",
  "tasks": {
    "arc_easy": {
      "accuracy": 0.6595,
      "correct": 1567,
      "total": 2376,
      "results": [...]
    },
    "hellaswag": {
      "accuracy": 0.3728,
      "correct": 932,
      "total": 2500,
      "results": [...]
    }
  }
}
```

The results array contains every individual question — what was predicted, what was correct, and raw scores for each option. This is essential for fault injection analysis later.

---

#### Reference Baseline Numbers

After running both pipelines on all 8 models, your numbers should be close to these:

**Perplexity (lower is better):**
```
Qwen 0.5B   FP16     ~15.87
Qwen 0.5B   Q8       ~15.95
Qwen 0.5B   Q4_K_M   ~16.43
Qwen 0.5B   Q4_0     ~17.54
Llama 1B    FP16     ~13.88
Llama 1B    Q8       ~13.89
Llama 1B    Q4_K_M   ~14.36
Llama 1B    Q4_0     ~15.06
```

**Task Accuracy:**
```
Qwen 0.5B   FP16     ARC ~65.7%   HellaSwag ~38.0%
Qwen 0.5B   Q8       ARC ~66.1%   HellaSwag ~37.8%
Qwen 0.5B   Q4_K_M   ARC ~65.9%   HellaSwag ~37.3%
Llama 1B    FP16     ARC ~69.1%   HellaSwag ~xx.x%
```

> Small differences (±1-2%) from these reference values are normal due to floating point differences between hardware.

---

### What is GGUF?

GGUF (GGML Unified Format) is the file format used by llama.cpp to store LLM weights. Think of it as a container — like a ZIP file — that holds the model weights at whatever precision was chosen during quantization, plus metadata like tokenizer config and architecture info.

Key point for this project: **your fault injection tool directly manipulates bytes inside these GGUF files.** Understanding the format matters because:
- The file contains both weight data and metadata
- Flipping bits in metadata will crash the model, not degrade it
- Flipping bits in scale factors (K-quant specific) has different effects than flipping weight bits
- Each quantization format has a different internal binary layout

The llama.cpp source code at `llama.cpp/src/llama.cpp` and `ggml/src/gguf.cpp` is the ground truth for the binary format if you need to understand the layout for fault injection targeting.
