#!/usr/bin/env bash
# run_baseline_perplexity.sh
#
# Run llama-perplexity on all 8 BitBreaker models against WikiText-2.
# Works on Mac (Metal), Grendel/Hydra (CUDA), or any Linux with a built
# llama.cpp binary.
#
# Usage:
#   bash scripts/run_baseline_perplexity.sh
#
# Prerequisites:
#   - llama.cpp built at <project_root>/llama.cpp/build/bin/llama-perplexity
#   - GGUF models in <project_root>/models/
#   - WikiText-2 text at <project_root>/configs/wikitext2_test.txt

set -euo pipefail

# ── Resolve paths relative to this script ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_BIN="$PROJECT_ROOT/llama.cpp/build/bin/llama-perplexity"
MODELS_DIR="$PROJECT_ROOT/models"
WIKITEXT="$PROJECT_ROOT/configs/wikitext2_test.txt"
RESULTS_DIR="$PROJECT_ROOT/experiments/results/baseline/perplexity"

# ── Validate prerequisites ─────────────────────────────────────────────────────
if [ ! -f "$LLAMA_BIN" ]; then
    echo "ERROR: llama-perplexity binary not found at: $LLAMA_BIN"
    echo "  Build llama.cpp first:"
    echo "    cd $PROJECT_ROOT/llama.cpp"
    echo "    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release  # Linux/CUDA"
    echo "    # OR: cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release  # Mac"
    echo "    cmake --build build --config Release -j\$(nproc)"
    exit 1
fi

if [ ! -f "$WIKITEXT" ]; then
    echo "ERROR: WikiText-2 file not found at: $WIKITEXT"
    echo "  Generate it with:"
    echo "    python -c \""
    echo "    from datasets import load_dataset"
    echo "    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')"
    echo "    open('$PROJECT_ROOT/configs/wikitext2_test.txt', 'w').write('\\n'.join(ds['text']))"
    echo "    \""
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ── Detect GPU offload flag ────────────────────────────────────────────────────
# -ngl 99: offload all layers to GPU (CUDA or Metal)
# Change to -ngl 0 for CPU-only inference
N_GPU_LAYERS="${BITBREAKER_N_GPU_LAYERS:-99}"

# ── Model list ─────────────────────────────────────────────────────────────────
MODELS=(
    "qwen2.5-0.5b-instruct-fp16.gguf:qwen_fp16"
    "qwen2.5-0.5b-instruct-q8_0.gguf:qwen_q8"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf:qwen_q4km"
    "qwen2.5-0.5b-instruct-q4_0.gguf:qwen_q4"
    "Llama-3.2-1B-Instruct-f16.gguf:llama_fp16"
    "Llama-3.2-1B-Instruct-Q8_0.gguf:llama_q8"
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf:llama_q4km"
    "Llama-3.2-1B-Instruct-Q4_0.gguf:llama_q4"
)

echo "=========================================="
echo "  BitBreaker -- Baseline Perplexity"
echo "=========================================="
echo "  Project root : $PROJECT_ROOT"
echo "  llama bin    : $LLAMA_BIN"
echo "  GPU layers   : $N_GPU_LAYERS  (override with BITBREAKER_N_GPU_LAYERS=0 for CPU)"
echo "  Results dir  : $RESULTS_DIR"
echo "=========================================="
echo ""

for entry in "${MODELS[@]}"; do
    filename="${entry%%:*}"
    label="${entry##*:}"
    model_path="$MODELS_DIR/$filename"
    log_path="$RESULTS_DIR/${label}_perplexity.log"

    if [ ! -f "$model_path" ]; then
        echo "SKIP: model file not found: $model_path"
        continue
    fi

    echo "=========================================="
    echo "  Running: $label"
    echo "  Model  : $filename"
    echo "  Log    : $log_path"
    echo "=========================================="

    "$LLAMA_BIN" \
        -m "$model_path" \
        -f "$WIKITEXT" \
        -ngl "$N_GPU_LAYERS" \
        --ctx-size 512 \
        2>&1 | tee "$log_path"

    echo ""
    echo "Done: $label"
    echo ""
done

echo "All perplexity baselines complete."
echo "Results saved to: $RESULTS_DIR"
