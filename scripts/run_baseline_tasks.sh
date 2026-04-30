#!/usr/bin/env bash
# run_baseline_tasks.sh
#
# Run ARC-Easy + HellaSwag baseline evaluation on all 8 BitBreaker models.
# Works on Mac, Grendel/Hydra (CUDA), or any Linux system.
#
# Usage:
#   bash scripts/run_baseline_tasks.sh
#
# Override GPU layers for CPU-only systems:
#   BITBREAKER_N_GPU_LAYERS=0 bash scripts/run_baseline_tasks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS_DIR="$PROJECT_ROOT/models"
RESULTS_DIR="$PROJECT_ROOT/experiments/results/baseline/tasks"
EVAL_SCRIPT="$PROJECT_ROOT/src/evaluation/evaluate_tasks.py"

# ── Activate conda (tries common install locations) ────────────────────────────
activate_conda() {
    if command -v conda &>/dev/null; then
        conda activate bitbreaker 2>/dev/null || true
        return 0
    fi
    for conda_sh in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "/opt/miniconda3/etc/profile.d/conda.sh" \
        "/opt/anaconda3/etc/profile.d/conda.sh" \
        "/usr/local/miniconda3/etc/profile.d/conda.sh" \
        "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    do
        if [ -f "$conda_sh" ]; then
            source "$conda_sh"
            conda activate bitbreaker
            return 0
        fi
    done
    echo "WARNING: conda not found — attempting to use current Python environment."
}
activate_conda || true

# ── Validate prerequisites ─────────────────────────────────────────────────────
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: $EVAL_SCRIPT not found."
    exit 1
fi

mkdir -p "$RESULTS_DIR"

N_GPU_LAYERS="${BITBREAKER_N_GPU_LAYERS:-99}"

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
echo "  BitBreaker -- Baseline Task Evaluation"
echo "=========================================="
echo "  Project root : $PROJECT_ROOT"
echo "  GPU layers   : $N_GPU_LAYERS"
echo "  Results dir  : $RESULTS_DIR"
echo "=========================================="

for entry in "${MODELS[@]}"; do
    filename="${entry%%:*}"
    label="${entry##*:}"
    model_path="$MODELS_DIR/$filename"
    output_path="$RESULTS_DIR/${label}_tasks.json"
    log_path="$RESULTS_DIR/${label}_tasks.log"

    if [ ! -f "$model_path" ]; then
        echo "SKIP: model not found: $model_path"
        continue
    fi

    echo "=========================================="
    echo "  Running: $label  ($filename)"
    echo "=========================================="

    python "$EVAL_SCRIPT" \
        --model "$model_path" \
        --tasks arc_easy hellaswag \
        --output "$output_path" \
        --label "$label" \
        --n-gpu-layers "$N_GPU_LAYERS" \
        --num-samples 2500 \
        2>&1 | tee "$log_path"

    echo ""
    echo "Done: $label"
    echo ""
done

echo "=========================================="
echo "All task baselines complete."
echo "Results saved to: $RESULTS_DIR"
