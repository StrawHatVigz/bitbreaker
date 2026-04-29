#!/bin/bash

# Run ARC-Easy + HellaSwag baseline evaluation on all 8 models
# Results saved to experiments/results/baseline/tasks/

BASE="/mnt/ncsudrive/r/rrgundam/NCSU/ECE591/Code/bitbreaker"
MODELS_DIR="$BASE/models"
RESULTS_DIR="$BASE/experiments/results/baseline/tasks"
EVAL_SCRIPT="$BASE/src/evaluation/evaluate_tasks.py"

# Activate conda env
source /mnt/ncsudrive/r/rrgundam/miniconda3/etc/profile.d/conda.sh
conda activate bitbreaker

# model filename : label
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

for entry in "${MODELS[@]}"; do
    filename="${entry%%:*}"
    label="${entry##*:}"
    model_path="$MODELS_DIR/$filename"
    output_path="$RESULTS_DIR/${label}_tasks.json"
    log_path="$RESULTS_DIR/${label}_tasks.log"

    echo "=========================================="
    echo "Running: $label"
    echo "Model:   $filename"
    echo "=========================================="

    python "$EVAL_SCRIPT" \
        --model "$model_path" \
        --tasks arc_easy hellaswag \
        --output "$output_path" \
        --label "$label" \
        --n-gpu-layers 99 \
        --num-samples 2500 \
        2>&1 | tee "$log_path"

    echo ""
    echo "Done: $label"
    echo ""
done

echo "=========================================="
echo "All task baselines complete."
echo "Results saved to: $RESULTS_DIR"
