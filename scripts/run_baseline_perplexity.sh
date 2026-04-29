#!/bin/zsh

LLAMA_BIN="/Users/viggy/Documents/Grad School/Sem_4/ECE 591 - SW HW Co-design/Project/bitbreaker/llama.cpp/build/bin/llama-perplexity"
MODELS_DIR="/Users/viggy/Documents/Grad School/Sem_4/ECE 591 - SW HW Co-design/Project/bitbreaker/models"
WIKITEXT="/Users/viggy/Documents/Grad School/Sem_4/ECE 591 - SW HW Co-design/Project/bitbreaker/configs/wikitext2_test.txt"
RESULTS_DIR="/Users/viggy/Documents/Grad School/Sem_4/ECE 591 - SW HW Co-design/Project/bitbreaker/experiments/results/baseline/perplexity"

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
    log_path="$RESULTS_DIR/${label}_perplexity.log"

    echo "=========================================="
    echo "Running: $label"
    echo "Model:   $filename"
    echo "Log:     $log_path"
    echo "=========================================="

    "$LLAMA_BIN" \
        -m "$model_path" \
        -f "$WIKITEXT" \
        -ngl 99 \
        --ctx-size 512 \
        2>&1 | tee "$log_path"

    echo ""
    echo "Done: $label"
    echo ""
done

echo "All perplexity baselines complete."
echo "Results saved to: $RESULTS_DIR"
