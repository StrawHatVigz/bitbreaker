#!/usr/bin/env bash
# scripts/run_build_maps.sh
#
# Builds GGUF byte-range maps for all 8 BitBreaker models and runs sanity checks.
# Run from the project root:
#
#   bash scripts/run_build_maps.sh
#
# Output maps go to: experiments/maps/<model_stem>_map.json
# One map per model. Safe to re-run — existing maps are overwritten.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS_DIR="$PROJECT_ROOT/models"
MAPS_DIR="$PROJECT_ROOT/experiments/maps"
BUILD_SCRIPT="$PROJECT_ROOT/src/fault_injection/build_model_map.py"
SANITY_SCRIPT="$PROJECT_ROOT/src/fault_injection/sanity_check_map.py"

mkdir -p "$MAPS_DIR"

# ── Model list ────────────────────────────────────────────────────────────────
# Ordered: FP16 → Q8_0 → Q4_K_M → Q4_0 per model family
MODELS=(
    "qwen2.5-0.5b-instruct-fp16.gguf"
    "qwen2.5-0.5b-instruct-q8_0.gguf"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    "qwen2.5-0.5b-instruct-q4_0.gguf"
    "Llama-3.2-1B-Instruct-f16.gguf"
    "Llama-3.2-1B-Instruct-Q8_0.gguf"
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    "Llama-3.2-1B-Instruct-Q4_0.gguf"
)

# ── Tracking ──────────────────────────────────────────────────────────────────
PASSED=()
FAILED=()
SKIPPED=()

# ── Per-model loop ────────────────────────────────────────────────────────────
for MODEL_FILE in "${MODELS[@]}"; do
    MODEL_PATH="$MODELS_DIR/$MODEL_FILE"
    # Strip .gguf extension, append _map.json
    MAP_NAME="${MODEL_FILE%.gguf}_map.json"
    MAP_PATH="$MAPS_DIR/$MAP_NAME"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Model : $MODEL_FILE"
    echo "════════════════════════════════════════════════════"

    # ── Check model exists ──────────────────────────────────
    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIP: model file not found"
        SKIPPED+=("$MODEL_FILE")
        continue
    fi

    # ── Build map ───────────────────────────────────────────
    echo ""
    echo "  [1/2] Building map..."
    if ! python3 "$BUILD_SCRIPT" \
        --model  "$MODEL_PATH" \
        --output "$MAP_PATH"; then
        echo "  FAILED: build_model_map.py"
        FAILED+=("$MODEL_FILE  (build failed)")
        continue
    fi

    # ── Sanity check ────────────────────────────────────────
    echo ""
    echo "  [2/2] Sanity check..."
    if ! python3 "$SANITY_SCRIPT" --map "$MAP_PATH"; then
        echo "  FAILED: sanity_check_map.py"
        FAILED+=("$MODEL_FILE  (sanity check failed)")
        continue
    fi

    PASSED+=("$MODEL_FILE")
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════"
echo ""

echo "  Maps directory: $MAPS_DIR"
echo ""

if ls "$MAPS_DIR"/*.json &>/dev/null; then
    echo "  Generated map files:"
    for f in "$MAPS_DIR"/*.json; do
        size_kb=$(du -k "$f" | cut -f1)
        echo "    ${size_kb}K  $(basename "$f")"
    done
else
    echo "  (no map files found)"
fi

echo ""

if [ ${#PASSED[@]} -gt 0 ]; then
    echo "  PASSED (${#PASSED[@]}):"
    for m in "${PASSED[@]}"; do echo "    ✓  $m"; done
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "  SKIPPED (${#SKIPPED[@]}) — model files not found:"
    for m in "${SKIPPED[@]}"; do echo "    -  $m"; done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  FAILED (${#FAILED[@]}):"
    for m in "${FAILED[@]}"; do echo "    ✗  $m"; done
    echo ""
    exit 1
fi

echo ""
echo "  All found models processed successfully."
echo ""
