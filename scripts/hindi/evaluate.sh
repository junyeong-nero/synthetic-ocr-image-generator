#!/bin/bash

set -e

DATASET_ID="junyeong-nero/synthetic-ocr-images-hindi"
MODEL_CONFIG="configs/models/qwen3-vl-2b.yaml"
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="evaluation_results/hindi"

echo "=========================================="
echo "Evaluating Hindi dataset with config: $MODEL_CONFIG"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

SUBSETS=("sentence" "table" "document" "markdown" "kie")

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "[$(date '+%H:%M:%S')] Evaluating subset: $subset"
    echo "=========================================="

    uv run python -m evaluation.cli evaluate \
        -m "$MODEL_ID" \
        --model-config "$MODEL_CONFIG" \
        -d "$DATASET_ID" \
        --subset "$subset" \
        -f "$subset" \
        --split "train" \
        --output-dir "$OUTPUT_DIR/$subset"

    echo ""
    echo "[$(date '+%H:%M:%S')] Results saved to: $OUTPUT_DIR/$subset"
done

echo ""
echo "=========================================="
echo "Hindi evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=========================================="
