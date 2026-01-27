#!/bin/bash

set -e

DATASET_ID="junyeong-nero/synthetic-ocr-images-hindi"
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
BACKEND="transformers"
BATCH_SIZE=8
OUTPUT_DIR="evaluation_results/hindi"

echo "=========================================="
echo "Evaluating Hindi dataset with: $MODEL_ID"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

SUBSETS=("sentence" "table" "document" "markdown" "kie")

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "[$(date '+%H:%M:%S')] Evaluating subset: $subset"
    echo "=========================================="

    uv run evaluate evaluate \
        -m "$MODEL_ID" \
        -b "$BACKEND" \
        -d "$DATASET_ID" \
        --subset "$subset" \
        -f "$subset" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUTPUT_DIR/$subset"

    echo ""
    echo "[$(date '+%H:%M:%S')] Results saved to: $OUTPUT_DIR/$subset"
done

echo ""
echo "=========================================="
echo "Hindi evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=========================================="
