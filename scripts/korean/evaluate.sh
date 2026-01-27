#!/bin/bash

set -e

DATASET_ID="junyeong-nero/synthetic-ocr-images-korean"
MODEL_ID="deepseek-ai/DeepSeek-OCR-2"
BACKEND="transformers"
OUTPUT_DIR="evaluation_results/korean"

echo "=========================================="
echo "Evaluating Korean dataset with: $MODEL_ID"
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
        -b "$BACKEND" \
        -d "$DATASET_ID" \
        --subset "$subset" \
        -f "$subset" \
        --output-dir "$OUTPUT_DIR/$subset" \
        --split "train"

    echo ""
    echo "[$(date '+%H:%M:%S')] Results saved to: $OUTPUT_DIR/$subset"
done

echo ""
echo "=========================================="
echo "Korean evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=========================================="
