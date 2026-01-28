#!/bin/bash

set -e

DATASET_ID="junyeong-nero/synthetic-ocr-images-korean"
MODEL_CONFIG="configs/models/deepseek-ocr-2.yaml"
OUTPUT_DIR="evaluation_results/korean"

echo "=========================================="
echo "Evaluating Korean dataset with config: $MODEL_CONFIG"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

SUBSETS=("sentence" "table" "document" "markdown" "kie")

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "[$(date '+%H:%M:%S')] Evaluating subset: $subset"
    echo "=========================================="

    uv run main.py evaluate \
        --model-config "$MODEL_CONFIG" \
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
