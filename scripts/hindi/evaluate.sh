#!/bin/bash

set -e

DATASET_ID="junyeong-nero/synthetic-ocr-images-hindi"
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
BATCH_SIZE=8
OUTPUT_DIR="evaluation_results/hindi"

echo "=========================================="
echo "Evaluating Hindi dataset with: $MODEL_ID"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

SUBSETS=("sentence" "table" "document" "markdown")

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "[$(date '+%H:%M:%S')] Evaluating subset: $subset"
    echo "=========================================="

    OUTPUT_FILE="$OUTPUT_DIR/${subset}_results.json"

    uv run python src/evaluate.py \
        --model-id "$MODEL_ID" \
        --dataset-id "$DATASET_ID" \
        --subset "$subset" \
        --batchsize "$BATCH_SIZE" \
        --output-file "$OUTPUT_FILE"

    echo ""
    echo "[$(date '+%H:%M:%S')] Results saved to: $OUTPUT_FILE"
done

echo ""
echo "=========================================="
echo "Hindi evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=========================================="
