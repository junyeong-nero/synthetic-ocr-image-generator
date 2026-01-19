#!/bin/bash

set -e

DATASET_ID="junyeong-nero/synthetic-ocr-images-korean"
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
BATCH_SIZE=1
OUTPUT_DIR="evaluation_results"

echo "=========================================="
echo "Evaluating all subsets with: $MODEL_ID"
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
    echo ""
done

echo ""
echo "=========================================="
echo "All subsets evaluation completed!"
echo "=========================================="
echo ""
echo "Results:"
for subset in "${SUBSETS[@]}"; do
    OUTPUT_FILE="$OUTPUT_DIR/${subset}_results.json"
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "--- $subset ---"
        uv run python -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
if 'metrics' in data:
    metrics = data['metrics']
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
"
    fi
done

echo ""
echo "=========================================="
echo "Aggregated Results:"
echo "=========================================="

uv run python src/evaluate.py \
    --model-id "$MODEL_ID" \
    --dataset-id "$DATASET_ID" \
    --all-subsets \
    --batchsize "$BATCH_SIZE" \
    --output-file "$OUTPUT_DIR/aggregated_results.json"

echo ""
echo "All results saved to: $OUTPUT_DIR/"
echo ""
