#!/bin/bash

set -e

usage() {
    echo "Usage: $0 --dataset-id <id> --model-config <path> --output-dir <dir> [options]"
    echo ""
    echo "Options:"
    echo "  --split <name>        Dataset split (default: train)"
    echo "  --subsets <list>      Comma-separated subsets (default: sentence,table,document,markdown,kie)"
    echo "  --report-format <fmt> Report format (json, markdown, html, all). Default: all"
    echo "  --label <label>       Display label for logs (optional)"
    echo "  --                    Pass-through args to main.py evaluate"
    exit 1
}

DATASET_ID=""
MODEL_CONFIG=""
OUTPUT_DIR=""
SPLIT="train"
SUBSETS="sentence,table,document,markdown,kie"
REPORT_FORMAT="all"
LABEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-id)
            DATASET_ID="$2"
            shift 2
            ;;
        --model-config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --subsets)
            SUBSETS="$2"
            shift 2
            ;;
        --report-format)
            REPORT_FORMAT="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@");
            break
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$DATASET_ID" || -z "$MODEL_CONFIG" || -z "$OUTPUT_DIR" ]]; then
    usage
fi

if [[ -z "$LABEL" ]]; then
    LABEL="$DATASET_ID"
fi

mkdir -p "$OUTPUT_DIR"

SUBSET_LIST=(${SUBSETS//,/ })

echo "=========================================="
echo "Evaluating dataset: $LABEL"
echo "Model config: $MODEL_CONFIG"
echo "=========================================="

for subset in "${SUBSET_LIST[@]}"; do
    echo ""
    echo "=========================================="
    echo "[$(date '+%H:%M:%S')] Evaluating subset: $subset"
    echo "=========================================="

    uv run main.py evaluate \
        --model-config "$MODEL_CONFIG" \
        -d "$DATASET_ID" \
        --subset "$subset" \
        --split "$SPLIT" \
        --report-format "$REPORT_FORMAT" \
        --output-dir "$OUTPUT_DIR/$subset" \
        "${EXTRA_ARGS[@]}"

    echo ""
    echo "[$(date '+%H:%M:%S')] Results saved to: $OUTPUT_DIR/$subset"
done

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=========================================="
