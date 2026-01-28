#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMMON_SCRIPT="$PROJECT_DIR/common/evaluate_dataset.sh"

"$COMMON_SCRIPT" \
    --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \
    --model-config "configs/models/deepseek-ocr-2.yaml" \
    --output-dir "evaluation_results/korean" \
    --split "train" \
    --label "Korean"
