#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$(dirname "$DATASET_DIR")")"
COMMON_SCRIPT="$DATASET_DIR/generate.sh"

"$COMMON_SCRIPT" \
    --repo-id "junyeong-nero/synthetic-ocr-images-gu" \
    --font-path "$PROJECT_DIR/fonts/gu/NotoSans-VariableFont_wdth,wght.ttf" \
    --lang "gu" \
    --size 1000 \
    --mixed \
    --label "Gujarati"
