#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$(dirname "$DATASET_DIR")")"
COMMON_SCRIPT="$DATASET_DIR/generate.sh"

"$COMMON_SCRIPT" \
    --repo-id "junyeong-nero/synthetic-ocr-images-fr" \
    --font-path "$PROJECT_DIR/fonts/fr/NotoSans-VariableFont_wdth,wght.ttf" \
    --lang "fr" \
    --size 1000 \
    --label "French"
