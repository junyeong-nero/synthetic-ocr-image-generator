#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$(dirname "$DATASET_DIR")")"
COMMON_SCRIPT="$DATASET_DIR/generate.sh"

"$COMMON_SCRIPT" \
    --repo-id "junyeong-nero/synthetic-ocr-images-gl" \
    --font-path "$PROJECT_DIR/fonts/gl/NotoSans-VariableFont_wdth,wght.ttf" \
    --lang "gl" \
    --size 1000 \
    --typo-ratio 0.15 \
    --similarity-threshold 0.6 \
    --similarity-top-k 8 \
    --label "Galician"
