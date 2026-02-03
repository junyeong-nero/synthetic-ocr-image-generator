#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMMON_SCRIPT="$PROJECT_DIR/common/generate_dataset.sh"

"$COMMON_SCRIPT"     --repo-id "junyeong-nero/synthetic-ocr-images-id"     --font-path "/Users/junyeong-nero/workspace/synthetic-ocr-image-generator/fonts/id/NotoSans-VariableFont_wdth,wght.ttf"     --lang "id"     --size 1000     --typo-ratio 0.15     --similarity-threshold 0.6     --similarity-top-k 8     --label "Indonesian"
