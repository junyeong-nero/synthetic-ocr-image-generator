#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMMON_SCRIPT="$PROJECT_DIR/common/generate_dataset.sh"

"$COMMON_SCRIPT" \
    --repo-id "junyeong-nero/synthetic-ocr-images-hindi" \
    --font-path "fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf" \
    --lang "hi" \
    --size 1000 \
    --typo-ratio 0.15 \
    --label "Hindi"
