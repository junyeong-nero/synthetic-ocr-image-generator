#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(dirname "$SCRIPT_DIR")"
COMMON_SCRIPT="$DATASET_DIR/generate.sh"

"$COMMON_SCRIPT"     --repo-id "junyeong-nero/synthetic-ocr-images-km"     --font-path "/Users/junyeong-nero/workspace/synthetic-ocr-image-generator/fonts/km/NotoSans-VariableFont_wdth,wght.ttf"     --lang "km"     --size 1000     --typo-ratio 0.15     --similarity-threshold 0.6     --similarity-top-k 8     --label "Khmer"
