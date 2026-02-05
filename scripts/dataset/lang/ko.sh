#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(dirname "$SCRIPT_DIR")"
COMMON_SCRIPT="$DATASET_DIR/generate.sh"

"$COMMON_SCRIPT"     --repo-id "junyeong-nero/synthetic-ocr-images-ko"     --font-path "/Users/junyeong-nero/workspace/synthetic-ocr-image-generator/fonts/ko/나눔손글씨 다시 시작해.ttf"     --lang "ko"     --size 1000     --typo-ratio 0.15     --similarity-threshold 0.6     --similarity-top-k 8     --label "Korean"
