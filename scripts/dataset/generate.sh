#!/bin/bash

set -e

usage() {
    echo "Usage: $0 --repo-id <repo> --font-path <path> --lang <code> [options]"
    echo ""
    echo "Options:"
    echo "  --size <n>            Number of samples per format (default: 1000)"
    echo "  --typo-ratio <ratio>  Typo ratio for sentence format (default: 0.15)"
    echo "  --similarity-threshold <t>  SSIM threshold for similar chars (default: 0.6)"
    echo "  --similarity-top-k <k>      Max similar chars per char (default: 8)"
    echo "  --formats <list>      Comma-separated formats (default: table,document,markdown,kie)"
    echo "  --mixed               Generate one mixed dataset upload (train/test splits)"
    echo "  --label <label>       Display label for logs (optional)"
    echo "  --repo-id <repo>      Hugging Face dataset repo id (required)"
    echo "  --font-path <path>    Font path (required)"
    echo "  --lang <code>         Language code (required)"
    exit 1
}

REPO_ID=""
FONT_PATH=""
LANG=""
SIZE=1000
TYPO_RATIO=0.15
SIMILARITY_THRESHOLD=0.6
SIMILARITY_TOP_K=8
FORMATS="table,document,markdown,kie"
LABEL=""
MIXED=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        --font-path)
            FONT_PATH="$2"
            shift 2
            ;;
        --lang)
            LANG="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --typo-ratio)
            TYPO_RATIO="$2"
            shift 2
            ;;
        --similarity-threshold)
            SIMILARITY_THRESHOLD="$2"
            shift 2
            ;;
        --similarity-top-k)
            SIMILARITY_TOP_K="$2"
            shift 2
            ;;
        --formats)
            FORMATS="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --mixed)
            MIXED=true
            shift
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

if [[ -z "$REPO_ID" || -z "$FONT_PATH" || -z "$LANG" ]]; then
    usage
fi

if [[ -z "$LABEL" ]]; then
    LABEL="$REPO_ID"
fi

FORMATS_LIST=(${FORMATS//,/ })
TOTAL=${#FORMATS_LIST[@]}

echo "=========================================="
echo "Generating OCR images: $LABEL"
echo "=========================================="

if [[ "$MIXED" == true ]]; then
    echo ""
    echo "[1/1] Generating mixed dataset (train/test splits)..."
    uv run --group generate main.py generate \
        --repo-id "$REPO_ID" \
        --font-path "$FONT_PATH" \
        --size "$SIZE" \
        --lang "$LANG" \
        --mixed

    echo ""
    echo "=========================================="
    echo "Dataset generated!"
    echo "Dataset: https://huggingface.co/datasets/$REPO_ID"
    echo "=========================================="
    exit 0
fi

INDEX=1
for format in "${FORMATS_LIST[@]}"; do
    echo ""
    echo "[$INDEX/$TOTAL] Generating ${format} format..."
    if [[ "$format" == "sentence" ]]; then
        echo "sentence format is disabled. skipping."
        INDEX=$((INDEX + 1))
        continue
    fi

    uv run --group generate main.py generate \
        --repo-id "$REPO_ID" \
        --font-path "$FONT_PATH" \
        --format "$format" \
        --size "$SIZE" \
        --lang "$LANG"
    INDEX=$((INDEX + 1))
done

echo ""
echo "=========================================="
echo "Dataset generated!"
echo "Dataset: https://huggingface.co/datasets/$REPO_ID"
echo "=========================================="
