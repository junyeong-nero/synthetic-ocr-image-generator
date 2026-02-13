#!/bin/bash

set -e

usage() {
    echo "Usage: $0 --repo-id <repo> --font-path <path> --lang <code> [options]"
    echo ""
    echo "Options:"
    echo "  --size <n>            Number of markdown samples (default: 1000)"
    echo "  --similar-char-ratio <r> Similar character replacement ratio (default: 0.2)"
    echo "  --train-ratio <r>     Train split ratio for mixed mode (default: 0.9)"
    echo "  --test-ratio <r>      Test split ratio for mixed mode (default: 0.1)"
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
LABEL=""
SIMILAR_CHAR_RATIO=0.2
TRAIN_RATIO=0.9
TEST_RATIO=0.1

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
        --label)
            LABEL="$2"
            shift 2
            ;;
        --similar-char-ratio)
            SIMILAR_CHAR_RATIO="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --test-ratio)
            TEST_RATIO="$2"
            shift 2
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

echo "=========================================="
echo "Generating OCR images: $LABEL"
echo "=========================================="

echo ""
echo "[1/1] Generating mixed dataset (train/test splits)..."
uv run --group generate main.py generate \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --size "$SIZE" \
    --lang "$LANG" \
    --similar-char-ratio "$SIMILAR_CHAR_RATIO" \
    --mixed \
    --train-ratio "$TRAIN_RATIO" \
    --test-ratio "$TEST_RATIO"

echo ""
echo "=========================================="
echo "Dataset generated!"
echo "Dataset: https://huggingface.co/datasets/$REPO_ID"
echo "=========================================="
