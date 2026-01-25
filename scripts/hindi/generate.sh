#!/bin/bash

set -e

REPO_ID="junyeong-nero/synthetic-ocr-images-hindi"
FONT_PATH="fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
SIZE=1000
LANG="hi"

echo "=========================================="
echo "Generating Hindi OCR images"
echo "=========================================="

echo ""
echo "[1/4] Generating sentence format..."
uv run main.py \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --format sentence \
    --size $SIZE \
    --lang $LANG \
    --typo-ratio 0.15

echo ""
echo "[2/4] Generating table format..."
uv run main.py \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --format table \
    --size $SIZE \
    --lang $LANG

echo ""
echo "[3/4] Generating document format..."
uv run main.py \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --format document \
    --size $SIZE \
    --lang $LANG

echo ""
echo "[4/4] Generating markdown format..."
uv run main.py \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --format markdown \
    --size $SIZE \
    --lang $LANG

echo ""
echo "=========================================="
echo "Hindi dataset generated!"
echo "Dataset: https://huggingface.co/datasets/$REPO_ID"
echo "=========================================="
