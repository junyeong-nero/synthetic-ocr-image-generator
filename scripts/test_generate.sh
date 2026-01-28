#!/bin/bash

set -e

REPO_ID="test/korean-dataset"
FONT_PATH="fonts/NotoSans-VariableFont_wdth,wght.ttf"
SIZE=2
LANG="ko"

echo "=========================================="
echo "TESTING: Generating Korean OCR images (Size: $SIZE)"
echo "=========================================="

echo ""
echo "[1/5] Generating sentence format..."
uv run main.py generate \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --format sentence \
    --size $SIZE \
    --lang $LANG \
    --typo-ratio 0.15

echo ""
echo "[2/5] Generating table format..."
uv run main.py generate \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --format table \
    --size $SIZE \
    --lang $LANG

echo ""
echo "[3/5] Generating document format..."
uv run main.py generate \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --format document \
    --size $SIZE \
    --lang $LANG

echo ""
echo "[4/5] Generating markdown format..."
uv run main.py generate \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --format markdown \
    --size $SIZE \
    --lang $LANG

echo ""
echo "[5/5] Generating kie format..."
uv run main.py generate \
    --repo-id "$REPO_ID" \
    --font-path "$FONT_PATH" \
    --format kie \
    --size $SIZE \
    --lang $LANG

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
