#!/bin/bash
# Synthetic OCR Image Generation Examples

# ============================================
# Sentence Generation (default format)
# ============================================

# Generate Korean sentence images
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format sentence \
    --size 100 \
    --typo-ratio 0.15

# ============================================
# Table Generation
# ============================================

# Generate Korean table images with invoice template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format table \
    --template invoice \
    --size 50 \
    --table-size 3-6

# Generate Korean table images with schedule template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format table \
    --template schedule \
    --size 50 \
    --table-size 4-6

# Generate Korean table images with product template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format table \
    --template product \
    --size 50 \
    --table-size 3-5

# ============================================
# Document Generation
# ============================================

# Generate Korean documents with invoice template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template invoice \
    --size 30 \
    --typo-ratio 0.0

# Generate Korean documents with receipt template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template receipt \
    --size 30 \
    --typo-ratio 0.0

# Generate Korean documents with form template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template form \
    --size 30 \
    --typo-ratio 0.0

# Generate Korean documents with letter template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template letter \
    --size 30 \
    --typo-ratio 0.0

# Generate Korean documents with report template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template report \
    --size 30 \
    --typo-ratio 0.0

# ============================================
# Mixed Format Generation
# ============================================

# Generate mixed format dataset (sentence + table + document)
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --mixed \
    --size 150

# Generate Japanese mixed format dataset
uv run main.py \
    --lang ja \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-japanese" \
    --mixed \
    --size 100

# ============================================
# English Generation Examples
# ============================================

# Generate English sentence images
uv run main.py \
    --lang en \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-english" \
    --format sentence \
    --size 100 \
    --typo-ratio 0.15

# Generate English table images
uv run main.py \
    --lang en \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-english" \
    --format table \
    --template invoice \
    --size 50

# Generate English document images
uv run main.py \
    --lang en \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-english" \
    --format document \
    --template report \
    --size 30
