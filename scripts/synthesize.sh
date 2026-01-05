#!/bin/bash
# Document Generation Examples

# Generate Korean documents with invoice template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template invoice \
    --size 100 \
    --typo-ratio 0.0

# Generate Korean documents with receipt template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template receipt \
    --size 100 \
    --typo-ratio 0.0

# Generate Korean documents with form template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template form \
    --size 100 \
    --typo-ratio 0.0

# Generate Korean documents with letter template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template letter \
    --size 100 \
    --typo-ratio 0.0

# Generate Korean documents with report template
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document \
    --template report \
    --size 100 \
    --typo-ratio 0.0

# Generate Japanese documents with mixed templates
uv run main.py \
    --lang ja \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-japanese" \
    --format document \
    --size 200 \
    --typo-ratio 0.0
