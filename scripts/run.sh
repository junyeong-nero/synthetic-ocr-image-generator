uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --corpus-size 10000 \
    --size 1000 \
    --typo-ratio 0.4

uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean-no-typos" \
    --corpus-size 10000 \
    --size 1000 \
    --typo-ratio 0.0