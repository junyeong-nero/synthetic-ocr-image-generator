uv run src/evaluate.py \
    "rednote-hilab/dots.ocr" \
    --target-column "typo_text" \
    --propmt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-result" \

