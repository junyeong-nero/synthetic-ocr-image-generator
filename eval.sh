python src/evaluate.py \
    "rednote-hilab/dots.ocr" \
    "junyeong-nero/synthetic-ocr-images-korean \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-dots.ocr" \

