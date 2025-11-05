uv run src/evaluate.py \
    "dummy" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-dummy" \
    --batchsize 4

uv run src/evaluate.py \
    "rednote-hilab/dots.ocr" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-dots.ocr" \
    --batchsize 8 

uv run src/evaluate.py \
    "nanonets/Nanonets-OCR2-3B" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-Nanonets-OCR2-3B" \
    --batchsize 8 

uv run src/evaluate.py \
    "lightonai/LightOnOCR-1B-1025" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-LightOnOCR-1B-1025" \
    --batchsize 8 

uv run src/evaluate.py \
    "allenai/olmOCR-2-7B-1025" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-olmOCR-2-7B-1025" \
    --batchsize 8 

uv run src/evaluate.py \
    "deepseek-ai/DeepSeek-OCR" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-DeepSeek-OCR" \
    --batchsize 8 

uv run src/evaluate.py \
    "google/gemma-3-4b-it" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-gemma-3-4b-it" \
    --batchsize 8 

uv run src/evaluate.py \
    "stepfun-ai/GOT-OCR-2.0-hf" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-GOT-OCR-2.0-hf" \
    --batchsize 8 

uv run src/evaluate.py \
    "PaddlePaddle/PaddleOCR-VL" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-PaddleOCR-VL" \
    --batchsize 8 

uv run src/evaluate.py \
    "Qwen/Qwen3-VL-2B-Instruct" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-Qwen3-VL-2B-Instruct" \
    --batchsize 8 

uv run src/evaluate.py \
    "NCSOFT/VARCO-VISION-2.0-1.7B-OCR" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-VARCO-VISION-2.0-1.7B-OCR" \
    --batchsize 8