# Synthetic OCR Image Generator & Evaluation

A comprehensive toolkit for generating synthetic multi-language OCR datasets and benchmarking OCR/VLM models.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync basic dependencies (for generation)
uv sync

# 3. Install optional dependencies for evaluation
uv sync --extra eval
```

---

## Synthesize Datasets

Generate synthetic images in various formats including sentences, tables, documents, markdown, and KIE (Key Information Extraction).

**Command Syntax:**
```bash
uv run main.py generate [OPTIONS]
```

### Examples

**1. Sentence Images (Korean)**
```bash
uv run main.py generate \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format sentence \
    --size 100 \
    --typo-ratio 0.1
```

**2. Table Images (Invoice)**
```bash
uv run main.py generate \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format table \
    --template invoice \
    --size 50
```

**3. Mixed Format Dataset**
```bash
uv run main.py generate \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --mixed \
    --size 200
```

| Parameter | Description |
|-----------|-------------|
| `--format` | `sentence`, `table`, `document`, `markdown`, `kie` |
| `--lang` | Language code (`ko`, `en`, `ja`, etc.) |
| `--size` | Number of images to generate |
| `--font-path` | Path to the TTF font file (Required) |

---

## Evaluate Models

Benchmark various OCR engines and Vision Language Models (VLMs) on the generated datasets.

**Command Syntax:**
```bash
uv run main.py evaluate [OPTIONS]
```

### Examples

**1. Evaluate HuggingFace Model (Transformers)**
```bash
uv run main.py evaluate \
    --model-config configs/models/qwen2-vl-7b.yaml \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    --subset sentence \
    --max-samples 100
```

**2. Evaluate PaddleOCR (Standard Engine)**
```bash
# Evaluate Korean subset
uv run main.py evaluate \
    --model-config configs/models/paddleocr.yaml \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    --subset sentence

# Evaluate English subset
uv run main.py evaluate \
    --model-config configs/models/paddleocr.yaml \
    -d "junyeong-nero/synthetic-ocr-images-english" \
    --subset sentence
```

**3. Evaluate OpenAI Model (GPT-4o)**
```bash
export OPENAI_API_KEY="sk-..."
# Ensure configs/models/gpt-4o.yaml exists
uv run main.py evaluate \
    --model-config configs/models/gpt-4o.yaml \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    --subset table
```

---

## Evaluation Metrics

The pipeline automatically calculates appropriate metrics based on the data format:

| Format | Metric | Description | Interpretation |
|--------|--------|-------------|----------------|
| **Sentence** | **CER** | Character Error Rate | Lower is better |
| | **WER** | Word Error Rate | Lower is better |
| **Table** | **TEDS** | Tree Edit Distance-based Similarity | Higher is better (0.0 - 1.0) |
| | **Cell Acc** | Cell Content Accuracy | Higher is better |
| **Document** | **Layout F1** | Layout Element Detection Score | Higher is better |
| | **Reading Order** | Reading Sequence Accuracy | Higher is better |
| **Markdown** | **Match Rate** | Exact/Normalized String Match | Higher is better |
| **KIE** | **Entity F1** | Key-Value Extraction F1 Score | Higher is better |
