# Synthetic OCR Image Generator

This project provides a synthetic OCR image generator with multi-language support and an evaluation pipeline for OCR/VLM models.

**Huggingface Datasets:**

- [Korean](https://huggingface.co/datasets/junyeong-nero/synthetic-ocr-images-korean)
- [Japanese](https://huggingface.co/datasets/junyeong-nero/synthetic-ocr-images-japanese)
- [Hindi](https://huggingface.co/datasets/junyeong-nero/synthetic-ocr-images-hindi)

## Quick Start

```bash
# Setup environment
uv sync

# Generate synthetic OCR images (Korean, sentence format, 100 images)
uv run main.py --lang ko --format sentence --size 100

# Evaluate a model on the dataset
uv sync --extra eval --extra transformers
uv run evaluate evaluate \
    -m "Qwen/Qwen3-VL-2B-Instruct" \
    -b transformers \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    -f sentence
```

---

# Generation

Generate synthetic OCR images in various formats.

## Supported Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `sentence` | Single-line text images | Basic OCR evaluation |
| `table` | Table structure images | Table extraction |
| `document` | Multi-element document images | Document understanding |
| `markdown` | Markdown-formatted content | Markdown conversion |
| `kie` | Key Information Extraction documents | Receipt/Invoice/Form extraction |

## Usage Examples

### Sentence Images

```bash
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format sentence \
    --size 1000 \
    --typo-ratio 0.3
```

### Table Images

```bash
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format table \
    --template invoice \
    --size 500
```

### Document Images

```bash
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format document \
    --template report \
    --size 500
```

### Markdown Images

```bash
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format markdown \
    --size 500
```

### KIE (Key Information Extraction) Images

```bash
# Generate all KIE types randomly
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format kie \
    --size 500

# Generate specific KIE document type
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format kie \
    --template receipt \
    --size 200
```

**KIE Document Types:**

- `receipt` - Store receipts (SROIE/CORD style)
- `invoice` - Business invoices
- `form` - Key-value pair forms (FUNSD style)
- `business_card` - Contact information cards

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lang` | Language code (`ko`, `ja`, `hi`) | `ko` |
| `--font-path` | Path to font file | Required |
| `--format` | Image format | `sentence` |
| `--template` | Template type for table/document/kie | Random |
| `--size` | Number of images to generate | `1000` |
| `--typo-ratio` | Ratio of typos to introduce | `0.0` |
| `--corpus-size` | Wikipedia sentences for corpus | `10000` |
| `--repo-id` | HuggingFace repo to push | None |
| `--mixed` | Generate mixed format dataset | `false` |

## Language-Specific Fonts

| Language | Code | Font |
|----------|------|------|
| Korean | `ko` | `NotoSans-VariableFont_wdth,wght.ttf` |
| Japanese | `ja` | `NotoSansJP-VariableFont_wght.ttf` |
| Hindi | `hi` | `NotoSansDevanagari-VariableFont_wdth,wght.ttf` |

## Batch Generation Scripts

```bash
# Generate for a single language
bash scripts/korean/generate.sh
bash scripts/japanese/generate.sh
bash scripts/hindi/generate.sh

# Generate for all languages
bash scripts/generate_all.sh
```

---

# Evaluation

Evaluate OCR/VLM models on synthetic datasets.

## Installation

```bash
# Install eval dependencies
uv sync --extra eval

# Install backend-specific dependencies
uv sync --extra eval --extra transformers  # For HuggingFace models
uv sync --extra eval --extra vllm          # For vLLM (Linux + CUDA)
uv sync --extra eval --extra ollama        # For Ollama
```

## Supported Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `transformers` | HuggingFace Transformers | `--extra transformers` |
| `vllm` | vLLM (high throughput) | `--extra vllm` (Linux + CUDA) |
| `sglang` | SGLang | `--extra sglang` (Linux only) |
| `ollama` | Ollama local models | `--extra ollama` |
| `openai` | OpenAI API | `OPENAI_API_KEY` env var |
| `anthropic` | Anthropic API | `ANTHROPIC_API_KEY` env var |
| `google` | Google AI API | `GOOGLE_API_KEY` env var |

## Usage Examples

### Sentence Evaluation (CER/WER)

```bash
uv run evaluate evaluate \
    -m "Qwen/Qwen3-VL-2B-Instruct" \
    -b transformers \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    -f sentence \
    --max-samples 100
```

**Metrics:** `avg_cer`, `avg_wer`, `std_cer`, `std_wer`

### Table Evaluation (TEDS)

```bash
uv run evaluate evaluate \
    -m "Qwen/Qwen3-VL-2B-Instruct" \
    -b transformers \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    --subset table \
    -f table \
    --max-samples 50
```

**Metrics:** `avg_teds`, `avg_cell_accuracy`, `avg_structure_f1`

### Document Evaluation

```bash
uv run evaluate evaluate \
    -m "Qwen/Qwen3-VL-2B-Instruct" \
    -b transformers \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    --subset document \
    -f document \
    --max-samples 50
```

**Metrics:** `avg_layout_f1`, `avg_reading_order`, `avg_kv_f1`, `avg_overall_f1`

### Markdown Evaluation

```bash
uv run evaluate evaluate \
    -m "Qwen/Qwen3-VL-2B-Instruct" \
    -b transformers \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    --subset markdown \
    -f markdown \
    --max-samples 50
```

**Metrics:** `avg_cer`, `exact_match_rate`, `normalized_match_rate`

### KIE Evaluation (Key Information Extraction)

```bash
uv run evaluate evaluate \
    -m "Qwen/Qwen3-VL-2B-Instruct" \
    -b transformers \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    --subset kie \
    -f kie \
    --max-samples 50
```

**Metrics:** `avg_entity_f1`, `avg_entity_precision`, `avg_entity_recall`, `avg_entity_accuracy`, `avg_item_f1`, `avg_overall_f1`

### Using OpenAI API

```bash
export OPENAI_API_KEY="your-api-key"

uv run evaluate evaluate \
    -m "gpt-4o" \
    -b openai \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    -f sentence \
    --max-samples 10
```

### Using vLLM (High Throughput)

```bash
uv run evaluate evaluate \
    -m "Qwen/Qwen2-VL-7B-Instruct" \
    -b vllm \
    -d "junyeong-nero/synthetic-ocr-images-korean" \
    -f sentence \
    --tensor-parallel 2 \
    --batch-size 8
```

## Evaluation CLI Reference

```bash
uv run evaluate evaluate [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Model ID (required) | - |
| `-b, --backend` | Inference backend (required) | - |
| `-d, --dataset` | HuggingFace dataset ID (required) | - |
| `-f, --format` | Evaluation format | `sentence` |
| `--subset` | Dataset subset | `default` |
| `--split` | Dataset split | `test` |
| `--batch-size` | Batch size | `1` |
| `--max-samples` | Max samples to evaluate | All |
| `--output-dir` | Output directory | `./evaluation_results` |
| `--temperature` | Generation temperature | `0.0` |
| `--max-tokens` | Max output tokens | `4096` |
| `--report-format` | Report format (`json`, `markdown`, `html`, `all`) | `all` |

## Comparing Models

```bash
uv run evaluate compare \
    results/model1/report.json \
    results/model2/report.json \
    -o comparison
```

## Python API

```python
from evaluation.pipeline import EvaluationPipeline
from evaluation.config import EvaluationConfig, ModelConfig, FormatType, InferenceBackend

# Configure evaluation
model_config = ModelConfig(
    model_id="Qwen/Qwen3-VL-2B-Instruct",
    backend=InferenceBackend.TRANSFORMERS,
)

config = EvaluationConfig(
    dataset_id="junyeong-nero/synthetic-ocr-images-korean",
    format_type=FormatType.KIE,  # sentence, table, document, markdown, kie
    model=model_config,
    max_samples=100,
)

# Run evaluation
pipeline = EvaluationPipeline(config)
output = pipeline.run()

# Access results
print(f"Entity F1: {output.metrics['avg_entity_f1']:.4f}")
print(f"Overall F1: {output.metrics['avg_overall_f1']:.4f}")
```

## Batch Evaluation Scripts

```bash
# Evaluate for a single language
bash scripts/korean/evaluate.sh
bash scripts/japanese/evaluate.sh
bash scripts/hindi/evaluate.sh

# Evaluate all languages
bash scripts/evaluate_all.sh
```

---

# Results

The results below are based on evaluations conducted with Korean text (sentence format).

| Model | Avg CER | Std CER |
|-------|---------|---------|
| allenai/olmOCR-2-7B-1025 | 0.1595 | 2.1595 |
| Qwen/Qwen3-VL-2B-Instruct | 0.1912 | 2.1570 |
| Qwen/Qwen3-VL-4B-Instruct | 0.2591 | 2.9649 |
| nanonets/Nanonets-OCR2-3B | 0.2680 | 4.3100 |
| Qwen/Qwen3-VL-8B-Instruct | 0.2902 | 4.0323 |
| NCSOFT/VARCO-VISION-2.0-1.7B-OCR | 0.3985 | 0.2703 |
| PaddlePaddle/PaddleOCR-VL | 0.4943 | 8.5313 |
| google/gemma-3-4b-it | 0.9973 | 7.2492 |
| rednote-hilab/dots.ocr | 1.9884 | 15.2084 |
| stepfun-ai/GOT-OCR-2.0-hf | 6.4971 | 16.6514 |

---

# Future Work

- **Expanding Scope**: Moving beyond basic text recognition to address more advanced document understanding tasks.
- **Target Data Types**: Our primary focus will be on generating more complex and diverse synthetic images.
- **Multi-language KIE**: Extending KIE support to additional languages with locale-specific templates.
