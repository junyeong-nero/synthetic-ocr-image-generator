# Synthetic OCR Image Generator

This project provides a synthetic OCR image generator with multi-language support.

**Huggingface Datasets:**
- [Korean](https://huggingface.co/datasets/junyeong-nero/synthetic-ocr-images-korean)
- [Japanese](https://huggingface.co/datasets/junyeong-nero/synthetic-ocr-images-japanese)
- [Hindi](https://huggingface.co/datasets/junyeong-nero/synthetic-ocr-images-hindi)

# How to Use

## Environment Setup

Set up the environment using `uv`:

```shell
uv sync
```

## Scripts Directory Structure

The scripts are organized by language for easy management:

```
scripts/
├── korean/
│   ├── generate.sh      # Korean OCR image generation
│   └── evaluate.sh      # Korean dataset evaluation
├── japanese/
│   ├── generate.sh      # Japanese OCR image generation
│   └── evaluate.sh      # Japanese dataset evaluation
├── hindi/
│   ├── generate.sh      # Hindi OCR image generation
│   └── evaluate.sh      # Hindi dataset evaluation
├── common/
│   └── analyze.sh       # Common analysis script
├── generate_all.sh      # Generate all languages (batch)
└── evaluate_all.sh      # Evaluate all languages (batch)
```

### Language-Specific Font Configurations

| Language | Code | Font |
|----------|------|------|
| Korean | `ko` | `NotoSans-VariableFont_wdth,wght.ttf` |
| Japanese | `ja` | `NotoSansJP-VariableFont_wght.ttf` |
| Hindi | `hi` | `NotoSansDevanagari-VariableFont_wdth,wght.ttf` |

## Running Scripts

### Generate for a Single Language

Run the language-specific generation script:

```bash
# Korean
bash scripts/korean/generate.sh

# Japanese
bash scripts/japanese/generate.sh

# Hindi
bash scripts/hindi/generate.sh
```

Each language script generates all four formats (sentence, table, document, markdown) with 1000 images per format.

### Generate for All Languages

To generate datasets for all supported languages at once:

```bash
bash scripts/generate_all.sh
```

### Evaluate for a Single Language

Run the language-specific evaluation script:

```bash
# Korean
bash scripts/korean/evaluate.sh

# Japanese
bash scripts/japanese/evaluate.sh

# Hindi
bash scripts/hindi/evaluate.sh
```

### Evaluate All Languages

To evaluate all languages and run analysis:

```bash
bash scripts/evaluate_all.sh
```

This will evaluate all language datasets and then run the common analysis script.

## Manual Script Usage

You can also run `main.py` directly with custom parameters:

```bash
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --corpus-size 10000 \
    --size 1000 \
    --typo-ratio 0.4
```

### Generating Different Formats

The generator supports multiple image formats:

**Sentence images** (default):
```bash
uv run main.py \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format sentence \
    --size 1000
```

**Table images**:
```bash
uv run main.py \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format table \
    --template invoice \
    --size 100
```

**Document images**:
```bash
uv run main.py \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --format document \
    --template invoice \
    --size 100
```

### Parameters:

- `lang`: Specifies the language for text generation.
- `font-path`: Path to the font directory used for calculating character-level similarity.
- `repo-id`: Hugging Face repository ID to update.
- `corpus-size`: The number of sentences to generate for the corpus, sourced from the [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) dataset.
- `size`: The total number of synthetic images to generate for the dataset.
- `typo-ratio`: The ratio of typos to introduce into the generated text.
- `format`: Format of images to generate (`sentence`, `table`, or `document`).
- `template`: Template for table or document generation (`invoice`, `receipt`, `form`, `letter`, `report`).
- `table-size`: Table size range as `min_rows-max_cols` (e.g., `3-8` for 3-8 rows and columns).
- `mixed`: Generate mixed format dataset (sentence, table, document combined).

### Generating Mixed Format Datasets

Generate a dataset with all three formats combined:

```bash
uv run main.py \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --mixed \
    --size 300 \
    --typo-ratio 0.15
```

# Evaluation

The evaluation script supports three modes for evaluating OCR models:

1. **Using pre-computed predictions** (recommended for custom models)
2. **Using a custom inference function** (Python API)
3. **Using built-in models** (requires vLLM/Transformers)

## Evaluating Your Own Model

### Option 1: Using Pre-computed Predictions (CLI)

Run your model separately and save predictions to a file, then evaluate:

```bash
# Predictions file formats supported:
# - .json: ["prediction1", "prediction2", ...]
# - .jsonl: {"prediction": "text"}\n{"prediction": "text"}\n...
# - .txt: one prediction per line

python src/evaluate.py \
    --predictions my_predictions.jsonl \
    --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format sentence \
    --output-file results.json
```

### Option 2: Using Custom Inference Function (Python API)

```python
from evaluate import evaluate

# Define your inference function
def my_ocr_model(images, prompts):
    results = []
    for img, prompt in zip(images, prompts):
        # Your model inference logic here
        result = your_model.predict(img, prompt)
        results.append(result)
    return results

# Run evaluation
result = evaluate(
    format_type="sentence",
    dataset="junyeong-nero/synthetic-ocr-images-korean",
    inference_fn=my_ocr_model,
    target_column="typo_text",
    batchsize=4,
)

print(f"Average CER: {result['metrics']['avg_cer']:.4f}")
```

### Option 3: Using Built-in Models

For built-in models (requires vLLM or Transformers installed separately):

```bash
python src/evaluate.py \
    --model-id "allenai/olmOCR-2-7B-1025" \
    --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format sentence \
    --batchsize 8
```

## Evaluation Formats

### Sentence Evaluation (CER)

```bash
python src/evaluate.py \
    --predictions predictions.jsonl \
    --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format sentence
```

**Metrics:**
- `avg_cer`: Average Character Error Rate
- `std_cer`: Standard deviation of CER

### Table Evaluation (TEDS)

```bash
python src/evaluate.py \
    --predictions predictions.jsonl \
    --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format table
```

**Metrics:**
- `avg_teds`: Tree-Edit Distance-based Similarity (structure accuracy)
- `avg_cell_accuracy`: Percentage of cells with perfect text match
- `avg_structure_f1`: Row/column detection F1 score

### Document Evaluation (Layout + Reading Order)

```bash
python src/evaluate.py \
    --predictions predictions.jsonl \
    --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \
    --format document
```

**Metrics:**
- `avg_layout_f1`: Layout element detection F1 (IoU-based)
- `avg_reading_order`: Reading order accuracy (Kendall's tau)
- `avg_kv_f1`: Key-value extraction F1 score
- `avg_overall_f1`: Combined overall F1 score

## CLI Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-id` | Built-in model ID (e.g., `allenai/olmOCR-2-7B-1025`) |
| `--predictions` | Path to predictions file (.json, .jsonl, or .txt) |
| `--dataset-id` | HuggingFace dataset ID (required) |
| `--format` | Evaluation format: `sentence`, `table`, or `document` |
| `--split` | Dataset split (default: `train`) |
| `--batchsize` | Batch size for inference (default: 1) |
| `--output-dataset-id` | Push results to HuggingFace dataset |
| `--output-file` | Save metrics to JSON file |
| `--image-column` | Image column name (default: `image`) |
| `--target-column` | Ground truth column for sentence format (default: `typo_text`) |
| `--prompt` | Custom prompt (uses format-specific default if not provided) |

## Python API

```python
from evaluate import evaluate, evaluate_sentence_metrics, evaluate_table_metrics, evaluate_document_metrics

# Mode 1: Evaluate with dataset + inference function
result = evaluate(
    format_type="table",
    dataset="your-dataset-id",
    inference_fn=your_inference_fn,
    batchsize=4,
)

# Mode 2: Evaluate pre-computed predictions directly
result = evaluate(
    format_type="sentence",
    predictions=["pred1", "pred2", ...],
    ground_truths=["gt1", "gt2", ...],
)

# Mode 3: Use format-specific evaluation functions
result = evaluate_sentence_metrics(predictions, ground_truths)
result = evaluate_table_metrics(predictions, ground_truths)
result = evaluate_document_metrics(predictions, ground_truths)
```

# Results

We attempted to use DeepSeek-OCR, but the model generated repeated, meaningless characters that did not match the target languages (e.g., "號號號號號...").

The results below are based on evaluations conducted with Korean text.

| Model                                                             | Avg CER    | Std CER    |
|-------------------------------------------------------------------|------------|------------|
| allenai/olmOCR-2-7B-1025                                          | 0.159544   | 2.159467   |
| Qwen/Qwen3-VL-2B-Instruct                                         | 0.191162   | 2.157042   |
| Qwen/Qwen3-VL-4B-Instruct                                         | 0.259124   | 2.964853   |
| nanonets/Nanonets-OCR2-3B                                         | 0.267985   | 4.309995   |
| Qwen/Qwen3-VL-8B-Instruct                                         | 0.290215   | 4.032342   |
| NCSOFT/VARCO-VISION-2.0-1.7B-OCR                                  | 0.398493   | 0.270318   |
| PaddlePaddle/PaddleOCR-VL                                         | 0.494337   | 8.531293   |
| google/gemma-3-4b-it                                              | 0.997308   | 7.249212   |
| rednote-hilab/dots.ocr                                            | 1.988376   | 15.208363   |
| stepfun-ai/GOT-OCR-2.0-hf                                         | 6.497117   | 16.651408   |

# Future Work

- Expanding Scope: Moving beyond basic text recognition to address the growing demand for document-level OCR and Key Information Extraction (KIE).
- Target Data Types: Our primary focus will be on generating more complex and diverse synthetic images.