# Synthetic OCR Image Generator

This project provides a synthetic OCR image generator.

- [Huggingface Dataset](https://huggingface.co/datasets/junyeong-nero/synthetic-ocr-images-korean)

# How to Use

## Environment Setup

Set up the environment using `uv`:

```shell
uv sync
```

## Running Scripts

To generate synthetic OCR images, run the `main.py` script via `scripts/generate.sh`:

```python
# scripts/generate.sh
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

For evaluation, we utilized vLLM and Transformers. However, this project's `uv` environment does not support direct evaluation for various OCR models due to their differing setup requirements (e.g., specific PyTorch and CUDA versions).

Please refer to `src/models` for details on integrating and inferring with different OCR models.

Example evaluation script:

```
uv run src/evaluate.py \
    "allenai/olmOCR-2-7B-1025" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-olmOCR-2-7B-1025" \
    --batchsize 8
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

# TODO

## Table Image Generation

- [ ] **Table Generator Module** (`src/generator/table_generator.py`)
  - [ ] Define table structure: rows, columns, headers, cell content
  - [ ] Support various table styles (bordered, borderless, alternating row colors)
  - [ ] Implement cell merging (colspan, rowspan)
  - [ ] Random table size generation (e.g., 2x2 ~ 10x10)
  - [ ] Support for mixed content types in cells (text, numbers, dates)

- [ ] **Table Rendering**
  - [ ] Grid-based layout rendering with PIL/Pillow
  - [ ] Configurable cell padding and margins
  - [ ] Header row styling (bold, background color differentiation)
  - [ ] Border style variations (solid, dashed, double, none)
  - [ ] Cell alignment options (left, center, right)

- [ ] **Table Data Generation**
  - [ ] Template-based table content (invoice, schedule, receipt, etc.)
  - [ ] Random numeric data generation (prices, quantities, dates)
  - [ ] Multi-language support for table headers and content

- [ ] **Table Ground Truth Format**
  - [x] HTML table representation for ground truth
  - [x] JSON structure with cell positions and content
  - [ ] Support for table structure recognition (TSR) evaluation metrics

## Document Layout Generation

- [x] **Document Generator Module** (`src/generator/document_generator.py`)
  - [x] Multi-section document layout (header, body, footer)
  - [x] Title and paragraph blocks
  - [x] Mixed content: text + tables + lists
  - [x] Page number and date stamps

- [x] **Document Templates**
  - [x] Invoice template (logo placeholder, billing info, item table, totals)
  - [x] Receipt template (store info, items, payment details)
  - [x] Form template (input fields, labels, checkboxes)
  - [x] Letter/memo template (header, greeting, body, signature)
  - [x] Report template (title, sections, tables, figures)

- [x] **Layout Variations**
  - [x] Single-column and multi-column layouts
  - [x] Margin and spacing randomization
  - [x] Background textures (paper-like, scanned document effects)
  - [x] Noise and artifacts for realistic scanned document simulation

- [x] **Document Ground Truth Format**
  - [x] Bounding box annotations for each element (title, paragraph, table, etc.)
  - [x] Reading order annotation
  - [x] Hierarchical document structure (sections, subsections)
  - [x] Key-value pair annotations for KIE tasks

## Pipeline Integration

- [x] **CLI Updates** (`main.py`)
  - [x] Add `--format` argument: `sentence`, `table`, `document`
  - [x] Add `--template` argument for document type selection
  - [x] Add `--table-size` argument for table dimension ranges
  - [x] Add `--mixed` argument for combined dataset generation

- [x] **Pipeline Refactoring** (`src/pipeline.py`)
  - [x] Abstract base generator class for unified interface
  - [x] Format-specific pipeline branches
  - [x] Combined dataset generation (mixed formats)

## Evaluation Extensions

- [ ] **Table-specific Metrics** (`src/metrics/table_edit_distance.py`)
  - [ ] Table structure recognition accuracy (TEDS score)
  - [ ] Cell-level text accuracy
  - [ ] Row/column detection metrics

- [ ] **Document-level Metrics**
  - [ ] Layout detection mAP (mean Average Precision)
  - [ ] Reading order accuracy
  - [ ] Key-value extraction F1 score