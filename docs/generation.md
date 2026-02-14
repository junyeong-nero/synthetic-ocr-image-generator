# Data Generation Guide

The `generate` command creates markdown-rendered OCR images and metadata, then uploads the generated subset(s) to Hugging Face Hub.

## Basic Usage

```bash
uv run main.py generate \
  --repo-id "username/my-ocr-dataset" \
  --lang "ko" \
  --size 100
```

Required arguments:

- `--repo-id`: Hugging Face dataset repository ID.

Common arguments:

- `--lang`: Language code (default: `ko`).
- `--size`: Number of images to generate (default: `100`).
- `--output-dir`: Base local output directory (default: `./data`).
- `--seed`: Random seed for reproducibility.

## Rendering and Augmentation Options

- `--template`: Optional template name.
- `--markdown-renderer`: `pil` or `html2image` (default: `pil`).
- `--similar-char-ratio`: Character substitution ratio (default: `0.08`).
- `--similarity-db-path`: Optional path to a prebuilt character similarity DB JSON.
- `--add-noise` / `--no-add-noise`: Override noise behavior.
- `--add-blur` / `--no-add-blur`: Override blur behavior.

Example:

```bash
uv run main.py generate \
  --repo-id "username/my-ocr-dataset" \
  --lang "ko" \
  --size 500 \
  --markdown-renderer html2image \
  --similar-char-ratio 0.1 \
  --similarity-db-path data/ko/char_similarity_db_ko.json \
  --add-noise
```

## Mixed Mode

`--mixed` enables train/test split upload flow using `--train-ratio` and `--test-ratio`.

```bash
uv run main.py generate \
  --repo-id "username/my-ocr-dataset" \
  --lang "ko" \
  --size 1000 \
  --mixed \
  --train-ratio 0.9 \
  --test-ratio 0.1
```

Notes:

- Ratios must each be in `[0, 1]` and sum to `1.0`.
- Current mixed-mode generation is markdown-focused.

## Character Similarity Database

Use the helper script to build language-specific DB files:

```bash
./scripts/dataset/generate_similarity_db.sh --lang ko
```

Useful options:

- `--font-path`: Override font used while building the DB.
- `--corpus-path`: Override source corpus.
- `--auto-generate-corpus`: Auto-generate `corpus_<lang>.txt` from Wikimedia when missing.
- `--corpus-sentences`: Sentence count for auto-generated corpus (default: `100000`).
- `--db-path`: Override output JSON path (single language).
- `--threshold`: Similarity threshold (default: `0.6`).
- `--top-k`: Max similar characters per character (default: `8`).
