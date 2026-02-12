# Data Generation Guide

The generation pipeline allows you to create diverse synthetic OCR datasets tailored to specific tasks.

## Basic Usage

The primary command for generation is `python main.py generate`.

### Required Arguments
- `--repo-id`: The Hugging Face repository where the dataset will be uploaded.
- `--font-path`: Path to a font file used for generating the character similarity database.
- `--lang`: Language code (e.g., `ko`, `en`).

## Generation Formats

### 1. Sentence (`--format sentence`)
Generates single lines of text.
- **Corpus**: Extracts text from Wikipedia.
- **Typos**: Introduces realistic typos based on character similarity.
- **Parameters**:
    - `--corpus-size`: Number of sentences to extract (default: 10,000).
    - `--typo-ratio`: Probability of introducing a typo in a word (default: 0.15).

### 2. Table (`--format table`)
Generates tabular images.
- **Parameters**:
    - `--table-size`: Range of rows and columns as `min-max` (default: `3-8`).
    - `--template`: (Optional) Specific table style template.

### 3. Document (`--format document`)
Generates full-page document layouts.

### 4. Markdown (`--format markdown`)
Renders images from Markdown content templates.

### 5. KIE (`--format kie`)
Generates Key Information Extraction data (e.g., forms, receipts).
- **Parameters**:
    - `--template`: Specific document type (e.g., `invoice`, `receipt`).

## Mixed Format Generation

To generate a balanced dataset containing all formats, use the `--mixed` flag:

```bash
python main.py generate 
    --repo-id "username/mixed-ocr-dataset" 
    --font-path "fonts/ko/my-font.ttf" 
    --lang "ko" 
    --size 1000 
    --mixed
```

This will automatically distribute the `--size` across all supported formats and upload them as separate subsets to the Hugging Face Hub.

## Character Similarity Database

For the `sentence` format, the pipeline first generates a similarity database (`char_similarity_db_<lang>.json`). This database maps characters to visually similar ones using SSIM (Structural Similarity Index).
- `--similarity-threshold`: Minimum SSIM to consider characters "similar" (default: 0.6).
- `--similarity-top-k`: Maximum number of similar characters to store per character (default: 8).

## Advanced Configuration

- `--seed`: Set a random seed for reproducible generation.
- `--output-dir`: Change the local storage path for generated images (default: `./data`).
