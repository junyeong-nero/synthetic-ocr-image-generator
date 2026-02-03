# Generation Pipeline

The generation pipeline creates synthetic OCR datasets with ground truth labels. It is capable of generating various document types to test different aspects of OCR and VLM performance.

## Usage

The primary command is `generate`:

```bash
uv run main.py generate \
    --repo-id <huggingface-repo-id> \
    --font-path <path-to-font> \
    --output-dir ./data \
    --lang ko \
    --format sentence
```

## Supported Formats

The `--format` argument controls the type of image generated:

| Format | Description | Target Use Case |
| :--- | :--- | :--- |
| `sentence` | Single lines or blocks of text. | Basic OCR text recognition. |
| `table` | Structured tables with borders and headers. | Table structure recognition. |
| `document` | Full-page layouts with mixed content. | Document understanding, layout analysis. |
| `markdown` | Rendered Markdown content. | Structured text and formatting preservation. |
| `kie` | Key Information Extraction forms (receipts, etc.). | Entity extraction and field recognition. |

## Configuration Options

-   **`--lang`**: Language code (e.g., `ko`, `en`). Determines the corpus source (Wikipedia) and character sets.
-   **`--font-path`**: Path to a TTF/OTF font file. Required for rendering.
-   **`--size`**: Number of images to generate.
-   **`--typo-ratio`**: Probability of introducing typos into the text (0.0 to 1.0).
-   **`--corpus-size`**: Number of sentences to fetch from Wikipedia for the text corpus.
-   **`--template`**: Optional template file for structured generation.
-   **`--table-size`**: For tables, the range of rows/columns (e.g., "3-8").
-   **`--mixed`**: If set, generates a mix of all supported formats.
-   **`--seed`**: Random seed for reproducibility.

## Output Structure

Generated data is saved to the `output_dir`:

```
data/
    └── <lang>/
        └── <format>/
            ├── images/
            │   ├── image_001.png
            │   └── ...
            └── metadata.jsonl
```

The `metadata.jsonl` file contains the ground truth for each image, compatible with Hugging Face Datasets.

## Conventions

-   **Metadata**: The `file_name` in `metadata.jsonl` is the relative path to the image.
-   **Fonts**: Ensure the provided font supports the characters of the target language.
