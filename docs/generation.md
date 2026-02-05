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

## Data Variety and Realism

To ensure high-quality synthetic data, the generator uses a centralized `DataProvider` (located in `src/generator/data_provider.py`) with a tiered data source strategy.

### Tiered Data Sources:
1.  **External Corpus (Priority 1)**: For large-scale generation (100k+ images), the provider loads data from pre-generated corpus files in `data/corpus/<lang>/`. This minimizes duplicates and ensures high variety.
2.  **Faker Integration (Priority 2)**: If corpus data is unavailable, it leverages the `Faker` library to generate realistic names, addresses, emails, phone numbers, and dates.
3.  **Curated Hardcoded Data (Fallback)**: Built-in datasets for domain-specific content like product names and technical terms.

### External Corpus Generation

You can generate a large-scale corpus using LLMs (OpenAI or Anthropic) to provide even more variety:

```bash
# Generate 1000 items for all categories in Korean using OpenAI
uv run python scripts/corpus/generate.py --lang ko --count 1000

# Generate using Anthropic
uv run python scripts/corpus/generate.py --lang en --provider anthropic --count 1000
```

Supported categories include `product_names`, `store_names`, `company_names`, `person_names`, `addresses`, `departments`, `positions`, `titles`, `paragraphs`, and `features`.

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
