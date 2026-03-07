# CLI Reference

The project provides a unified CLI via `main.py`.

Recommended invocation:

```bash
uv run main.py <command> [options]
```

## Global Options
- `-h, --help`: Show help message and exit.

---

## `corpus generate`
Generate corpus text data using an LLM provider.

```bash
uv run main.py corpus generate [OPTIONS]
```

### Options
- `--lang`: Language code to generate for (default: `ko`).
- `--lang-name`: Optional language name hint for custom or unsupported codes.
- `--category`: Specific corpus category to generate (default: all categories).
- `--count`: Number of items to generate per category (default: `1000`).
- `--provider`: LLM provider (`openai`, `anthropic`) (default: `openai`).
- `--model`: Optional provider-specific model override.
- `--output-dir`: Output directory for saved corpus files.
- `--batch-size`: Number of items requested per API call (default: `100`).

---

## `generate`
Generate synthetic OCR datasets.

```bash
uv run main.py generate [OPTIONS]
```

### Options
- `--repo-id`: Optional HF Hub repository ID. Required only when `--upload` is used or when you want it stored in `run_manifest.json` for later `publish`.
- `--output-dir`: Base directory for generated data (default: `./data`).
- `--lang`: Language code (default: `ko`).
- `--seed`: Random seed for reproducible generation.
- `--size`: Number of images to generate (default: `100`).
- `--shard-size`: Samples per shard directory.
- `--max-shards`: Limit generation to the first N planned shards.
- `--resume`: Resume a previous sharded generation run.
- `--upload`: Upload to Hugging Face Hub after generation completes.
- `--template`: Optional generation template name.
- `--template-family`: Optional template family filter.
- `--min-template-complexity`: Minimum template complexity filter (`1-5`).
- `--max-template-complexity`: Maximum template complexity filter (`1-5`).
- `--template-config-dir`: Template catalog directory path.
- `--markdown-renderer`: Markdown render backend (`pil`, `html2image`) (default: `pil`).
- `--style-profile`: Style variation profile (`legacy`, `balanced`, `aggressive`).
- `--coverage-target`: Family target ratio (`family=ratio`), repeatable.
- `--novelty-window`: Recent-sample window size for novelty guard.
- `--novelty-threshold`: Similarity threshold for novelty guard.
- `--novelty-max-attempts`: Retry count before accepting low-novelty sample.
- `--similar-char-ratio`: Ratio of similar-character substitutions (default: `0.08`).
- `--similarity-db-path`: Optional similarity DB JSON path.
- `--add-noise`, `--no-add-noise`: Enable/disable noise effect.
- `--add-blur`, `--no-add-blur`: Enable/disable blur effect.
- `--mixed`: Generate a mixed-format dataset.
- `--train-ratio`: Train split ratio in mixed mode (default: `0.9`).
- `--test-ratio`: Test split ratio in mixed mode (default: `0.1`).

Notes:

- `generate` is local-first. It writes local artifacts and shard manifests even when `--repo-id` is omitted.
- Use `--upload` for inline upload, or run `publish` later against the generated path.

---

## `publish`
Publish a previously generated dataset.

```bash
uv run main.py publish --generated-path <path> [OPTIONS]
```

### Options
- `--generated-path`: (Required) Generated dataset root containing `run_manifest.json`.
- `--repo-id`: Override the repository ID stored in the manifest.
- `--train-ratio`: Override the train ratio used for mixed publishing.
- `--test-ratio`: Override the test ratio used for mixed publishing.

---

## `evaluate`
Run model evaluation.

```bash
uv run main.py evaluate [OPTIONS]
```

### Options
- `--model-config`: (Required) Path to model config YAML.
- `-d, --dataset`: (Required) HF dataset ID or local path.
- `-b, --backend`: Override inference backend.
- `--split`: Dataset split (default: `train`).
- `--max-samples`: Limit evaluation samples.
- `--seed`: Random seed for reproducible evaluation.
- `--batch-api`: Use OpenAI Batch API.
- `--batch-poll-seconds`: Polling interval for batch status.
- `--batch-timeout-seconds`: Max wait time for batch completion.
- `--batch-completion-window`: Batch completion window (default: `24h`).
- `--output-dir`: Results directory (default: `./evaluation_result`).
- `--report-format`: Output format (`json`, `markdown`, `html`, `all`) (default: `all`).
- `--inference-only`: Run inference only and save `checkpoints.json`.
- `--evaluate-only`: Skip inference and evaluate from `checkpoints.json`.
- `--batch-size`: Override batch size from model config.
- `--temperature`: Override generation temperature.
- `--max-tokens`: Override max output tokens.
- `--api-base`: Override API base URL.
- `--tensor-parallel`: Override tensor parallel size.

---

## `compare`
Compare multiple evaluation reports.

```bash
uv run main.py compare [REPORT_FILES...] [OPTIONS]
```

### Options
- `-o, --output`: Output file prefix (default: `comparison`).

---

## `list-backends`
List all available inference backends.

---

## `list-configs`
List all available model configurations in `configs/models/`.

---

## Script Wrappers

For common workflows, these scripts are recommended:

- `scripts/synthesize/generate.sh`
- `scripts/evaluate/run.sh`
- `scripts/evaluate/run-all.sh`
- `scripts/evaluate/update-leaderboard.sh`

### `scripts/evaluate/run.sh`

Runs a single model config with dependency-group handling.

```bash
scripts/evaluate/run.sh <config_name>|--model-id <config_name_or_model_id> [evaluation options...]
```

Wrapper-specific options:

- `-m, --model-id <ref>`: Model config name or model ID.
- `-d, --dataset <repo>`: Dataset ID/path override.
- `-l, --language <code>`: Language code (default: `ko`).
- `-n, --max-samples <n>`: Limit evaluation samples.
- `--split <train|test>`: Dataset split override.

All other evaluation flags are forwarded to `uv run main.py evaluate`.

### `scripts/evaluate/run-all.sh`

Runs all configs under `configs/models/`.

```bash
scripts/evaluate/run-all.sh [--dataset <repo>] [--language <code>] [-n|--max-samples <n>] [--split <train|test>]
scripts/evaluate/run-all.sh [DATASET] [MAX_SAMPLES] [SPLIT] [LANGUAGE]
```

Notes:

- Prefer `-n, --max-samples` for sample limits.
- `-m` is still accepted for max samples as a deprecated alias.
- `-l, --language` defaults to `ko` and is forwarded to each run.

### `scripts/synthesize/generate_similarity_db.sh`

Builds language-specific character similarity DB files used by `generate`.

Key options:

- `--lang <code>`: Target language (repeatable).
- `--all`: Build for all language scripts in `scripts/synthesize/lang/`.
- `--font-path <path>`: Override font file.
- `--corpus-path <path>`: Override the final merged corpus file.
- `--generate-corpus`: Run `main.py corpus generate` first, merge the generated category files, then build the DB.
- `--corpus-provider <name>` / `--corpus-model <name>`: Control the LLM corpus generation backend.
- `--corpus-count <n>` / `--corpus-batch-size <n>`: Control how much corpus text is generated before merging.
- `--corpus-category <name>`: Limit LLM corpus generation to specific categories (repeatable).
- `--auto-generate-corpus`: Auto-generate `corpus_<lang>.txt` from Wikimedia when missing.
- `--corpus-sentences <n>`: Sentence count for auto-generated corpus (default: `100000`).
- `--db-path <path>`: Override output DB path (single language only).
- `--threshold <float>`: Similarity threshold (default: `0.6`).
- `--top-k <int>`: Max similar chars per character (default: `8`).
