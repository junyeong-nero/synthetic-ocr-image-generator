# CLI Reference

The project provides a unified CLI via `main.py`.

Recommended invocation:

```bash
uv run main.py <command> [options]
```

## Global Options
- `-h, --help`: Show help message and exit.

---

## `generate`
Generate synthetic OCR datasets.

```bash
uv run main.py generate [OPTIONS]
```

### Options
- `--repo-id`: (Required) HF Hub repository ID.
- `--output-dir`: Base directory for generated data (default: `./data`).
- `--lang`: Language code (default: `ko`).
- `--seed`: Random seed for reproducible generation.
- `--size`: Number of images to generate (default: `100`).
- `--template`: Optional generation template name.
- `--markdown-renderer`: Markdown render backend (`pil`, `html2image`) (default: `pil`).
- `--similar-char-ratio`: Ratio of similar-character substitutions (default: `0.08`).
- `--similarity-db-path`: Optional similarity DB JSON path.
- `--add-noise`, `--no-add-noise`: Enable/disable noise effect.
- `--add-blur`, `--no-add-blur`: Enable/disable blur effect.
- `--mixed`: Generate a mixed-format dataset.
- `--train-ratio`: Train split ratio in mixed mode (default: `0.9`).
- `--test-ratio`: Test split ratio in mixed mode (default: `0.1`).

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
- `--output-dir`: Results directory (default: `./evaluation_results`).
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

- `scripts/dataset/generate.sh`
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
- `-n, --max-samples <n>`: Limit evaluation samples.
- `--split <train|test>`: Dataset split override.

All other evaluation flags are forwarded to `uv run main.py evaluate`.

### `scripts/evaluate/run-all.sh`

Runs all configs under `configs/models/`.

```bash
scripts/evaluate/run-all.sh [--dataset <repo>] [-n|--max-samples <n>] [--split <train|test>]
scripts/evaluate/run-all.sh [DATASET] [MAX_SAMPLES] [SPLIT]
```

Notes:

- Prefer `-n, --max-samples` for sample limits.
- `-m` is still accepted for max samples as a deprecated alias.

### `scripts/dataset/generate_similarity_db.sh`

Builds language-specific character similarity DB files used by `generate`.

Key options:

- `--lang <code>`: Target language (repeatable).
- `--all`: Build for all language scripts in `scripts/dataset/lang/`.
- `--font-path <path>`: Override font file.
- `--corpus-path <path>`: Override source corpus file.
- `--auto-generate-corpus`: Auto-generate `corpus_<lang>.txt` from Wikimedia when missing.
- `--corpus-sentences <n>`: Sentence count for auto-generated corpus (default: `100000`).
- `--db-path <path>`: Override output DB path (single language only).
- `--threshold <float>`: Similarity threshold (default: `0.6`).
- `--top-k <int>`: Max similar chars per character (default: `8`).
