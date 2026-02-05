# CLI Reference

The main entry point is `main.py`.

## Global Usage

```bash
uv run main.py <command> [arguments]
```

## Commands

### `generate`
Generates synthetic OCR datasets.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--repo-id` | str | **Required** | Hugging Face Hub repository ID. |
| `--font-path` | str | **Required** | Path to font file. |
| `--output-dir` | str | `./data` | Output directory. |
| `--lang` | str | `ko` | Language code. |
| `--format` | str | `sentence` | `sentence`, `table`, `document`, `markdown`, `kie`. |
| `--size` | int | `100` | Number of images to generate. |
| `--typo-ratio` | float | `0.15` | Ratio of words with typos. |
| `--seed` | int | `None` | Random seed for reproducibility. |
| `--mixed` | flag | `False` | Generate mixed format dataset. |

### `evaluate`
Evaluates a model on a dataset.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model-config` | str | **Required** | Path to model YAML config. |
| `--dataset` | str | **Required** | Dataset ID or path. |
| `--subset` | str | `None` | Format/subset to evaluate. |
| `--batch-api` | flag | `False` | Use Batch API where available. |
| `--output-dir` | str | `./evaluation_results` | Results directory. |

### `compare`
Compares multiple evaluation reports.

```bash
uv run main.py compare report1.json report2.json -o comparison
```

### `list-configs`
Lists available model configurations and their statuses.

```bash
uv run main.py list-configs
```

### `list-backends`
Lists registered inference backends.

```bash
uv run main.py list-backends
```

## Helper Scripts

The `scripts/` directory contains several helper scripts for automation:

### Corpus Generation
Generate large-scale text corpus using LLMs for highly varied synthetic data.

```bash
uv run python scripts/corpus/generate.py --lang ko --count 1000
```

### Dataset Management
- `scripts/dataset/generate.sh`: Automated generation for specific languages.
- `scripts/dataset/evaluate.sh`: Evaluate models on generated datasets.

### Model Benchmarking
- `scripts/models/run.sh`: Run a model evaluation (handles dependency isolation).
- `scripts/models/test_all.sh`: Run evaluation for all configured models.
