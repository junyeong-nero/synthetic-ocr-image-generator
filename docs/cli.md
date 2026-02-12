# CLI Reference

The project provides a unified CLI via `main.py`.

## Global Options
- `-h, --help`: Show help message and exit.

---

## `generate`
Generate synthetic OCR datasets.

```bash
python main.py generate [OPTIONS]
```

### Options
- `--repo-id`: (Required) HF Hub repository ID.
- `--font-path`: (Required) Path to a font file.
- `--lang`: Language code (default: `ko`).
- `--size`: Number of images to generate (default: `100`).
- `--format`: Format type (`sentence`, `table`, `document`, `markdown`, `kie`) (default: `sentence`).
- `--mixed`: Generate a mixed-format dataset.
- `--typo-ratio`: Ratio of words with typos (default: `0.15`).
- `--corpus-size`: Number of sentences for corpus (default: `10000`).
- `--table-size`: Table dimensions `min-max` (default: `3-8`).
- `--seed`: Random seed.
- `--output-dir`: Base directory for data (default: `./data`).

---

## `evaluate`
Run model evaluation.

```bash
python main.py evaluate [OPTIONS]
```

### Options
- `--model-config`: (Required) Path to model config YAML.
- `-d, --dataset`: (Required) HF dataset ID or local path.
- `-s, --subset`: Subsets to evaluate (comma-separated).
- `-b, --backend`: Override inference backend.
- `--split`: Dataset split (default: `train`).
- `--max-samples`: Limit evaluation samples.
- `--batch-api`: Use OpenAI Batch API.
- `--output-dir`: Results directory (default: `./evaluation_results`).
- `--report-format`: Output format (`json`, `markdown`, `html`, `all`) (default: `all`).

---

## `compare`
Compare multiple evaluation reports.

```bash
python main.py compare [REPORT_FILES...] [OPTIONS]
```

### Options
- `-o, --output`: Output file prefix (default: `comparison`).

---

## `list-backends`
List all available inference backends.

---

## `list-configs`
List all available model configurations in `configs/models/`.
