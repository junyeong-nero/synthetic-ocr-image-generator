# CLI Reference

All commands run through `main.py`.

## Generate

```
uv run --group generate main.py generate [OPTIONS]
```

Options (defaults in brackets):

- `--repo-id` (required): Hugging Face dataset repo ID
- `--font-path` (required): TTF font path used for similarity DB
- `--output-dir` [./data]: base output directory
- `--lang` [ko]: language code
- `--seed` [None]: random seed for reproducible generation
- `--corpus-size` [10000]: sentences to extract for corpus
- `--size` [100]: number of images
- `--typo-ratio` [0.15]: sentence typo injection ratio
- `--format` [sentence]: one of `sentence`, `table`, `document`, `markdown`, `kie`
- `--template` [None]: generator template name (format-specific)
- `--table-size` [3-8]: row/col range for tables
- `--mixed` [false]: generate all formats and upload as subsets

Examples:

```bash
uv run main.py generate \
  --repo-id "org/synth-ocr" \
  --lang en \
  --font-path "fonts/en/YourFont.ttf" \
  --format table \
  --template invoice \
  --size 50

uv run main.py generate \
  --repo-id "org/synth-ocr" \
  --lang en \
  --font-path "fonts/en/YourFont.ttf" \
  --mixed \
  --size 200
```

## Evaluate

```
uv run --group evaluate --group <model> main.py evaluate [OPTIONS]
```

Required:

- `--model-config`: path to YAML in `configs/models/`
- `-d, --dataset`: Hugging Face dataset ID or local path

Optional:

- `-b, --backend`: `openai`, `anthropic`, `google`, `transformers`, `paddleocr`
- `-s, --subset`: comma-separated subset list (if omitted, runs all defaults)
- `--split` [train]: dataset split
- `--max-samples` [None]: limit evaluation samples
- `--seed` [None]: random seed for reproducible evaluation
- `--batch-api` [false]: use OpenAI Batch API for evaluation (OpenAI backend only)
- `--batch-poll-seconds` [60]: polling interval for batch status
- `--batch-timeout-seconds` [86400]: max wait time for batch completion
- `--batch-completion-window` [24h]: batch completion window (only `24h` supported)
- `--output-dir` [./evaluation_results]: output directory
- `--report-format` [all]: `json`, `markdown`, `html`, `all`

Overrides:

- `--batch-size`: overrides model config batch size
- `--temperature`: overrides model config temperature
- `--max-tokens`: overrides model config max tokens
- `--api-base`: custom API base URL
- `--tensor-parallel`: tensor parallel size

Examples:

```bash
uv run --group evaluate --group api main.py evaluate \
  --model-config configs/models/gpt-5.yaml \
  -d "org/synth-ocr" \
  --subset sentence \
  --max-samples 100

uv run --group evaluate --group api main.py evaluate \
  --model-config configs/models/gpt-5.yaml \
  -d "org/synth-ocr" \
  --subset sentence \
  --max-samples 100 \
  --batch-api
```

Artifacts per run:

- `report.json` / `report.md` / `report.html`
- `protocol.json`
- `leaderboard.json` / `leaderboard.md`

## Compare

```
uv run --group evaluate main.py compare report1.json report2.json -o comparison
```

Outputs `comparison.json` and `comparison.md`.

## List Backends and Configs

```
uv run --group evaluate main.py list-backends
uv run --group evaluate main.py list-configs
```

## Helper Scripts

- `./scripts/run_model.sh <config_name> [eval args...]`
- `./scripts/test_all_models.sh [DATASET] [SUBSET] [MAX_SAMPLES]`
