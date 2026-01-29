# Synthetic OCR Image Generator and Evaluation

Generate synthetic OCR datasets across multiple formats and evaluate OCR or VLM models with consistent metrics and reports.

## Quickstart

```bash
# Install dependencies
uv sync
uv sync --group generate
uv sync --group evaluate
uv sync --group api

# Generate a dataset
uv run --group generate main.py generate \
  --repo-id "your-org/synth-ocr" \
  --lang en \
  --font-path "fonts/en/YourFont.ttf" \
  --format sentence \
  --size 100

# Evaluate a model
uv run --group evaluate --group api main.py evaluate \
  --model-config configs/models/gpt-5.yaml \
  -d "your-org/synth-ocr" \
  --subset sentence \
  --max-samples 100
```

## Core Commands

- `uv run --group generate main.py generate` - Create synthetic datasets (single format or mixed)
- `uv run --group evaluate --group <model> main.py evaluate` - Run evaluation and generate reports
- `uv run --group evaluate main.py compare` - Compare multiple JSON reports
- `uv run --group evaluate main.py list-backends` - Show supported backends
- `uv run --group evaluate main.py list-configs` - Show available model configs
- `./scripts/run_model.sh <config> ...` - Run eval with dependency group
- `./scripts/test_all_models.sh [DATASET] [SUBSET] [MAX_SAMPLES]` - Batch eval

## Key Paths

- `main.py` - CLI entry point
- `src/pipeline.py` - Generation pipeline
- `src/evaluation/` - Evaluation pipeline, runner, reports
- `src/models/` - Inference backends and registry
- `src/metrics/` - CER/WER/TEDS/Layout/KIE metrics
- `configs/models/` - Model YAML configs and prompts
- `scripts/` - Helper scripts

## Notes

- Fonts are required for generation and `--font-path` is required by the CLI.
- Generated data and reports live in `data/`, `evaluation_results/`, and `test_results/`.
- Dependency groups live in `pyproject.toml` (generate/evaluate/api + model groups).
- `evaluate` defaults to `--split train` to match generated datasets.
- Use `--seed` for reproducible generation/evaluation.
- Evaluation writes `protocol.json` and updates `leaderboard.json`/`leaderboard.md` per run.
- Generation writes `realism_stats.json` alongside `metadata.jsonl`.
- Use `--batch-api` for OpenAI Batch API cost savings.

## Documentation

- `docs/overview.md`
- `docs/cli.md`
- `docs/generation.md`
- `docs/model-configs.md`
- `docs/evaluation.md`
- `docs/metrics.md`
- `docs/gotchas.md`
- `docs/benchmark-protocol.md`
