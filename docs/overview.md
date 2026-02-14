# Overview

Synthetic OCR Image Generator and Benchmark is an end-to-end toolkit for dataset generation and model evaluation.

## Current Scope

The main CLI flow is currently markdown-focused:

- `generate` creates markdown-rendered OCR images and metadata.
- `evaluate` computes markdown-oriented metrics and reports.

Some additional evaluator/generator modules exist in the codebase, but the unified CLI pipeline is centered on markdown.

## Architecture

### Generation (`src/pipeline.py`, `src/generator/`)

- Orchestrates markdown image creation.
- Loads fonts from `fonts/<lang>/`.
- Supports renderer selection (`pil` or `html2image`).
- Applies optional noise/blur and similarity-based substitutions.
- Uploads generated outputs to Hugging Face Hub.

### Evaluation (`src/evaluation/`)

- Loads dataset and model configuration.
- Runs inference through selected backend.
- Supports normal mode, inference-only mode, and evaluate-only-from-checkpoint mode.
- Produces JSON/Markdown/HTML reports and leaderboard artifacts.

## Typical Workflow

1. Prepare fonts and optional corpus/similarity DB assets.
2. Run `uv run main.py generate ...` (or `scripts/dataset/generate.sh`).
3. Run `uv run main.py evaluate ...` (or `scripts/evaluate/run.sh`).
4. Compare reports with `uv run main.py compare ...`.

## Key Paths

- `main.py`: CLI commands
- `configs/models/`: model YAML configs
- `scripts/dataset/`: generation helpers
- `scripts/evaluate/`: evaluation helpers
- `docs/`: user-facing documentation
