# Overview

Synthetic OCR Image Generator and Benchmark is an end-to-end toolkit for dataset generation and model evaluation.

## Current Scope

The main CLI flow covers corpus preparation plus markdown-focused generation and evaluation:

- `corpus generate` creates reusable corpus text files with LLM-backed providers.
- `generate` creates markdown-rendered OCR images and metadata.
- `evaluate` computes markdown-oriented metrics and reports.

Some additional evaluator/generator modules exist in the codebase, but the unified CLI pipeline is centered on markdown.

## Architecture

### Generation (`src/pipeline.py`, `src/generator/`)

The generation pipeline follows a structured A/B/C phase model:

- **Phase A: Legacy Compatibility**: Traditional template methods (e.g., `readme`, `tutorial`) are now catalog-driven via YAML but remain backward compatible.
- **Phase B: Dynamic Blueprints**: Flexible document structures defined in `configs/generator/templates/*.yaml` using blueprint specifications (sections, blocks, complexity levels).
- **Phase C: Quality & Diversity**: Advanced controls including novelty guards, template family coverage targets, and style profiles to ensure high-quality, diverse datasets.

Core logic:
- Orchestrates markdown image creation via a multi-stage pipeline (config -> generation -> novelty check -> rendering -> upload).
- Loads fonts from `fonts/<lang>/`.
- Supports renderer selection (`pil` or `html2image`).
- Applies optional noise/blur and similarity-based substitutions.
- Uploads generated outputs to Hugging Face Hub with comprehensive metadata traces.

### Evaluation (`src/evaluation/`)

- Loads dataset and model configuration.
- Runs inference through selected backend.
- Supports normal mode, inference-only mode, and evaluate-only-from-checkpoint mode.
- Produces JSON/Markdown/HTML reports and leaderboard artifacts.

## Typical Workflow

1. Optionally prepare corpus assets with `uv run main.py corpus generate ...`, plus fonts and similarity DB assets.
2. Run `uv run main.py generate ...` (or `scripts/synthesize/generate.sh`).
3. Run `uv run main.py evaluate ...` (or `scripts/evaluate/run.sh`).
4. Compare reports with `uv run main.py compare ...`.

## Key Paths

- `main.py`: CLI commands, including `corpus generate`
- `configs/models/`: model YAML configs
- `scripts/synthesize/`: generation helpers
- `scripts/evaluate/`: evaluation helpers
- `docs/`: user-facing documentation
