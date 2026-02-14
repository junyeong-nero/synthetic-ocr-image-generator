# Synthetic OCR Image Generator and Benchmark

Synthetic OCR dataset generation and model benchmarking toolkit with a markdown-first pipeline.

## Overview

This repository provides two core workflows:

1. Generate synthetic OCR images and metadata, then upload datasets to Hugging Face Hub.
2. Evaluate OCR/VLM models against those datasets and produce reproducible reports.

The current unified CLI flow is markdown-focused for both generation and evaluation.

## Key Features

- Markdown OCR dataset generation with configurable rendering, noise/blur, and typo-like character substitutions.
- Character similarity database tooling for realistic substitutions.
- Evaluation pipeline with model config YAMLs, backend overrides, batch API support, and checkpoint-based resume.
- Report generation in JSON/Markdown/HTML plus protocol snapshots and leaderboard files.

## Installation

Use `uv` for dependency management.

```bash
git clone https://github.com/your-repo/synthetic-ocr-image-generator.git
cd synthetic-ocr-image-generator

uv sync
```

Install extra dependency groups only when needed (for example, model-specific backends).

## Quick Start

1) Generate a dataset

```bash
uv run main.py generate \
  --repo-id "your-username/my-ocr-dataset" \
  --lang "ko" \
  --size 100
```

2) Evaluate a model config

```bash
uv run main.py evaluate \
  --model-config configs/models/gpt-4o.yaml \
  --dataset "your-username/my-ocr-dataset" \
  --split train
```

3) Compare evaluation reports

```bash
uv run main.py compare \
  evaluation_result/model_a/report.json \
  evaluation_result/model_b/report.json \
  -o comparison_results
```

## Recommended Script Wrappers

- Dataset generation wrapper: `scripts/dataset/generate.sh`
- Evaluation wrapper with dependency-group handling: `scripts/evaluate/run.sh`
- Batch evaluation for all configs: `scripts/evaluate/run-all.sh`

## Project Structure

- `main.py`: CLI entrypoint (`generate`, `evaluate`, `compare`, list commands)
- `src/pipeline.py`: generation orchestration
- `src/generator/`: image generation and rendering utilities
- `src/evaluation/`: evaluation orchestration, runner, reports
- `src/metrics/`: metric implementations
- `configs/models/`: model config YAML files
- `scripts/`: automation helpers for dataset generation and evaluation

## Documentation

- `docs/overview.md`
- `docs/generation.md`
- `docs/evaluation.md`
- `docs/model-configs.md`
- `docs/metrics.md`
- `docs/benchmark-protocol.md`
- `docs/cli.md`

## Contributing

Contributions are welcome. See `AGENTS.md` for repository conventions.
