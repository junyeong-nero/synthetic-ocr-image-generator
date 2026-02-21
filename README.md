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

### Formula Rendering System Dependencies

Markdown formula rendering uses `latex-to-image`, which requires a working XeLaTeX runtime.

- Verify XeLaTeX is installed: `xelatex --help`
- On macOS, install MacTeX so `xelatex` is available on your system

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
  --model-config configs/models/gpt-5-mini.yaml \
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

## Korean OCR Leaderboard (ko)

Latest consolidated Korean leaderboard is generated at `evaluation_result/leaderboard.md`.

Top 5 snapshot (`avg_markdown_overall_score`, higher is better):

| Rank | Model | Backend | Metric | Text | Table | Formula | Success/Total |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | lightonai/LightOnOCR-2-1B | transformers | 0.9737 | 0.9549 | 1.0000 | 0.9437 | 100/100 |
| 2 | ./weights/DotsOCR | transformers | 0.9464 | 0.9177 | 0.9874 | 0.9004 | 100/100 |
| 3 | deepseek-ai/DeepSeek-OCR-2 | transformers | 0.9461 | 0.9376 | 0.9991 | 0.8719 | 100/100 |
| 4 | nanonets/Nanonets-OCR2-3B | transformers | 0.9201 | 0.9025 | 0.9988 | 0.8341 | 100/100 |
| 5 | Qwen/Qwen3-VL-4B-Instruct | transformers | 0.8639 | 0.7141 | 1.0000 | 0.8860 | 100/100 |

## Japanese OCR Leaderboard (ja)

Latest consolidated Japanese leaderboard is generated at `evaluation_result/leaderboard.md`.

Top 5 snapshot (`avg_markdown_overall_score`, higher is better):

| Rank | Model | Backend | Metric | Text | Table | Formula | Success/Total |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | lightonai/LightOnOCR-2-1B | transformers | 0.9777 | 0.9682 | 0.9995 | 0.9458 | 100/100 |
| 2 | nanonets/Nanonets-OCR2-3B | transformers | 0.9605 | 0.9700 | 1.0000 | 0.8871 | 100/100 |
| 3 | ./weights/DotsOCR | transformers | 0.9288 | 0.8884 | 0.9923 | 0.9084 | 100/100 |
| 4 | deepseek-ai/DeepSeek-OCR-2 | transformers | 0.9141 | 0.8252 | 0.9847 | 0.8794 | 100/100 |
| 5 | Qwen/Qwen3-VL-4B-Instruct | transformers | 0.8641 | 0.7234 | 0.9934 | 0.8876 | 100/100 |

Refresh leaderboard files:

```bash
bash scripts/evaluate/update-leaderboard.sh
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
