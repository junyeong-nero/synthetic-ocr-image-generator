# Synthetic OCR Image Generator & Benchmark

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive toolkit for generating synthetic OCR datasets and evaluating Vision-Language Models (VLMs) on OCR tasks.

## 📚 Documentation

-   [**Overview**](docs/overview.md): High-level introduction.
-   [**Generation Guide**](docs/generation.md): How to create datasets.
-   [**Evaluation Guide**](docs/evaluation.md): How to run benchmarks.
-   [**Model Configs**](docs/model-configs.md): Configuring models and dependencies.
-   [**Metrics**](docs/metrics.md): Understanding the scores (CER, TEDS, etc.).
-   [**CLI Reference**](docs/cli.md): Command usage.

## 🚀 Quick Start

### 1. Installation

This project uses `uv` for dependency management.

```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies
uv sync
```

### 2. Generate Data

Generate a small Korean sentence dataset:

```bash
uv run main.py generate \
    --repo-id my-ocr-dataset \
    --font-path fonts/ko/NanumGothic.ttf \
    --lang ko \
    --format sentence \
    --size 50
```

### 3. Evaluate a Model

Evaluate GPT-4o on the generated data:

```bash
# Ensure you have your API key set
export OPENAI_API_KEY=sk-...

uv run main.py evaluate \
    --model-config configs/models/gpt-4o.yaml \
    --dataset ./data/ko \
    --subset sentence
```

## 🏗️ Project Structure

-   `configs/models/`: Model YAML configurations.
-   `src/generator/`: Synthetic data generation logic.
-   `src/evaluation/`: Evaluation pipeline.
-   `scripts/`: Helper scripts for batch processing.

## 🤝 Contributing

See `AGENTS.md` files in source directories for development conventions.
