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

# Create environment file for API models
cp .env.sample .env
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

### 3. Evaluate Models

You can evaluate models individually or in batches. The evaluation scripts automatically manage environment dependencies using `uv` groups.

#### Batch Evaluation
Use `test_all.sh` to run benchmarks across all configured models. This script automatically identifies and syncs the required `uv` dependency groups for each model (e.g., syncing the `glm-ocr` group when testing `configs/models/glm-ocr.yaml`).

```bash
# Test all models on a specific dataset and subset
# Usage: ./scripts/models/test_all.sh [DATASET] [SUBSET] [MAX_SAMPLES]
./scripts/models/test_all.sh junyeong-nero/synthetic-ocr-images-korean sentence 200
```

#### Single Model Evaluation
To evaluate a specific model like **GLM-OCR** with automatic dependency handling:

```bash
# Using the helper script (automatically syncs 'glm-ocr' uv group)
./scripts/models/run.sh glm-ocr \
    --dataset junyeong-nero/synthetic-ocr-images-ko \
    --subset sentence

# Or run directly via uv
uv run --group evaluate --group glm-ocr main.py evaluate \
    --model-config configs/models/glm-ocr.yaml \
    --dataset junyeong-nero/synthetic-ocr-images-ko \
    --subset sentence
```

## 🏗️ Project Structure

-   `configs/models/`: Model YAML configurations.
-   `src/generator/`: Synthetic data generation logic.
-   `src/evaluation/`: Evaluation pipeline.
-   `scripts/`: Helper scripts for batch processing.

## 🤝 Contributing

See `AGENTS.md` files in source directories for development conventions.
