# Project Overview

The **Synthetic OCR Image Generator and VLM Evaluation Pipeline** is a comprehensive Python toolkit designed to generate synthetic OCR datasets and evaluate Vision-Language Models (VLMs) on OCR tasks.

## Key Features

-   **Synthetic Data Generation**: Create realistic OCR images in various formats including sentences, tables, documents, markdown, and KIE (Key Information Extraction) forms.
-   **Model Evaluation**: A robust pipeline to evaluate VLMs using different backends (local Transformers, OpenAI, Anthropic, Google, PaddleOCR).
-   **Dependency Isolation**: Uses `uv` dependency groups to manage conflicting requirements for different model backends.
-   **Configurable Benchmarks**: flexible YAML configurations for defining model prompts, parameters, and subsets.
-   **Detailed Reporting**: Generates JSON and Markdown reports, including leaderboards and protocol snapshots.

## Project Structure

```
./
├── configs/models/        # Model YAML configs (prompts, backends, dependency_group)
├── src/                   # Core pipeline and packages
│   ├── generator/         # Synthetic data generation logic
│   ├── evaluation/        # Evaluation pipeline and reporting
│   ├── models/            # Model backends and registry
│   └── metrics/           # Evaluation metrics (CER, WER, TEDS, etc.)
├── scripts/               # CLI helpers and batch scripts
├── fonts/                 # Local font assets
├── data/                  # Generated datasets
└── test_results/          # Evaluation output and logs
```

## Core Components

### Generator
The generator module produces synthetic images with ground truth annotations. It supports:
-   **Formats**: Sentences, Tables, Documents, Markdown, KIE.
-   **Customization**: Configurable fonts, typo ratios, layout templates, and noise effects.

### Evaluator
The evaluation module runs models against datasets and computes metrics. It features:
-   **Backends**: Support for API-based models (GPT-4o, Claude 3.5, Gemini) and local models (Qwen2-VL, DeepSeek-VL, etc.).
-   **Metrics**: Character Error Rate (CER), Word Error Rate (WER), Tree Edit Distance (TEDS) for tables, and F1 scores for KIE.

### Configuration
Model configurations are defined in YAML files within `configs/models/`. These files control:
-   Inference parameters (temperature, max tokens).
-   Prompts for different tasks (subsets).
-   Dependency groups for environment isolation.

## Getting Started

Refer to the [CLI Reference](cli.md) for command usage, or explore the [Generation](generation.md) and [Evaluation](evaluation.md) guides.
