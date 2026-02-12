# Synthetic OCR Image Generator & Benchmark

A comprehensive pipeline for generating synthetic OCR datasets and evaluating Vision Language Models (VLMs) on OCR tasks.

## 🚀 Overview

This project provides a robust toolkit for:
1.  **Synthetic Data Generation**: Create high-quality, diverse OCR images for various tasks (Sentences, Tables, Documents, Markdown, KIE).
2.  **Multilingual Support**: Generate data in multiple languages using a wide array of fonts.
3.  **VLM Evaluation**: Benchmark state-of-the-art OCR models and VLMs using standardized metrics and automated pipelines.
4.  **Hugging Face Integration**: Seamlessly upload generated datasets and evaluate models directly from the Hub.

## ✨ Key Features

- **Multiple Formats**:
    - `sentence`: Individual text lines with realistic typos and character similarity-based augmentations.
    - `table`: Complex tabular structures with varying rows and columns.
    - `document`: Full-page document layouts.
    - `markdown`: Content rendered from Markdown templates.
    - `kie`: Key Information Extraction (e.g., forms, receipts).
- **Extensive Model Support**: Integrated backends for OpenAI, Anthropic, Google Gemini, Hugging Face Transformers, PaddleOCR, and more.
- **Realistic Augmentation**: Character similarity database-driven typo generation for more robust model training and evaluation.
- **Automated Benchmarking**: Leaderboard generation and model comparison tools.

## 🛠 Installation

We recommend using `uv` for fast dependency management.

```bash
# Clone the repository
git clone https://github.com/your-repo/synthetic-ocr-image-generator.git
cd synthetic-ocr-image-generator

# Install dependencies (base)
uv sync

# Install specific model groups as needed
uv sync --group qwen2-vl
uv sync --group gpt-4o
```

## 📖 Usage

### 1. Generate Synthetic Data

Generate a dataset of 100 Korean sentences and upload to Hugging Face:

```bash
python main.py generate 
    --repo-id "your-username/my-ocr-dataset" 
    --font-path "fonts/ko/your-font.ttf" 
    --lang "ko" 
    --size 100 
    --format "sentence"
```

### 2. Evaluate a Model

Evaluate a model using a configuration file:

```bash
python main.py evaluate 
    --model-config configs/models/gpt-4o.yaml 
    --dataset "your-username/my-ocr-dataset" 
    --subset "sentence" 
    --split "train"
```

### 3. Compare Models

Compare results from multiple evaluation runs:

```bash
python main.py compare 
    evaluation_results/model_a/report.json 
    evaluation_results/model_b/report.json 
    -o comparison_results
```

## 📂 Project Structure

- `src/`: Core logic for generation and evaluation.
    - `generator/`: OCR image generation engines.
    - `evaluation/`: Inference and scoring pipelines.
    - `metrics/`: OCR-specific metric implementations (CER, WER, TEDS, etc.).
- `configs/`: Model and task configurations.
- `fonts/`: Multilingual font collections.
- `scripts/`: Helper scripts for automation.
- `docs/`: Detailed documentation.

## 📜 Documentation

For more detailed information, please refer to the [documentation](docs/overview.md):

- [Overview](docs/overview.md)
- [Data Generation](docs/generation.md)
- [Model Evaluation](docs/evaluation.md)
- [Model Configurations](docs/model-configs.md)
- [Metrics](docs/metrics.md)
- [CLI Reference](docs/cli.md)

## 🤝 Contributing

Contributions are welcome! Please check our [AGENTS.md](AGENTS.md) for architectural insights and contribution guidelines.

## 📄 License

[Insert License Information]
