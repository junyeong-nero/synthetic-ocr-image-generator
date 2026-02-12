# Overview

The **Synthetic OCR Image Generator & Benchmark** is an end-to-end framework designed to bridge the gap between synthetic data generation and large-scale evaluation of OCR-capable models.

## System Architecture

The project is divided into two main subsystems:

### 1. Generation Pipeline (`src/generator`)
This subsystem handles the creation of synthetic OCR images. It uses various specialized generators to produce different types of content:
- **Sentence Generator**: Uses Wikipedia corpora to generate realistic text lines. It utilizes a **Character Similarity DB** to introduce realistic typos (e.g., replacing '0' with 'O').
- **Table Generator**: Creates complex tables with varying styles and content.
- **Document/Markdown/KIE Generators**: Produce structured documents, markdown-rendered pages, and key-value pair layouts.

### 2. Evaluation Pipeline (`src/evaluation`)
This subsystem evaluates the performance of models (VLMs or traditional OCR engines) on the generated datasets:
- **Inference Backends**: Supports multiple backends including proprietary APIs (OpenAI, Anthropic, Gemini) and local models (Transformers, PaddleOCR).
- **Metric Computation**: Calculates industry-standard metrics like **CER** (Character Error Rate), **WER** (Word Error Rate), **TEDS** (Tree Edit Distance for Tables), and **F1-score** for KIE.
- **Reporting**: Generates comprehensive reports in JSON, Markdown, and HTML formats, including leaderboards and error analysis.

## Workflow

1.  **Corpus Collection**: Download and process text data for the target language.
2.  **Asset Preparation**: Gather fonts and templates.
3.  **Synthetic Generation**: Run the `generate` command to create images and metadata.
4.  **Dataset Hosting**: Upload to Hugging Face Hub for versioning and accessibility.
5.  **Model Configuration**: Define model parameters in YAML files.
6.  **Benchmarking**: Run the `evaluate` command against the hosted datasets.
7.  **Analysis**: Use the `compare` command to visualize differences between model versions or architectures.

## Supported Languages

The system is designed to be language-agnostic, provided that appropriate fonts and corpora are available. The current directory structure in `fonts/` indicates support for a wide range of languages including:
- English (`en`)
- Korean (`ko`)
- Japanese (`ja`)
- Chinese (`zh`)
- Hindi (`hi`)
- And many others...

## Key Technologies

- **Python 3.11+**
- **Pillow & OpenCV**: Image processing and rendering.
- **Transformers & Accelerate**: Local model inference.
- **Hugging Face Hub**: Dataset management.
- **PyYAML**: Configuration management.
