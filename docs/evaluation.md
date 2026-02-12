# Model Evaluation Guide

The evaluation pipeline benchmarks models on synthetic or real-world OCR datasets.

## Basic Usage

The primary command for evaluation is `python main.py evaluate`.

### Required Arguments
- `--model-config`: Path to a model-specific configuration YAML file (e.g., `configs/models/gpt-4o.yaml`).
- `-d`, `--dataset`: Hugging Face dataset ID or local path to the dataset.

### Common Arguments
- `-s`, `--subset`: Comma-separated list of dataset subsets to evaluate (e.g., `sentence,table`).
- `--split`: Dataset split to use (default: `train`).
- `--max-samples`: Limit the number of samples to evaluate (useful for quick testing).
- `--output-dir`: Directory to save results (default: `./evaluation_results`).

## Evaluation Workflow

1.  **Configure the Model**: Ensure your model's parameters (API keys, backend, temperature, etc.) are correctly set in `configs/models/<model>.yaml`.
2.  **Run Evaluation**:
    ```bash
    python main.py evaluate 
        --model-config configs/models/qwen2-vl-7b.yaml 
        --dataset "username/my-ocr-dataset" 
        --subset "sentence" 
        --max-samples 50
    ```
3.  **Review Results**: The pipeline will output metrics to the console and save detailed reports in the output directory.

## Inference Backends

Supported backends include:
- `openai`: For GPT-4o, etc. (Requires `OPENAI_API_KEY`)
- `anthropic`: For Claude 3.5 Sonnet, etc. (Requires `ANTHROPIC_API_KEY`)
- `google`: For Gemini models. (Requires `GOOGLE_API_KEY`)
- `transformers`: For local Hugging Face models (e.g., Qwen2-VL, LLaVA).
- `paddleocr`: For PaddleOCR engines.

## Batch Processing (OpenAI)

For large-scale evaluation on OpenAI models, use the `--batch-api` flag to significantly reduce costs and avoid rate limits:

```bash
python main.py evaluate 
    --model-config configs/models/gpt-4o.yaml 
    --dataset "username/my-dataset" 
    --batch-api
```

The script will submit the batch, poll for status, and download results once completed.

## Reports and Artifacts

The pipeline generates several artifacts in the `--output-dir`:
- `report.json`: Detailed metrics and raw model outputs for every sample.
- `report.md`: A human-readable summary of the evaluation.
- `protocol.json`: A snapshot of the evaluation configuration and environment.
- `model_summary.json`: An aggregated summary of all evaluations run for a specific model.
- `leaderboard.md`: An updated leaderboard incorporating the new results.
