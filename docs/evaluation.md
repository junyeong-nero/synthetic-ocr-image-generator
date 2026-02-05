# Evaluation Pipeline

The evaluation pipeline assesses the performance of OCR and VLM models against generated datasets. It handles inference, metric calculation, and report generation.

## Usage

To evaluate a model:

```bash
uv run main.py evaluate \
    --model-config configs/models/gpt-4o.yaml \
    --dataset ./data/ko \
    --subset sentence
```

## Workflow

1.  **Configuration**: Load the model-specific YAML config.
2.  **Dataset Loading**: Load the specified dataset (Hugging Face ID or local path).
3.  **Inference**: Run the model on the dataset samples using the configured backend.
4.  **Metric Calculation**: Compute relevant metrics (CER, WER, TEDS, etc.) based on the format.
5.  **Reporting**: Generate JSON and Markdown reports.

## Arguments

-   **`--model-config`**: Path to the YAML configuration file for the model.
-   **`--dataset`**: Path to the local dataset or Hugging Face Dataset ID.
-   **`--subset`**: The format/subset to evaluate (e.g., `sentence`, `table`). Can be comma-separated for multiple.
-   **`--split`**: Dataset split to use (default: `train`).
-   **`--backend`**: (Optional) Override the backend defined in the config.
-   **`--max-samples`**: Limit the number of samples for quick testing.
-   **`--batch-api`**: Use OpenAI Batch API (if applicable) for lower costs.
-   **`--output-dir`**: Directory to save results and reports.

## Batch Evaluation

To evaluate multiple models or run a full benchmark, use the provided scripts:

```bash
# Run a specific model config (handles dependency groups)
./scripts/models/run.sh configs/models/deepseek-ocr.yaml -d ./data/ko

# Run all models (caution: resource intensive)
./scripts/models/test_all.sh ./data/ko
```

## Reports

Evaluation produces several artifacts in the `output_dir`:

-   **`report.json` / `report.md`**: Detailed results including individual metrics.
-   **`protocol.json`**: A snapshot of the evaluation parameters and summary (see [Benchmark Protocol](benchmark-protocol.md)).
-   **`leaderboard.json` / `leaderboard.md`**: Rankings if multiple models/subsets are run.

## Dependency Management

Different local models often require conflicting library versions (e.g., different Transformers versions). The project uses `uv` dependency groups to handle this. The `scripts/run_model.sh` script automatically detects the `dependency_group` in the model YAML and runs the command in the correct environment.

```