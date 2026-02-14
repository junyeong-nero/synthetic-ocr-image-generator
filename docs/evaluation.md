# Model Evaluation Guide

The current unified evaluation pipeline is markdown-focused.

## Basic Usage

```bash
uv run main.py evaluate \
  --model-config configs/models/gpt-4o.yaml \
  --dataset "username/my-ocr-dataset" \
  --split train
```

Required arguments:

- `--model-config`: Model YAML file.
- `-d`, `--dataset`: Hugging Face dataset ID or local dataset path.

Common arguments:

- `-b`, `--backend`: Override backend from model config.
- `--split`: Dataset split (default: `train`).
- `--max-samples`: Evaluate only first N samples.
- `--seed`: Random seed for reproducibility.
- `--output-dir`: Output directory (default: `./evaluation_results`).
- `--report-format`: `json`, `markdown`, `html`, `all` (default: `all`).

Execution mode flags:

- `--inference-only`: Run inference and store `checkpoints.json`.
- `--evaluate-only`: Skip inference and evaluate from `checkpoints.json`.

Config override flags:

- `--batch-size`
- `--temperature`
- `--max-tokens`
- `--api-base`
- `--tensor-parallel`

## Pipeline Notes

- CLI entrypoint: `main.py`
- Orchestrator: `src/evaluation/pipeline.py`
- Runner/checkpointing: `src/evaluation/runner.py`
- Metrics/evaluator: `src/evaluation/strategies.py`
- Reports: `src/evaluation/report.py`

Important behavior:

- Evaluation format is fixed to `markdown` in the current pipeline.
- Prompt resolution priority is `CLI override > model config prompt(markdown) > default`.

## Batch API (OpenAI)

```bash
uv run main.py evaluate \
  --model-config configs/models/gpt-4o.yaml \
  --dataset "username/my-dataset" \
  --batch-api
```

Related flags:

- `--batch-poll-seconds` (default: `60`)
- `--batch-timeout-seconds` (default: `86400`)
- `--batch-completion-window` (default: `24h`)

## Reported Metrics (Markdown)

Core markdown block metrics:

- `avg_markdown_text_score`
- `avg_markdown_table_teds`
- `avg_markdown_formula_score`
- `avg_markdown_order_score`
- `avg_markdown_overall_score`

Quality metrics:

- `empty_count`, `empty_rate`
- `parse_fail_count`, `parse_fail_rate`

Representative metric for summary/leaderboard aggregation is `avg_markdown_overall_score`.

## Output Artifacts

Output directory contains:

- `report.json`
- `report.md`
- `report.html` (when requested)
- `protocol.json`
- `checkpoints.json`
- `model_summary.json`
- `leaderboard.json`
- `leaderboard.md`

## Script Wrappers

Recommended wrappers:

- Single model run: `scripts/evaluate/run.sh`
- Run all configs: `scripts/evaluate/run-all.sh`
- Leaderboard refresh: `scripts/evaluate/update-leaderboard.sh`
