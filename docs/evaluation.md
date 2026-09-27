# Model Evaluation Guide

The current unified evaluation pipeline is markdown-focused.

## Basic Usage

```bash
uv run main.py evaluate \
  --model-config configs/models/gpt-5-mini.yaml \
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
- `--output-dir`: Output directory (default: `./evaluation_result`).
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
  --model-config configs/models/gpt-5-mini.yaml \
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

## Synthetic vs Real Rank Correlation

To validate that the benchmark's model rankings agree with a real (non-synthetic)
reference benchmark, use `rank-correlation`:

```bash
uv run main.py rank-correlation \
  --synthetic evaluation_result/leaderboard.json \
  --real real_scores.csv \
  --language ko \
  --output rank_correlation_report.md
```

- Synthetic scores are read straight from this repo's evaluation outputs
  (`evaluation_result/leaderboard.json`, a single `report.json`, or a
  directory scanned recursively for `report.json` files) for a chosen
  metric (default `avg_markdown_overall_score`).
- Real scores come from a CSV (`model,score` columns) or JSON file supplied
  by the caller.
- Models are matched by a normalized id (org/path prefix stripped, folded to
  lowercase alphanumerics); an optional `--aliases` file covers names that
  do not normalize to the same id on both sides.
- Output is Spearman rho and Kendall tau with a bootstrap 95% CI over the
  matched models, plus a per-model rank table and any unmatched models on
  either side.

See `docs/cli.md` (`rank-correlation`) for the full flag reference. Implementation:
`src/evaluation/rank_correlation.py` (scoring, matching, correlation) and
`src/cli/rank_correlation.py` (CLI wiring).

## Script Wrappers

Recommended wrappers:

- Single model run: `scripts/evaluate/run.sh`
- Run all configs: `scripts/evaluate/run-all.sh`
- Leaderboard refresh: `scripts/evaluate/update-leaderboard.sh`

Single model wrapper example:

```bash
scripts/evaluate/run.sh gpt-5-mini -d "username/my-dataset" --language ko -n 200 --split test
```

Batch wrapper example:

```bash
scripts/evaluate/run-all.sh -d "username/my-dataset" --language ko -n 200 --split test
```

Wrapper option notes:

- `run.sh` uses `-m, --model-id` for model reference and `-n, --max-samples` for sample limit.
- `run.sh` supports `-l, --language` (default: `ko`) and forwards it to `main.py evaluate`.
- `run-all.sh` uses `-n, --max-samples` for sample limit.
- `run-all.sh` supports `-l, --language` (default: `ko`) and forwards it to each config run.
- `run-all.sh` still accepts `-m` for max samples as a deprecated alias.
