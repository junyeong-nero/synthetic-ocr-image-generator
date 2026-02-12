# Model Evaluation Guide

The evaluation pipeline benchmarks OCR/VLM models across `sentence`, `table`, `document`, `markdown`, and `kie` subsets, then generates per-run reports and cross-run leaderboards.

## Basic Usage

Preferred command:

```bash
uv run main.py evaluate \
  --model-config configs/models/gpt-4o.yaml \
  --dataset "username/my-ocr-dataset"
```

Required arguments:
- `--model-config`: Model YAML (backend, model_id, prompts, subset overrides).
- `-d`, `--dataset`: Hugging Face dataset ID (or local dataset path).

Common arguments:
- `-s`, `--subset`: Comma-separated subset list (for example, `sentence,table,document`).
- `--split`: Dataset split (default: `train`).
- `--max-samples`: Evaluate only the first N samples.
- `--output-dir`: Output directory (default: `./evaluation_results`).
- `--report-format`: `json`, `markdown`, `html`, or `all` (default: `all`).

Config override arguments (override values from model config):
- `--batch-size`
- `--temperature`
- `--max-tokens`
- `--api-base`
- `--tensor-parallel`

## Evaluation Pipeline (End-to-End)

Implementation path:
- CLI orchestration: `main.py`
- Pipeline orchestrator: `src/evaluation/pipeline.py`
- Inference runner/checkpointing: `src/evaluation/runner.py`
- Format-specific evaluators: `src/evaluation/strategies.py`
- Report rendering: `src/evaluation/report.py`

Execution flow:
1. Parse CLI args in `main.py` and load model YAML via `ModelConfigLoader`.
2. For each selected subset, build `EvaluationConfig` + `ModelConfig` and create `EvaluationPipeline`.
3. Load dataset (`datasets.load_dataset`) with subset/split, then apply `--max-samples` if set.
4. Resolve prompt using strict priority:
   - CLI prompt override
   - model config subset prompt
   - model config format prompt
   - default format prompt
   - then append format output contract for `table`, `document`, `kie`
5. Extract ground truths by format evaluator (`EvaluatorRegistry`).
6. Run inference through `EvaluationRunner`:
   - normal path: batched `model.run(...)` / `run_async(...)`
   - retry empty predictions once
   - checkpoint state to `checkpoints.json` for resume
7. Compute metrics in two views:
   - `raw`: minimally normalized parsing path
   - `normalized`: canonicalized parsing path used as primary metrics
8. Add quality metrics:
   - `empty_count`, `empty_rate`
   - `parse_fail_count`, `parse_fail_rate`
9. Save reports and protocol snapshot, then aggregate per-subset representative scores into model summary + leaderboard.

## Batch API (OpenAI)

Enable OpenAI Batch API for large runs:

```bash
uv run main.py evaluate \
  --model-config configs/models/gpt-4o.yaml \
  --dataset "username/my-dataset" \
  --batch-api
```

Related controls:
- `--batch-poll-seconds` (default: `60`)
- `--batch-timeout-seconds` (default: `86400`)
- `--batch-completion-window` (currently only `24h` is accepted)

Notes:
- Batch mode is validated to OpenAI backend only.
- Batch metadata and API artifacts are written under `batch_<subset>/` inside the output directory.

## Metrics by Subset

Metrics are implemented in `src/evaluation/strategies.py` and `src/metrics/*`.

### Sentence
- `avg_cer`, `std_cer`, `min_cer`, `max_cer`
- `avg_wer`, `std_wer`

Source: `SentenceEvaluator` + `src/metrics/edit_distance.py` (`cer`, `wer`).

### Table
- `avg_teds`, `std_teds`
- `avg_cell_accuracy`, `std_cell_accuracy`
- `avg_structure_f1`, `std_structure_f1`

Source:
- `TableEvaluator`
- `src/metrics/table_document_metrics.py` (`evaluate_table`)
- `src/metrics/table_edit_distance.py` (`TEDS`, structure-only mode)

### Document
- `avg_layout_f1`, `std_layout_f1`
- `avg_reading_order`, `std_reading_order`
- `avg_kv_f1`, `std_kv_f1`
- `avg_text_score`, `std_text_score`
- `avg_table_teds`, `std_table_teds`
- `avg_formula_edit_distance`, `std_formula_edit_distance`
- `avg_text_table_formula_score`, `std_text_table_formula_score`
- legacy compatibility keys: `avg_text_table_score`, `avg_overall_f1`

Key formula:
- `text_table_formula_score = (text_score + table_teds + (1 - formula_edit_distance)) / 3`

Source:
- `DocumentEvaluator`
- `evaluate_document(...)` in `src/metrics/table_document_metrics.py`
- formula distance helper in `src/evaluation/strategies.py`

### Markdown
- `avg_cer`, `std_cer`, `min_cer`, `max_cer`
- `exact_match_rate`
- `normalized_match_rate`

Source: `MarkdownEvaluator` + `cer` from `src/metrics/edit_distance.py`.

### KIE
- `avg_entity_f1`, `std_entity_f1`
- `avg_entity_precision`, `avg_entity_recall`
- `avg_entity_accuracy`, `std_entity_accuracy`
- `avg_overall_f1`, `std_overall_f1`
- optional: `avg_item_f1`, `std_item_f1` when line items exist
- `per_field_metrics` (aggregated field-level accuracy)

Source:
- `KIEEvaluator`
- `evaluate_kie(...)`, `aggregate_kie_metrics(...)` in `src/metrics/kie_metrics.py`

## Representative Subset Score Keys

The summary/leaderboard path uses these representative metrics per subset:
- `sentence` -> `avg_cer`
- `table` -> `avg_teds`
- `document` -> `avg_text_table_formula_score`
- `markdown` -> `avg_cer`
- `kie` -> `avg_entity_f1`

In leaderboard normalization (`main.py`):
- lower-is-better metrics currently normalized as `1 - value` for `avg_cer`, `avg_wer`
- other representative metrics are treated as higher-is-better
- weighted averaging uses subset sample counts when available

## Output Artifacts

Per subset output directory contains:
- `report.json`: config, metrics, metric views, summary, per-sample results
- `report.md`: human-readable report
- `report.html`: HTML report (when `--report-format all` or `html`)
- `protocol.json`: protocol snapshot for reproducibility
- `checkpoints.json`: resumable progress state

Top-level output directory also contains:
- `model_summary.json`: append-only run summaries (across subsets/runs)
- `leaderboard.json`: ranked aggregate entries
- `leaderboard.md`: markdown leaderboard
