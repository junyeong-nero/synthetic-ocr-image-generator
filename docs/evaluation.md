# Evaluation

Evaluation is orchestrated by `src/evaluation/pipeline.py` and configured by `EvaluationConfig` and model YAML configs.

## Execution Flow

1. Load dataset (`datasets.load_dataset`) with `dataset_id`, `subset`, `split`.
2. Resolve prompt (subset prompt > format prompt > default prompt).
3. Run inference via `EvaluationRunner` with batching and checkpointing.
4. Compute format-specific metrics via `EvaluatorRegistry`.
5. Write reports and summary outputs.

## Prompts

Prompt resolution is handled in `EvaluationPipeline._get_prompt`:

- Subset prompt in model YAML if present
- Format prompt in model YAML if present
- Default prompt in `src/evaluation/config.py`

`EvaluationConfig.prompt` can override this when the pipeline is used programmatically.

## Subset and Format

`main.py` maps `--subset` to a format type:

- Exact match to a format name: `sentence`, `table`, `document`, `markdown`, `kie`
- Heuristic: subset name containing `table`, `document`, `markdown`, `kie`
- Default: `sentence`

When multiple subsets are provided, each subset gets a separate output directory with a safe name and hash.

## Reports

`ReportGenerator` produces:

- `report.json` (config, metrics, summary, per_sample_results)
- `report.md`
- `report.html`

The report format is controlled by `--report-format` (default `all`).

## Checkpointing

`EvaluationRunner` writes `checkpoint.json` in the output directory. If `resume_from_checkpoint` is true, it resumes incomplete runs.

## Model Summary

`main.py` writes `model_summary.json` in the output directory after evaluation:

- Appends a summary entry per run
- If the summary file is corrupt, it is renamed to `model_summary.corrupt.json`
- For multi-subset runs, `average_score` is computed over representative metrics

Representative metric mapping:

- `sentence` -> `avg_cer`
- `table` -> `avg_teds`
- `document` -> `avg_overall_f1`
- `markdown` -> `avg_cer`
- `kie` -> `avg_entity_f1`

## Compare

`main.py compare` loads multiple JSON reports and writes:

- `<output>.json`
- `<output>.md`

The comparator also prints a summary table to the console.
