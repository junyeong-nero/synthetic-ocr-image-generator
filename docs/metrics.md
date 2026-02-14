# Metrics

The current evaluation pipeline reports markdown-oriented block metrics and quality metrics.

## Markdown Block Metrics

Computed by `MarkdownEvaluator` with `metrics.markdown_block_metrics`:

- `avg_markdown_text_score`
- `avg_markdown_table_teds`
- `avg_markdown_formula_score`
- `avg_markdown_order_score`
- `avg_markdown_overall_score`

Each metric also has a corresponding standard deviation key (`std_*`).

## Quality Metrics

Pipeline-level quality metrics from `src/evaluation/pipeline.py`:

- `empty_count`
- `empty_rate`
- `parse_fail_count`
- `parse_fail_rate`

## Representative Score

`main.py` uses `avg_markdown_overall_score` as the representative metric when writing model summaries and leaderboard entries.

## Leaderboard Normalization

Normalization in `main.py` converts lower-is-better metrics (`avg_cer`, `avg_wer`) using `1.0 - value`. Other metrics are treated as higher-is-better.

For the current markdown-focused flow, representative score tracking is based on `avg_markdown_overall_score`.

## Evaluator Scope

The evaluation pipeline uses a single markdown evaluator implementation in `src/evaluation/strategies.py`.
