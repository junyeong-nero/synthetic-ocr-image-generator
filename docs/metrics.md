# Metrics

Metrics are computed by evaluators in `src/evaluation/strategies.py` and implementations in `src/metrics/`.

## Sentence Metrics

Computed in `SentenceEvaluator` using CER/WER:

- `avg_cer`, `std_cer`, `min_cer`, `max_cer`
- `avg_wer`, `std_wer`

## Table Metrics

Computed in `TableEvaluator` using `evaluate_table`:

- `avg_teds`, `std_teds`
- `avg_cell_accuracy`, `std_cell_accuracy`
- `avg_structure_f1`, `std_structure_f1`

Implementation details:

- TEDS is computed from HTML tables (`metrics/table_edit_distance.py`).
- Cell accuracy uses CER over matched cells (`metrics/table_document_metrics.py`).

## Document Metrics

Computed in `DocumentEvaluator` using `evaluate_document`:

- `avg_layout_f1`, `std_layout_f1`
- `avg_reading_order`, `std_reading_order`
- `avg_kv_f1`, `std_kv_f1`
- `avg_overall_f1`, `std_overall_f1`

Implementation details:

- Layout detection uses IoU and per-type F1.
- Reading order uses Kendall tau, Spearman rho, and adjacent pair accuracy.
- Key-value extraction computes F1 over matched pairs.

## Markdown Metrics

Computed in `MarkdownEvaluator`:

- `avg_cer`, `std_cer`, `min_cer`, `max_cer`
- `exact_match_rate`
- `normalized_match_rate`

Normalization collapses whitespace and empty lines before comparison.

## KIE Metrics

Computed in `KIEEvaluator` using `evaluate_kie` and `aggregate_kie_metrics`:

- `avg_entity_f1`, `std_entity_f1`
- `avg_entity_precision`, `avg_entity_recall`
- `avg_entity_accuracy`, `std_entity_accuracy`
- `avg_overall_f1`, `std_overall_f1`
- `avg_item_f1`, `std_item_f1` (if line items exist)
- `per_field_metrics` (per-field accuracy summaries)

KIE evaluation aggregates entity-level matches using normalized edit distance and optional line item matching.
