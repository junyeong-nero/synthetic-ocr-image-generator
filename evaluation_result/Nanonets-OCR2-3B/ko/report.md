# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | nanonets/Nanonets-OCR2-3B |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9025 |
| std_markdown_text_score | 0.2231 |
| avg_markdown_table_teds | 0.9988 |
| std_markdown_table_teds | 0.0089 |
| avg_markdown_formula_score | 0.8341 |
| std_markdown_formula_score | 0.3085 |
| avg_markdown_order_score | 0.9450 |
| std_markdown_order_score | 0.1668 |
| avg_markdown_overall_score | 0.9201 |
| std_markdown_overall_score | 0.1658 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9025 |
| std_markdown_text_score | 0.2231 |
| avg_markdown_table_teds | 0.9988 |
| std_markdown_table_teds | 0.0089 |
| avg_markdown_formula_score | 0.8341 |
| std_markdown_formula_score | 0.3085 |
| avg_markdown_order_score | 0.9450 |
| std_markdown_order_score | 0.1668 |
| avg_markdown_overall_score | 0.9201 |
| std_markdown_overall_score | 0.1658 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 16451.70 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.9025 |
| Table | 0.9988 |
| Formula | 0.8341 |
| Order | 0.9450 |
| Overall | 0.9201 |
