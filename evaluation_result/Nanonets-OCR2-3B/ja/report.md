# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | nanonets/Nanonets-OCR2-3B |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9700 |
| std_markdown_text_score | 0.0140 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 0.8871 |
| std_markdown_formula_score | 0.2067 |
| avg_markdown_order_score | 0.9850 |
| std_markdown_order_score | 0.0853 |
| avg_markdown_overall_score | 0.9605 |
| std_markdown_overall_score | 0.0700 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9700 |
| std_markdown_text_score | 0.0140 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 0.8871 |
| std_markdown_formula_score | 0.2067 |
| avg_markdown_order_score | 0.9850 |
| std_markdown_order_score | 0.0853 |
| avg_markdown_overall_score | 0.9605 |
| std_markdown_overall_score | 0.0700 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 17337.79 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.9700 |
| Table | 1.0000 |
| Formula | 0.8871 |
| Order | 0.9850 |
| Overall | 0.9605 |
