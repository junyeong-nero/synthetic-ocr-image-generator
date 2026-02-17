# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | document-parse-260128 |
| Backend | upstage |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.5463 |
| std_markdown_text_score | 0.3854 |
| avg_markdown_table_teds | 0.6435 |
| std_markdown_table_teds | 0.4526 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.7100 |
| std_markdown_order_score | 0.3961 |
| avg_markdown_overall_score | 0.7249 |
| std_markdown_overall_score | 0.2940 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.5463 |
| std_markdown_text_score | 0.3854 |
| avg_markdown_table_teds | 0.6435 |
| std_markdown_table_teds | 0.4526 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.7100 |
| std_markdown_order_score | 0.3961 |
| avg_markdown_overall_score | 0.7249 |
| std_markdown_overall_score | 0.2940 |


## Execution Summary

- **Total Samples**: 10
- **Successful**: 10
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 1252.06 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.5463 |
| Table | 0.6435 |
| Formula | 1.0000 |
| Order | 0.7100 |
| Overall | 0.7249 |
