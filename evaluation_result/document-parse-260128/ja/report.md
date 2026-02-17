# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | document-parse-260128 |
| Backend | upstage |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7629 |
| std_markdown_text_score | 0.3244 |
| avg_markdown_table_teds | 0.8000 |
| std_markdown_table_teds | 0.4000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.8833 |
| std_markdown_order_score | 0.2363 |
| avg_markdown_overall_score | 0.8615 |
| std_markdown_overall_score | 0.2361 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7629 |
| std_markdown_text_score | 0.3244 |
| avg_markdown_table_teds | 0.8000 |
| std_markdown_table_teds | 0.4000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.8833 |
| std_markdown_order_score | 0.2363 |
| avg_markdown_overall_score | 0.8615 |
| std_markdown_overall_score | 0.2361 |


## Execution Summary

- **Total Samples**: 10
- **Successful**: 10
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 1258.60 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.7629 |
| Table | 0.8000 |
| Formula | 1.0000 |
| Order | 0.8833 |
| Overall | 0.8615 |
