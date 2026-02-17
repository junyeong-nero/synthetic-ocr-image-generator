# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | gpt-5-mini |
| Backend | openai |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9629 |
| std_markdown_text_score | 0.0235 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 1.0000 |
| std_markdown_order_score | 0.0000 |
| avg_markdown_overall_score | 0.9907 |
| std_markdown_overall_score | 0.0059 |
| empty_count | 7.0000 |
| empty_rate | 0.7000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9629 |
| std_markdown_text_score | 0.0235 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 1.0000 |
| std_markdown_order_score | 0.0000 |
| avg_markdown_overall_score | 0.9907 |
| std_markdown_overall_score | 0.0059 |


## Execution Summary

- **Total Samples**: 10
- **Successful**: 3
- **Failed**: 7
- **Empty Rate**: 0.7000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 27398.98 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.9629 |
| Table | 1.0000 |
| Formula | 1.0000 |
| Order | 1.0000 |
| Overall | 0.9907 |
