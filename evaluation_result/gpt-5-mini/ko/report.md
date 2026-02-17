# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | gpt-5-mini |
| Backend | openai |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9315 |
| std_markdown_text_score | 0.0284 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 1.0000 |
| std_markdown_order_score | 0.0000 |
| avg_markdown_overall_score | 0.9829 |
| std_markdown_overall_score | 0.0071 |
| empty_count | 3.0000 |
| empty_rate | 0.3000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9315 |
| std_markdown_text_score | 0.0284 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 1.0000 |
| std_markdown_order_score | 0.0000 |
| avg_markdown_overall_score | 0.9829 |
| std_markdown_overall_score | 0.0071 |


## Execution Summary

- **Total Samples**: 10
- **Successful**: 7
- **Failed**: 3
- **Empty Rate**: 0.3000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 15170.16 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.9315 |
| Table | 1.0000 |
| Formula | 1.0000 |
| Order | 1.0000 |
| Overall | 0.9829 |
