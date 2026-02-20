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
| avg_markdown_text_score | 0.7620 |
| std_markdown_text_score | 0.3447 |
| avg_markdown_table_teds | 0.9982 |
| std_markdown_table_teds | 0.0096 |
| avg_markdown_formula_score | 0.5257 |
| std_markdown_formula_score | 0.4956 |
| avg_markdown_order_score | 0.7667 |
| std_markdown_order_score | 0.2697 |
| avg_markdown_overall_score | 0.7631 |
| std_markdown_overall_score | 0.2543 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7620 |
| std_markdown_text_score | 0.3447 |
| avg_markdown_table_teds | 0.9982 |
| std_markdown_table_teds | 0.0096 |
| avg_markdown_formula_score | 0.5257 |
| std_markdown_formula_score | 0.4956 |
| avg_markdown_order_score | 0.7667 |
| std_markdown_order_score | 0.2697 |
| avg_markdown_overall_score | 0.7631 |
| std_markdown_overall_score | 0.2543 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 26343.07 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.7620 |
| Table | 0.9982 |
| Formula | 0.5257 |
| Order | 0.7667 |
| Overall | 0.7631 |
