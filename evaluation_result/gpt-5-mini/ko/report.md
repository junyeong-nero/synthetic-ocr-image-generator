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
| avg_markdown_text_score | 0.5584 |
| std_markdown_text_score | 0.3648 |
| avg_markdown_table_teds | 0.7997 |
| std_markdown_table_teds | 0.3764 |
| avg_markdown_formula_score | 0.5420 |
| std_markdown_formula_score | 0.4926 |
| avg_markdown_order_score | 0.6996 |
| std_markdown_order_score | 0.3154 |
| avg_markdown_overall_score | 0.6499 |
| std_markdown_overall_score | 0.3277 |
| empty_count | 18.0000 |
| empty_rate | 0.1800 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.5584 |
| std_markdown_text_score | 0.3648 |
| avg_markdown_table_teds | 0.7997 |
| std_markdown_table_teds | 0.3764 |
| avg_markdown_formula_score | 0.5420 |
| std_markdown_formula_score | 0.4926 |
| avg_markdown_order_score | 0.6996 |
| std_markdown_order_score | 0.3154 |
| avg_markdown_overall_score | 0.6499 |
| std_markdown_overall_score | 0.3277 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 82
- **Failed**: 18
- **Empty Rate**: 0.1800
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 50749.87 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.5584 |
| Table | 0.7997 |
| Formula | 0.5420 |
| Order | 0.6996 |
| Overall | 0.6499 |
