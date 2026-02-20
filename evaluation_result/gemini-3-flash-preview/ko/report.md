# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | gemini-3-flash-preview |
| Backend | google |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6121 |
| std_markdown_text_score | 0.3696 |
| avg_markdown_table_teds | 0.7432 |
| std_markdown_table_teds | 0.4182 |
| avg_markdown_formula_score | 0.7089 |
| std_markdown_formula_score | 0.4253 |
| avg_markdown_order_score | 0.7242 |
| std_markdown_order_score | 0.3231 |
| avg_markdown_overall_score | 0.6971 |
| std_markdown_overall_score | 0.3229 |
| empty_count | 60.0000 |
| empty_rate | 0.6000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6121 |
| std_markdown_text_score | 0.3696 |
| avg_markdown_table_teds | 0.7432 |
| std_markdown_table_teds | 0.4182 |
| avg_markdown_formula_score | 0.7089 |
| std_markdown_formula_score | 0.4253 |
| avg_markdown_order_score | 0.7242 |
| std_markdown_order_score | 0.3231 |
| avg_markdown_overall_score | 0.6971 |
| std_markdown_overall_score | 0.3229 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 40
- **Failed**: 60
- **Empty Rate**: 0.6000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 8834.14 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.6121 |
| Table | 0.7432 |
| Formula | 0.7089 |
| Order | 0.7242 |
| Overall | 0.6971 |
