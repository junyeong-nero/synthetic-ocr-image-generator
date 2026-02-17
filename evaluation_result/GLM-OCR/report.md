# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | zai-org/GLM-OCR |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7178 |
| std_markdown_text_score | 0.2352 |
| avg_markdown_table_teds | 0.6000 |
| std_markdown_table_teds | 0.4899 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.8000 |
| std_markdown_order_score | 0.2449 |
| avg_markdown_overall_score | 0.7794 |
| std_markdown_overall_score | 0.2336 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7178 |
| std_markdown_text_score | 0.2352 |
| avg_markdown_table_teds | 0.6000 |
| std_markdown_table_teds | 0.4899 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.8000 |
| std_markdown_order_score | 0.2449 |
| avg_markdown_overall_score | 0.7794 |
| std_markdown_overall_score | 0.2336 |


## Execution Summary

- **Total Samples**: 10
- **Successful**: 10
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 118155.24 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.7178 |
| Table | 0.6000 |
| Formula | 1.0000 |
| Order | 0.8000 |
| Overall | 0.7794 |
