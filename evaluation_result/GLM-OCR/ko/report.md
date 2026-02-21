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
| avg_markdown_text_score | 0.6606 |
| std_markdown_text_score | 0.3634 |
| avg_markdown_table_teds | 0.6544 |
| std_markdown_table_teds | 0.4730 |
| avg_markdown_formula_score | 0.9098 |
| std_markdown_formula_score | 0.2211 |
| avg_markdown_order_score | 0.7802 |
| std_markdown_order_score | 0.3195 |
| avg_markdown_overall_score | 0.7512 |
| std_markdown_overall_score | 0.2923 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6606 |
| std_markdown_text_score | 0.3634 |
| avg_markdown_table_teds | 0.6544 |
| std_markdown_table_teds | 0.4730 |
| avg_markdown_formula_score | 0.9098 |
| std_markdown_formula_score | 0.2211 |
| avg_markdown_order_score | 0.7802 |
| std_markdown_order_score | 0.3195 |
| avg_markdown_overall_score | 0.7512 |
| std_markdown_overall_score | 0.2923 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 12127.08 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.6606 |
| Table | 0.6544 |
| Formula | 0.9098 |
| Order | 0.7802 |
| Overall | 0.7512 |
