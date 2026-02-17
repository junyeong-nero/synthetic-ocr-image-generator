# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | ./weights/DotsOCR |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7656 |
| std_markdown_text_score | 0.2929 |
| avg_markdown_table_teds | 0.8897 |
| std_markdown_table_teds | 0.2982 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.8833 |
| std_markdown_order_score | 0.2363 |
| avg_markdown_overall_score | 0.8847 |
| std_markdown_overall_score | 0.1772 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7656 |
| std_markdown_text_score | 0.2929 |
| avg_markdown_table_teds | 0.8897 |
| std_markdown_table_teds | 0.2982 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.8833 |
| std_markdown_order_score | 0.2363 |
| avg_markdown_overall_score | 0.8847 |
| std_markdown_overall_score | 0.1772 |


## Execution Summary

- **Total Samples**: 10
- **Successful**: 10
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 2274752.92 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.7656 |
| Table | 0.8897 |
| Formula | 1.0000 |
| Order | 0.8833 |
| Overall | 0.8847 |
