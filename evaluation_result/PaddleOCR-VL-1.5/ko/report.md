# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | PaddlePaddle/PaddleOCR-VL-1.5 |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.5020 |
| std_markdown_text_score | 0.4364 |
| avg_markdown_table_teds | 0.4400 |
| std_markdown_table_teds | 0.4964 |
| avg_markdown_formula_score | 0.6562 |
| std_markdown_formula_score | 0.4408 |
| avg_markdown_order_score | 0.5480 |
| std_markdown_order_score | 0.3564 |
| avg_markdown_overall_score | 0.5366 |
| std_markdown_overall_score | 0.3458 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.5020 |
| std_markdown_text_score | 0.4364 |
| avg_markdown_table_teds | 0.4400 |
| std_markdown_table_teds | 0.4964 |
| avg_markdown_formula_score | 0.6562 |
| std_markdown_formula_score | 0.4408 |
| avg_markdown_order_score | 0.5480 |
| std_markdown_order_score | 0.3564 |
| avg_markdown_overall_score | 0.5366 |
| std_markdown_overall_score | 0.3458 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 161440.46 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.5020 |
| Table | 0.4400 |
| Formula | 0.6562 |
| Order | 0.5480 |
| Overall | 0.5366 |
