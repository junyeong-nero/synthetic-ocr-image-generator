# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | lightonai/LightOnOCR-2-1B |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9682 |
| std_markdown_text_score | 0.0333 |
| avg_markdown_table_teds | 0.9995 |
| std_markdown_table_teds | 0.0050 |
| avg_markdown_formula_score | 0.9458 |
| std_markdown_formula_score | 0.1233 |
| avg_markdown_order_score | 0.9971 |
| std_markdown_order_score | 0.0284 |
| avg_markdown_overall_score | 0.9777 |
| std_markdown_overall_score | 0.0388 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9682 |
| std_markdown_text_score | 0.0333 |
| avg_markdown_table_teds | 0.9995 |
| std_markdown_table_teds | 0.0050 |
| avg_markdown_formula_score | 0.9458 |
| std_markdown_formula_score | 0.1233 |
| avg_markdown_order_score | 0.9971 |
| std_markdown_order_score | 0.0284 |
| avg_markdown_overall_score | 0.9777 |
| std_markdown_overall_score | 0.0388 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 14239.35 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.9682 |
| Table | 0.9995 |
| Formula | 0.9458 |
| Order | 0.9971 |
| Overall | 0.9777 |
