# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | lightonai/LightOnOCR-2-1B |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9549 |
| std_markdown_text_score | 0.0611 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 0.9437 |
| std_markdown_formula_score | 0.1397 |
| avg_markdown_order_score | 0.9960 |
| std_markdown_order_score | 0.0398 |
| avg_markdown_overall_score | 0.9737 |
| std_markdown_overall_score | 0.0482 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9549 |
| std_markdown_text_score | 0.0611 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 0.9437 |
| std_markdown_formula_score | 0.1397 |
| avg_markdown_order_score | 0.9960 |
| std_markdown_order_score | 0.0398 |
| avg_markdown_overall_score | 0.9737 |
| std_markdown_overall_score | 0.0482 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 13351.07 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.9549 |
| Table | 1.0000 |
| Formula | 0.9437 |
| Order | 0.9960 |
| Overall | 0.9737 |
