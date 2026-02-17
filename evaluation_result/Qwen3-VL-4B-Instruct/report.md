# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-VL-4B-Instruct |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6629 |
| std_markdown_text_score | 0.2326 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.8667 |
| std_markdown_order_score | 0.1633 |
| avg_markdown_overall_score | 0.8824 |
| std_markdown_overall_score | 0.0980 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6629 |
| std_markdown_text_score | 0.2326 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.8667 |
| std_markdown_order_score | 0.1633 |
| avg_markdown_overall_score | 0.8824 |
| std_markdown_overall_score | 0.0980 |


## Execution Summary

- **Total Samples**: 10
- **Successful**: 10
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 11337.59 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.6629 |
| Table | 1.0000 |
| Formula | 1.0000 |
| Order | 0.8667 |
| Overall | 0.8824 |
