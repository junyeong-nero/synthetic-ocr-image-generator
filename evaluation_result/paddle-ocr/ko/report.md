# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | paddleocr/paddle-ocr |
| Backend | paddleocr |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.1865 |
| std_markdown_text_score | 0.1784 |
| avg_markdown_table_teds | 0.4400 |
| std_markdown_table_teds | 0.4964 |
| avg_markdown_formula_score | 0.4300 |
| std_markdown_formula_score | 0.4951 |
| avg_markdown_order_score | 0.4598 |
| std_markdown_order_score | 0.2717 |
| avg_markdown_overall_score | 0.3791 |
| std_markdown_overall_score | 0.2663 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.1865 |
| std_markdown_text_score | 0.1784 |
| avg_markdown_table_teds | 0.4400 |
| std_markdown_table_teds | 0.4964 |
| avg_markdown_formula_score | 0.4300 |
| std_markdown_formula_score | 0.4951 |
| avg_markdown_order_score | 0.4598 |
| std_markdown_order_score | 0.2717 |
| avg_markdown_overall_score | 0.3791 |
| std_markdown_overall_score | 0.2663 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 5572.20 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.1865 |
| Table | 0.4400 |
| Formula | 0.4300 |
| Order | 0.4598 |
| Overall | 0.3791 |
