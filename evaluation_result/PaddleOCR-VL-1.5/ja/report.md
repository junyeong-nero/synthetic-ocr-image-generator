# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | PaddlePaddle/PaddleOCR-VL-1.5 |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.5018 |
| std_markdown_text_score | 0.4388 |
| avg_markdown_table_teds | 0.4103 |
| std_markdown_table_teds | 0.4916 |
| avg_markdown_formula_score | 0.7144 |
| std_markdown_formula_score | 0.4226 |
| avg_markdown_order_score | 0.5400 |
| std_markdown_order_score | 0.3613 |
| avg_markdown_overall_score | 0.5416 |
| std_markdown_overall_score | 0.3467 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.5018 |
| std_markdown_text_score | 0.4388 |
| avg_markdown_table_teds | 0.4103 |
| std_markdown_table_teds | 0.4916 |
| avg_markdown_formula_score | 0.7144 |
| std_markdown_formula_score | 0.4226 |
| avg_markdown_order_score | 0.5400 |
| std_markdown_order_score | 0.3613 |
| avg_markdown_overall_score | 0.5416 |
| std_markdown_overall_score | 0.3467 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 152132.76 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.5018 |
| Table | 0.4103 |
| Formula | 0.7144 |
| Order | 0.5400 |
| Overall | 0.5416 |
