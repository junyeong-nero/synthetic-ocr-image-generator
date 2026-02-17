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
| avg_markdown_text_score | 0.3837 |
| std_markdown_text_score | 0.3631 |
| avg_markdown_table_teds | 0.1000 |
| std_markdown_table_teds | 0.3000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.4000 |
| std_markdown_order_score | 0.2708 |
| avg_markdown_overall_score | 0.4709 |
| std_markdown_overall_score | 0.2055 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.3837 |
| std_markdown_text_score | 0.3631 |
| avg_markdown_table_teds | 0.1000 |
| std_markdown_table_teds | 0.3000 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 0.4000 |
| std_markdown_order_score | 0.2708 |
| avg_markdown_overall_score | 0.4709 |
| std_markdown_overall_score | 0.2055 |


## Execution Summary

- **Total Samples**: 10
- **Successful**: 10
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 18435.18 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.3837 |
| Table | 0.1000 |
| Formula | 1.0000 |
| Order | 0.4000 |
| Overall | 0.4709 |
