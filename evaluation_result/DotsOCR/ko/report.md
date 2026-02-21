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
| avg_markdown_text_score | 0.9177 |
| std_markdown_text_score | 0.1484 |
| avg_markdown_table_teds | 0.9874 |
| std_markdown_table_teds | 0.1015 |
| avg_markdown_formula_score | 0.9004 |
| std_markdown_formula_score | 0.1953 |
| avg_markdown_order_score | 0.9800 |
| std_markdown_order_score | 0.0891 |
| avg_markdown_overall_score | 0.9464 |
| std_markdown_overall_score | 0.0920 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9177 |
| std_markdown_text_score | 0.1484 |
| avg_markdown_table_teds | 0.9874 |
| std_markdown_table_teds | 0.1015 |
| avg_markdown_formula_score | 0.9004 |
| std_markdown_formula_score | 0.1953 |
| avg_markdown_order_score | 0.9800 |
| std_markdown_order_score | 0.0891 |
| avg_markdown_overall_score | 0.9464 |
| std_markdown_overall_score | 0.0920 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 12338.65 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.9177 |
| Table | 0.9874 |
| Formula | 0.9004 |
| Order | 0.9800 |
| Overall | 0.9464 |
