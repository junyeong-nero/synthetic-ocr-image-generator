# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | deepseek-ai/DeepSeek-OCR-2 |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.8252 |
| std_markdown_text_score | 0.2303 |
| avg_markdown_table_teds | 0.9847 |
| std_markdown_table_teds | 0.0614 |
| avg_markdown_formula_score | 0.8794 |
| std_markdown_formula_score | 0.2341 |
| avg_markdown_order_score | 0.9672 |
| std_markdown_order_score | 0.0889 |
| avg_markdown_overall_score | 0.9141 |
| std_markdown_overall_score | 0.1150 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.8252 |
| std_markdown_text_score | 0.2303 |
| avg_markdown_table_teds | 0.9847 |
| std_markdown_table_teds | 0.0614 |
| avg_markdown_formula_score | 0.8794 |
| std_markdown_formula_score | 0.2341 |
| avg_markdown_order_score | 0.9672 |
| std_markdown_order_score | 0.0889 |
| avg_markdown_overall_score | 0.9141 |
| std_markdown_overall_score | 0.1150 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 21467.70 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.8252 |
| Table | 0.9847 |
| Formula | 0.8794 |
| Order | 0.9672 |
| Overall | 0.9141 |
