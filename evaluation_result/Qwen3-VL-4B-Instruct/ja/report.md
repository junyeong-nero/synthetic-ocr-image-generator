# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-VL-4B-Instruct |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7234 |
| std_markdown_text_score | 0.2174 |
| avg_markdown_table_teds | 0.9934 |
| std_markdown_table_teds | 0.0652 |
| avg_markdown_formula_score | 0.8876 |
| std_markdown_formula_score | 0.2519 |
| avg_markdown_order_score | 0.8518 |
| std_markdown_order_score | 0.1685 |
| avg_markdown_overall_score | 0.8641 |
| std_markdown_overall_score | 0.1332 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7234 |
| std_markdown_text_score | 0.2174 |
| avg_markdown_table_teds | 0.9934 |
| std_markdown_table_teds | 0.0652 |
| avg_markdown_formula_score | 0.8876 |
| std_markdown_formula_score | 0.2519 |
| avg_markdown_order_score | 0.8518 |
| std_markdown_order_score | 0.1685 |
| avg_markdown_overall_score | 0.8641 |
| std_markdown_overall_score | 0.1332 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 18996.09 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.7234 |
| Table | 0.9934 |
| Formula | 0.8876 |
| Order | 0.8518 |
| Overall | 0.8641 |
