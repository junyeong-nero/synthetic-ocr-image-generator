# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-VL-2B-Instruct |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6315 |
| std_markdown_text_score | 0.2994 |
| avg_markdown_table_teds | 0.9908 |
| std_markdown_table_teds | 0.0912 |
| avg_markdown_formula_score | 0.7567 |
| std_markdown_formula_score | 0.3978 |
| avg_markdown_order_score | 0.7847 |
| std_markdown_order_score | 0.2502 |
| avg_markdown_overall_score | 0.7909 |
| std_markdown_overall_score | 0.2247 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6315 |
| std_markdown_text_score | 0.2994 |
| avg_markdown_table_teds | 0.9908 |
| std_markdown_table_teds | 0.0912 |
| avg_markdown_formula_score | 0.7567 |
| std_markdown_formula_score | 0.3978 |
| avg_markdown_order_score | 0.7847 |
| std_markdown_order_score | 0.2502 |
| avg_markdown_overall_score | 0.7909 |
| std_markdown_overall_score | 0.2247 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 15593.18 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.6315 |
| Table | 0.9908 |
| Formula | 0.7567 |
| Order | 0.7847 |
| Overall | 0.7909 |
