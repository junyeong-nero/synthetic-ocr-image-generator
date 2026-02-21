# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | zai-org/GLM-OCR |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7908 |
| std_markdown_text_score | 0.3519 |
| avg_markdown_table_teds | 0.7831 |
| std_markdown_table_teds | 0.4019 |
| avg_markdown_formula_score | 0.9085 |
| std_markdown_formula_score | 0.1878 |
| avg_markdown_order_score | 0.8582 |
| std_markdown_order_score | 0.2839 |
| avg_markdown_overall_score | 0.8351 |
| std_markdown_overall_score | 0.2667 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7908 |
| std_markdown_text_score | 0.3519 |
| avg_markdown_table_teds | 0.7831 |
| std_markdown_table_teds | 0.4019 |
| avg_markdown_formula_score | 0.9085 |
| std_markdown_formula_score | 0.1878 |
| avg_markdown_order_score | 0.8582 |
| std_markdown_order_score | 0.2839 |
| avg_markdown_overall_score | 0.8351 |
| std_markdown_overall_score | 0.2667 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 9581.03 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.7908 |
| Table | 0.7831 |
| Formula | 0.9085 |
| Order | 0.8582 |
| Overall | 0.8351 |
