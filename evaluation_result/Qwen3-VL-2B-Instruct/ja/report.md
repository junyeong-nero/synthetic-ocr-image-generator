# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-VL-2B-Instruct |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6448 |
| std_markdown_text_score | 0.2821 |
| avg_markdown_table_teds | 0.9927 |
| std_markdown_table_teds | 0.0727 |
| avg_markdown_formula_score | 0.7815 |
| std_markdown_formula_score | 0.3806 |
| avg_markdown_order_score | 0.7795 |
| std_markdown_order_score | 0.2357 |
| avg_markdown_overall_score | 0.7996 |
| std_markdown_overall_score | 0.2039 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6448 |
| std_markdown_text_score | 0.2821 |
| avg_markdown_table_teds | 0.9927 |
| std_markdown_table_teds | 0.0727 |
| avg_markdown_formula_score | 0.7815 |
| std_markdown_formula_score | 0.3806 |
| avg_markdown_order_score | 0.7795 |
| std_markdown_order_score | 0.2357 |
| avg_markdown_overall_score | 0.7996 |
| std_markdown_overall_score | 0.2039 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 15295.22 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.6448 |
| Table | 0.9927 |
| Formula | 0.7815 |
| Order | 0.7795 |
| Overall | 0.7996 |
