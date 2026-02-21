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
| avg_markdown_text_score | 0.7141 |
| std_markdown_text_score | 0.2251 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 0.8860 |
| std_markdown_formula_score | 0.2684 |
| avg_markdown_order_score | 0.8557 |
| std_markdown_order_score | 0.1895 |
| avg_markdown_overall_score | 0.8639 |
| std_markdown_overall_score | 0.1489 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.7141 |
| std_markdown_text_score | 0.2251 |
| avg_markdown_table_teds | 1.0000 |
| std_markdown_table_teds | 0.0000 |
| avg_markdown_formula_score | 0.8860 |
| std_markdown_formula_score | 0.2684 |
| avg_markdown_order_score | 0.8557 |
| std_markdown_order_score | 0.1895 |
| avg_markdown_overall_score | 0.8639 |
| std_markdown_overall_score | 0.1489 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 17236.44 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.7141 |
| Table | 1.0000 |
| Formula | 0.8860 |
| Order | 0.8557 |
| Overall | 0.8639 |
