# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | deepseek-ai/DeepSeek-OCR-2 |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9376 |
| std_markdown_text_score | 0.1511 |
| avg_markdown_table_teds | 0.9991 |
| std_markdown_table_teds | 0.0090 |
| avg_markdown_formula_score | 0.8719 |
| std_markdown_formula_score | 0.2482 |
| avg_markdown_order_score | 0.9760 |
| std_markdown_order_score | 0.1050 |
| avg_markdown_overall_score | 0.9461 |
| std_markdown_overall_score | 0.1163 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.9376 |
| std_markdown_text_score | 0.1511 |
| avg_markdown_table_teds | 0.9991 |
| std_markdown_table_teds | 0.0090 |
| avg_markdown_formula_score | 0.8719 |
| std_markdown_formula_score | 0.2482 |
| avg_markdown_order_score | 0.9760 |
| std_markdown_order_score | 0.1050 |
| avg_markdown_overall_score | 0.9461 |
| std_markdown_overall_score | 0.1163 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 19652.35 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.9376 |
| Table | 0.9991 |
| Formula | 0.8719 |
| Order | 0.9760 |
| Overall | 0.9461 |
