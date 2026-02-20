# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | document-parse-260128 |
| Backend | upstage |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.5017 |
| std_markdown_text_score | 0.4000 |
| avg_markdown_table_teds | 0.7342 |
| std_markdown_table_teds | 0.4203 |
| avg_markdown_formula_score | 0.4534 |
| std_markdown_formula_score | 0.4933 |
| avg_markdown_order_score | 0.5810 |
| std_markdown_order_score | 0.3292 |
| avg_markdown_overall_score | 0.5676 |
| std_markdown_overall_score | 0.3178 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.5017 |
| std_markdown_text_score | 0.4000 |
| avg_markdown_table_teds | 0.7342 |
| std_markdown_table_teds | 0.4203 |
| avg_markdown_formula_score | 0.4534 |
| std_markdown_formula_score | 0.4933 |
| avg_markdown_order_score | 0.5810 |
| std_markdown_order_score | 0.3292 |
| avg_markdown_overall_score | 0.5676 |
| std_markdown_overall_score | 0.3178 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 1519.30 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.5017 |
| Table | 0.7342 |
| Formula | 0.4534 |
| Order | 0.5810 |
| Overall | 0.5676 |
