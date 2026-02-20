# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | document-parse-260128 |
| Backend | upstage |
| Dataset | junyeong-nero/synthetic-ocr-images-ja |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6508 |
| std_markdown_text_score | 0.3956 |
| avg_markdown_table_teds | 0.9000 |
| std_markdown_table_teds | 0.3000 |
| avg_markdown_formula_score | 0.4103 |
| std_markdown_formula_score | 0.4870 |
| avg_markdown_order_score | 0.6668 |
| std_markdown_order_score | 0.2929 |
| avg_markdown_overall_score | 0.6570 |
| std_markdown_overall_score | 0.2867 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.6508 |
| std_markdown_text_score | 0.3956 |
| avg_markdown_table_teds | 0.9000 |
| std_markdown_table_teds | 0.3000 |
| avg_markdown_formula_score | 0.4103 |
| std_markdown_formula_score | 0.4870 |
| avg_markdown_order_score | 0.6668 |
| std_markdown_order_score | 0.2929 |
| avg_markdown_overall_score | 0.6570 |
| std_markdown_overall_score | 0.2867 |


## Execution Summary

- **Total Samples**: 100
- **Successful**: 100
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 1523.37 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.6508 |
| Table | 0.9000 |
| Formula | 0.4103 |
| Order | 0.6668 |
| Overall | 0.6570 |
