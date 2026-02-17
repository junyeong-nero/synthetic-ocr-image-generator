# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | nanonets/Nanonets-OCR2-3B |
| Backend | transformers |
| Dataset | junyeong-nero/synthetic-ocr-images-ko |
| Format | markdown |

## Metrics

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.8916 |
| std_markdown_text_score | 0.1121 |
| avg_markdown_table_teds | 0.9929 |
| std_markdown_table_teds | 0.0214 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 1.0000 |
| std_markdown_order_score | 0.0000 |
| avg_markdown_overall_score | 0.9711 |
| std_markdown_overall_score | 0.0273 |
| empty_count | 0.0000 |
| empty_rate | 0.0000 |
| parse_fail_count | 0.0000 |
| parse_fail_rate | 0.0000 |

## Metric Views

### Normalized

| Metric | Value |
|--------|-------|
| avg_markdown_text_score | 0.8916 |
| std_markdown_text_score | 0.1121 |
| avg_markdown_table_teds | 0.9929 |
| std_markdown_table_teds | 0.0214 |
| avg_markdown_formula_score | 1.0000 |
| std_markdown_formula_score | 0.0000 |
| avg_markdown_order_score | 1.0000 |
| std_markdown_order_score | 0.0000 |
| avg_markdown_overall_score | 0.9711 |
| std_markdown_overall_score | 0.0273 |


## Execution Summary

- **Total Samples**: 10
- **Successful**: 10
- **Failed**: 0
- **Empty Rate**: 0.0000
- **Parse Fail Rate**: 0.0000
- **Average Latency**: 9431.99 ms

## Markdown Block Scores

| Component | Value |
|-----------|-------|
| Text | 0.8916 |
| Table | 0.9929 |
| Formula | 1.0000 |
| Order | 1.0000 |
| Overall | 0.9711 |
