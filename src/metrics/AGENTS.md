# PROJECT KNOWLEDGE BASE

## OVERVIEW
Evaluation metrics for OCR outputs across sentence, table, document, markdown, and KIE formats.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CER/WER | `edit_distance.py` | Text error rates |
| TEDS | `table_edit_distance.py` | Table structure similarity |
| Layout/KIE | `table_document_metrics.py` | Layout + key-value scoring |

## CONVENTIONS
- Metric outputs are dictionaries consumed by `src/evaluation/report.py`.
- Keep metric keys stable; reports and summaries expect existing names.

## ANTI-PATTERNS (THIS PROJECT)
- Do not change metric key names without updating report formatting.
