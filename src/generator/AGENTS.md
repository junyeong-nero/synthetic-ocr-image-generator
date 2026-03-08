# PROJECT KNOWLEDGE BASE

## OVERVIEW
Synthetic dataset generators for sentence, table, document, markdown, and KIE formats.

## STRUCTURE
```
src/generator/
├── base.py                # Base generator utilities
├── registry.py            # Format name -> generator class
├── sentence_generator.py  # Sentence images + typos
├── table_generator.py     # Table layouts + cells
├── document_generator.py  # Invoice/report layouts
├── markdown_generator.py  # Markdown page rendering
└── kie_generator.py       # Forms/receipts for KIE
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new format | `registry.py` | Register name -> class |
| Markdown datasets | `src/pipeline.py` | MarkdownDatasetGenerator orchestration |
| Image effects | `effects.py` | Noise, blur, distortions |
| Metadata schema | generator modules | `metadata.jsonl` for uploads |

## CONVENTIONS
- Generators write images under `output_dir/<format>/` and emit `metadata.jsonl`.
- `file_name` in metadata should be a full path to the saved image file.
- Use language code from CLI (`--lang`) and keep metadata `format` fixed to `markdown`.

## ANTI-PATTERNS (THIS PROJECT)
- Do not add a new format without updating `registry.py` and `src/pipeline.py`.
