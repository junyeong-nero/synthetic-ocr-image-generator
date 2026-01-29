# Project Overview

This toolkit generates synthetic OCR datasets (sentence, table, document, markdown, KIE) and evaluates OCR or VLM models against those datasets. The CLI entry point is `main.py`.

## Architecture at a Glance

```
main.py
  generate -> src/pipeline.py -> src/generator/* -> data/<lang>/images_<format>
  evaluate -> src/evaluation/pipeline.py -> src/evaluation/runner.py -> evaluation_results/
  compare  -> src/evaluation/comparator.py -> comparison.json/.md
```

## Core Directories

- `src/generator/` - Format-specific generators and metadata writers
- `src/evaluation/` - Dataset loading, prompt selection, inference, metrics, reports
- `src/models/` - Backend registry and model implementations
- `src/metrics/` - Metric implementations (CER/WER/TEDS/Layout/KIE)
- `configs/models/` - Model YAML configs and prompts
- `scripts/` - Convenience shell scripts
- `docs/benchmark-protocol.md` - Standardized evaluation protocol

## Dependency Groups

- `generate`: generation pipeline only
- `evaluate`: evaluation pipeline + reporting
- `api`: API backends (OpenAI/Anthropic/Gemini)
- model groups (e.g., `qwen2-vl`, `paddle-ocr`) for specific backends

## Data Flow

Generation:

1. `main.py generate` builds corpus and similarity DB (sentence only) in `data/<lang>/`.
2. A generator produces images and metadata.jsonl.
3. `utils.upload_subset_to_hub` pushes a Hugging Face dataset subset.

Evaluation:

1. `main.py evaluate` loads a model config and dataset.
2. `EvaluationPipeline._resolve_prompt` selects a prompt (subset > format > default).
3. `EvaluationRunner` batches inference and writes `checkpoint.json`.
4. Evaluators compute metrics and reports are written to `evaluation_results/`.
5. `protocol.json` and `leaderboard.json`/`leaderboard.md` are updated per run.

## Dataset Schema (by Format)

All subsets include an `image` column when uploaded to Hugging Face. The remaining columns are derived from `metadata.jsonl`.

Sentence:
- `typo_text` (target column for evaluation)
- `original_text`
- rendering parameters (font size, blur, etc.)

Table:
- `html` (ground truth table HTML)
- `json` (table structure and cell metadata)
- `template`, `num_rows`, `num_cols`, `font_size`

Document:
- `ground_truth` (elements with `type`, `text`, `bounding_box`, `reading_order`)
- `template`, `elements_count`, `font_size`, `add_noise`, `add_blur`

Markdown:
- `markdown` (ground truth markdown string)
- `template`, `add_noise`, `add_blur`

KIE:
- `document_type`
- `ground_truth` (entities, line_items, raw_text)
- `entities` may be present if provided by dataset or model outputs

## Extension Points

- Add a new generator: implement a `BaseGenerator` subclass and register it in `src/generator/registry.py` and `src/pipeline.py`.
- Add a new evaluator: implement `BaseEvaluator` and register in `src/evaluation/strategies.py`.
- Add a new backend: extend `InferenceBackend` and wire `src/models/registry.py`.
