# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-28 21:08:25 KST
**Commit:** d3f36c1
**Branch:** main

## OVERVIEW
Python toolkit for synthetic OCR dataset generation and model evaluation, using uv dependency groups to isolate model backends.

## STRUCTURE
```
./
├── configs/models/        # Model YAML configs (prompts, backends, dependency_group)
├── src/                   # Core pipeline and packages
│   ├── generator/         # Synthetic data generation
│   ├── evaluation/        # Evaluation pipeline + reporting
│   ├── models/            # Model backends and registry
│   └── metrics/           # Evaluation metrics
├── scripts/               # CLI helpers (models/run.sh, models/test_all.sh, language scripts)
├── fonts/                 # Local font assets (ttf ignored by git)
├── data/                  # Local datasets (ignored by git)
└── test_results/          # Evaluation runs (generated)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CLI entry points | `main.py` | `generate`, `evaluate`, `compare`, list commands |
| Generation pipeline | `src/pipeline.py` | Mixed/format generation + upload hooks |
| Generators | `src/generator/` | Sentence/table/document/markdown/kie |
| Evaluation flow | `src/evaluation/` | Config, pipeline, runner, reporting |
| Model backends | `src/models/` | Registry + api/local/transformers |
| Metrics | `src/metrics/` | CER/WER/TEDS/Layout/KIE metrics |
| Model configs | `configs/models/` | YAML source of prompts + params |
| Run model groups | `scripts/models/run.sh` | Uses dependency_group |
| Batch eval | `scripts/models/test_all.sh` | Generates `test_results/` |

## CONVENTIONS
- Use `uv` dependency groups for model-specific installs; prefer `scripts/models/run.sh` to resolve `dependency_group`. Core groups: `generate`, `evaluate`, `api`.
- `configs/models/*.yaml` are source of truth for prompts and per-subset overrides.
- `main.py` injects `src` into `sys.path`; no console script entrypoint.

## ANTI-PATTERNS (THIS PROJECT)
- Do not commit generated artifacts in `data/`, `evaluation_results/`, or `test_results/`.
- Do not add a new `dependency_group` without updating `[dependency-groups]` and `[tool.uv.conflicts]` in `pyproject.toml`.

## UNIQUE STYLES
- Prompt selection is tiered: CLI override > subset prompt > format prompt > defaults.
- Scripts parse YAML keys with `grep "^model_id:"` and `grep "^dependency_group:"`.

## COMMANDS
```bash
uv sync
uv sync --extra eval
uv run main.py generate --help
uv run main.py evaluate --help
./scripts/models/run.sh <config_name> [args]
./scripts/models/test_all.sh [DATASET] [SUBSET] [MAX_SAMPLES]
```

## NOTES
- Fonts (`*.ttf`) and datasets are local assets; keep them out of git.
- Uploading to HF Hub requires `huggingface-cli login`.
