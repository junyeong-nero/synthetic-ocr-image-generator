# PROJECT KNOWLEDGE BASE

## OVERVIEW
Evaluation pipeline for running models against datasets and producing reports.

## STRUCTURE
```
src/evaluation/
├── config.py        # Core evaluation schemas and defaults
├── model_config.py  # YAML loader for single markdown prompt config
├── pipeline.py      # End-to-end evaluation flow
├── runner.py        # Batching + model execution
├── report.py        # Report rendering (json/md)
└── comparator.py    # Multi-run comparisons
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Config schema | `model_config.py` | `ModelSpecificConfig` + loader |
| Prompt selection | `pipeline.py` | CLI > model config prompt > defaults |
| Model creation | `src/models/registry.py` | Backend + model_id mapping |
| Output reports | `report.py` | JSON/Markdown output |

## CONVENTIONS
- `--model-config` YAML is the source of truth for prompts and overrides.
- Evaluation is markdown-only; keep `format` value fixed to `markdown`.
- Batch size and inference params come from config, overridden by CLI args when present.

## ANTI-PATTERNS (THIS PROJECT)
- Do not bypass `ModelConfigLoader` when loading YAML; keep validation centralized.
