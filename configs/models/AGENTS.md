# PROJECT KNOWLEDGE BASE

## OVERVIEW
Model YAML configs define backends, prompts, and per-subset overrides for evaluation.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| New model config | `configs/models/_template.yaml` | Copy and customize |
| Config loader | `src/evaluation/model_config.py` | Schema + parser |
| CLI usage | `main.py` | `--model-config` argument |

## CONVENTIONS
- File names are lowercase, model-specific (e.g., `qwen2-vl-7b.yaml`).
- Top-level `model_id` and `dependency_group` must be at column 0 for scripts to parse.
- `prompts` define defaults by format; `subsets` override per dataset subset.
- If a model needs isolated deps, add a matching `dependency_group` in `pyproject.toml`.

## ANTI-PATTERNS (THIS PROJECT)
- Do not indent or rename `model_id` / `dependency_group` keys; bash scripts rely on `grep`.
- Do not introduce a new dependency group without adding to `[tool.uv.conflicts]`.
