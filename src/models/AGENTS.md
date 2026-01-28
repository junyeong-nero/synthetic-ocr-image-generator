# PROJECT KNOWLEDGE BASE

## OVERVIEW
Model backends for API and local inference, with a registry for specialized models.

## STRUCTURE
```
src/models/
├── registry.py      # Backend + model_id -> class mapping
├── base.py          # VLMModel base class
├── api/             # OpenAI/Anthropic/Gemini backends
├── local/           # Local engines (transformers, paddle)
└── transformers/    # Model-specific implementations
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add specialized model | `registry.py` | Update `SPECIALIZED_MODEL_REGISTRY` |
| Add new backend | `registry.py` + `api/` or `local/` | Wire `InferenceBackend` |
| Generic transformers | `local/transformers_vlm.py` | Default HF path |

## CONVENTIONS
- All models implement `run(prompts, images)` and optionally `run_async`.
- `ModelConfig` (from `evaluation.config`) is the only config object passed in.
- Specialized classes are selected by substring match on `model_id`.

## ANTI-PATTERNS (THIS PROJECT)
- Do not hardcode API keys in model classes; rely on env or config injection.
