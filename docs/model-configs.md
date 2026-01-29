# Model Configs

Model configs live in `configs/models/*.yaml`. Use `configs/models/_template.yaml` as the baseline. Configs are loaded by `ModelConfigLoader` and override defaults at runtime.

## File Naming

- Lowercase, model-specific name: `qwen2-vl-7b.yaml`
- For HF models, use the last path segment
- Full paths are normalized (`Qwen/Qwen2-VL` -> `qwen_qwen2-vl.yaml`)

## Required Keys

- `model_id`: model identifier used by the backend
- `backend`: inference backend (see supported backends below)

## Common Keys

```
temperature: 0.0
max_tokens: 4096
top_p: 1.0
batch_size: 1
timeout: 120
max_retries: 3
api_base: https://custom-api.example.com/v1
rate_limit_rpm: 500
device: cuda
dtype: bfloat16
tensor_parallel_size: 1
max_model_len: 32768
```

## Prompts

```
prompts:
  sentence:
    prompt: |
      Extract all text from the image verbatim.
  table:
    prompt: |
      Extract the table from this image as HTML.
```

## Subset Overrides

```
subsets:
  korean:
    batch_size: 4
    temperature: 0.1
    prompts:
      sentence:
        prompt: |
          Extract text for Korean subset.
```

## Prompt Selection Order

1. Subset prompt in YAML (`subsets.<name>.prompts`)
2. Format prompt in YAML (`prompts.<format>`)
3. Default prompt in `src/evaluation/config.py`

`EvaluationConfig.prompt` exists for programmatic overrides, but it is not exposed as a CLI flag in `main.py`.

## Supported Backends

`main.py` and `InferenceBackend` support:

- `openai`
- `anthropic`
- `google`
- `transformers`
- `paddleocr`

If the YAML uses other backend names, they will require code changes to `InferenceBackend` and the model registry.

## Dependency Groups

Optional `dependency_group` maps to `pyproject.toml` under `[dependency-groups]`.

```
dependency_group: qwen3-vl
```

Notes:

- `scripts/run_model.sh` and `scripts/test_all_models.sh` parse `model_id` and `dependency_group` with `grep`.
- Keep `model_id:` and `dependency_group:` at column 0 (no indentation).
- If you add a new group, update `[dependency-groups]` and `[tool.uv.conflicts]` in `pyproject.toml`.

## Config Search Paths

`ModelConfigLoader` searches:

- `configs/models`
- `~/.config/ocr-eval/models`
