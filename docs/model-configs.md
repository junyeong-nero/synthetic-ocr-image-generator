# Model Configurations

Model YAML files in `configs/models/` define backend selection, inference parameters, and prompts.

## Minimal Example

```yaml
model_id: "gpt-5-mini"
backend: "openai"
dependency_group: "api"

temperature: 0.0
max_tokens: 4096
batch_size: 1
timeout: 120
max_retries: 3

prompt:
  prompt: "Convert the image content to clean markdown. Output markdown only."
```

## Supported Fields

Common fields loaded by `src/evaluation/model_config.py`:

- `model_id`
- `backend` (`openai`, `anthropic`, `google`, `upstage`, `transformers`, `paddleocr`, `surya`)
- `dependency_group`
- `temperature`, `max_tokens`, `top_p`, `batch_size`
- `timeout`, `max_retries`
- `api_base`, `rate_limit_rpm`
- `device`, `dtype`, `tensor_parallel_size`, `max_model_len`
- `prompt` (single markdown prompt config)

## Prompt Configuration

Current evaluation pipeline uses a single markdown prompt. Recommended structure:

```yaml
prompt:
  prompt: "..."
  system_prompt: "..."  # optional
```

## Creating a New Config

1. Copy `configs/models/_template.yaml` or an existing config.
2. Set `model_id`, `backend`, and optional `dependency_group`.
3. Add/adjust generation parameters.
4. Set at least `prompt.prompt`.
5. Run a small validation:

```bash
uv run main.py evaluate \
  --model-config configs/models/<your-config>.yaml \
  --dataset <dataset-id> \
  --max-samples 5
```

## List Available Configs

```bash
uv run main.py list-configs
```
