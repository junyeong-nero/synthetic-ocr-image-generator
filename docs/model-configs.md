# Model Configurations

Model configurations are defined in YAML files located in `configs/models/`. These files control how the evaluation pipeline interacts with different models.

## Config Structure

A typical configuration file (e.g., `configs/models/qwen2-vl-7b.yaml`) looks like this:

```yaml
# Model identification
model_id: "Qwen/Qwen2-VL-7B-Instruct"
backend: "transformers"

# Dependency group for 'uv'
dependency_group: "qwen2-vl"

# Model parameters
temperature: 0.0
max_tokens: 1024
tensor_parallel_size: 1

# Resource limits
rate_limit_rpm: 60
timeout: 300
max_retries: 3

# Execution environment
device: "cuda"
dtype: "bfloat16"

# Subset-specific overrides (optional)
subsets:
  table:
    max_tokens: 2048
    temperature: 0.2
  document:
    batch_size: 2
```

## Key Fields

- `model_id`: The identifier used by the backend (e.g., Hugging Face ID or API model name).
- `backend`: The inference engine to use (`openai`, `anthropic`, `google`, `transformers`, `paddleocr`).
- `dependency_group`: The `uv` dependency group required for this model (as defined in `pyproject.toml`).
- `temperature`: Sampling temperature for the model.
- `max_tokens`: Maximum number of tokens to generate.
- `subsets`: (Optional) Allows overriding parameters for specific dataset formats (e.g., higher `max_tokens` for tables).

## Creating a New Config

1.  Copy the `configs/models/_template.yaml` (if available) or an existing config.
2.  Update the `model_id` and `backend`.
3.  Add any necessary dependency groups to `pyproject.toml` if they don't exist.
4.  Test the configuration using `python main.py evaluate` with a small number of samples (`--max-samples 5`).

## Listing Available Configs

You can list all registered configurations using:

```bash
python main.py list-configs
```
