# Model Configurations

Model behavior in the evaluation pipeline is defined by YAML configuration files located in `configs/models/`. These files serve as the source of truth for prompts, inference parameters, and backend selection.

## Structure

A typical model configuration looks like this:

```yaml
model_id: "my-model-v1"
backend: "transformers"  # or "openai", "anthropic", "google", etc.
dependency_group: "my-model-group" # Optional: for uv dependency isolation

# Default inference parameters
temperature: 0.1
max_tokens: 1024

# Prompts by format
prompts:
  sentence: "Transcribe the text in this image exactly."
  table: "Convert the table in this image to HTML."

# Per-subset overrides
subsets:
  korean_handwriting:
    temperature: 0.2
    prompt: "Transcribe this Korean handwriting."
```

## Key Fields

| Field | Description |
| :--- | :--- |
| `model_id` | Unique identifier for the model (passed to the backend). |
| `backend` | The inference backend to use. Must match a registered backend. |
| `dependency_group` | The `uv` dependency group required to run this model (see `pyproject.toml`). |
| `prompts` | A dictionary mapping format types (e.g., `sentence`, `table`) to system or user prompts. |
| `subsets` | specific configurations for dataset subsets, overriding defaults. |

## Creating a New Config

1.  **Copy the Template**: Start by copying `configs/models/_template.yaml`.
2.  **Define Backend**: Set the `backend` and `model_id`.
3.  **Set Dependencies**: If the model requires specific libraries (e.g., a specific `transformers` version), add a group to `pyproject.toml` and reference it in `dependency_group`.
4.  **Tune Prompts**: Adjust prompts for each target format.

## Dependency Groups

This project uses `uv` dependency groups to manage conflicts. For example, `deepseek-ocr` might need a different environment than `qwen2-vl`.

-   **In `pyproject.toml`**: Define the group.
    ```toml
    [dependency-groups]
    deepseek-ocr = ["transformers==4.38.0", ...]
    ```
-   **In YAML**: Reference it.
    ```yaml
    dependency_group: "deepseek-ocr"
    ```
-   **Execution**: Use `scripts/run_model.sh`, which parses the YAML and runs `uv run --group deepseek-ocr ...`.
