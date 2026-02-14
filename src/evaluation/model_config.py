"""Model-specific configuration loading and management."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

class PromptConfig(BaseModel):
    prompt: str = Field(description="The prompt template to use")
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt"
    )


class ModelSpecificConfig(BaseModel):
    """Model-specific configuration loaded from YAML."""

    # Model identification
    model_id: str = Field(description="Model identifier")
    backend: str = Field(description="Inference backend (openai, transformers, etc.)")
    dependency_group: Optional[str] = Field(
        default=None, description="uv dependency group for this model"
    )

    # Default generation parameters
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    batch_size: int = Field(default=1, ge=1)

    # Timeout and retry
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=3, ge=0)

    # API-specific
    api_base: Optional[str] = Field(default=None)
    rate_limit_rpm: Optional[int] = Field(default=None)

    # Local model specific
    device: str = Field(default="cuda")
    dtype: str = Field(default="bfloat16")
    tensor_parallel_size: int = Field(default=1, ge=1)
    max_model_len: Optional[int] = Field(default=None)

    prompt: PromptConfig = Field(description="Default markdown prompt configuration")

    model_config = {"protected_namespaces": ()}

    def get_prompt(self) -> PromptConfig:
        return self.prompt

    def get_batch_size(self) -> int:
        """Get batch size."""
        return self.batch_size

    def get_temperature(self) -> float:
        """Get temperature."""
        return self.temperature

    def get_max_tokens(self) -> int:
        """Get max tokens."""
        return self.max_tokens

    def get_model_id(self) -> str:
        """Get model ID."""
        return self.model_id


class ModelConfigLoader:
    """Loader for model-specific configuration files."""

    DEFAULT_CONFIG_DIRS = [
        Path("configs/models"),
        Path.home() / ".config" / "ocr-eval" / "models",
    ]

    def __init__(self, config_dirs: Optional[list[Path]] = None):
        self.config_dirs = config_dirs or self.DEFAULT_CONFIG_DIRS

    def _find_config_file(self, model_id: str) -> Optional[Path]:
        """Find config file for a model."""
        # Normalize model ID to filename (e.g., "Qwen/Qwen2-VL-7B" -> "qwen2-vl-7b.yaml")
        normalized = model_id.lower().replace("/", "_").replace(" ", "-")

        # Also try the last part of the path (e.g., "qwen2-vl-7b.yaml")
        short_name = model_id.split("/")[-1].lower().replace(" ", "-")

        candidates = [
            f"{normalized}.yaml",
            f"{normalized}.yml",
            f"{short_name}.yaml",
            f"{short_name}.yml",
        ]

        for config_dir in self.config_dirs:
            if not config_dir.exists():
                continue
            for candidate in candidates:
                config_path = config_dir / candidate
                if config_path.exists():
                    return config_path

        return None

    def load(self, model_id: str) -> Optional[ModelSpecificConfig]:
        """Load model-specific config from YAML file."""
        config_path = self._find_config_file(model_id)
        if not config_path:
            return None

        return self.load_from_path(config_path)

    def load_from_path(self, config_path: Path) -> ModelSpecificConfig:
        """Load model-specific config from a specific path."""
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._parse_config(data)

    def _parse_config(self, data: dict[str, Any]) -> ModelSpecificConfig:
        """Parse raw YAML data into ModelSpecificConfig."""
        if "prompt" in data:
            data["prompt"] = self._parse_prompt(data["prompt"])

        return ModelSpecificConfig(**data)

    def _parse_prompt(self, prompt_data: Any) -> PromptConfig:
        if isinstance(prompt_data, str):
            return PromptConfig(prompt=prompt_data)
        return PromptConfig(**prompt_data)

    def list_available_configs(self) -> list[str]:
        """List all available model configs."""
        configs = []
        for config_dir in self.config_dirs:
            if not config_dir.exists():
                continue
            for path in config_dir.glob("*.yaml"):
                configs.append(path.stem)
            for path in config_dir.glob("*.yml"):
                configs.append(path.stem)
        return sorted(set(configs))
