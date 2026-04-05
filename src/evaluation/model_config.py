"""Model-specific configuration loading and management."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

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

    prompt: PromptConfig = Field(description="Default markdown prompt configuration")

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

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

    def _iter_config_paths(self, *, include_templates: bool = True) -> list[Path]:
        config_paths: list[Path] = []
        for config_dir in self.config_dirs:
            if not config_dir.exists():
                continue
            for pattern in ("*.yaml", "*.yml"):
                for path in sorted(config_dir.glob(pattern)):
                    if not include_templates and path.stem.startswith("_"):
                        continue
                    config_paths.append(path)
        return config_paths

    def _load_raw_yaml(self, config_path: Path) -> dict[str, Any]:
        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}

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
        data = self._load_raw_yaml(config_path)
        return self._parse_config(data)

    def resolve_config_path(self, model_ref: str) -> Optional[Path]:
        candidate = Path(model_ref)
        if candidate.exists():
            return candidate

        for config_dir in self.config_dirs:
            for suffix in (".yaml", ".yml"):
                by_name = config_dir / f"{model_ref}{suffix}"
                if by_name.exists() and not by_name.stem.startswith("_"):
                    return by_name

        for config_path in self._iter_config_paths(include_templates=False):
            raw_data = self._load_raw_yaml(config_path)
            if str(raw_data.get("model_id") or "").strip() == model_ref:
                return config_path

        return None

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
        return sorted({path.stem for path in self._iter_config_paths()})

    def list_public_config_paths(self) -> list[Path]:
        return self._iter_config_paths(include_templates=False)
