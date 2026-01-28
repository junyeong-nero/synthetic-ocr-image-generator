"""Model-specific configuration loading and management."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from evaluation.config import FormatType


class PromptConfig(BaseModel):
    """Prompt configuration for a specific format type."""

    prompt: str = Field(description="The prompt template to use")
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt"
    )


class SubsetConfig(BaseModel):
    """Configuration for a specific subset (format type)."""

    model_id: Optional[str] = Field(
        default=None, description="Override model ID for this subset"
    )
    prompts: dict[str, PromptConfig] = Field(
        default_factory=dict,
        description="Prompts keyed by format type (sentence, table, etc.)",
    )
    batch_size: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)


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

    # Default prompts for each format type
    prompts: dict[str, PromptConfig] = Field(
        default_factory=dict,
        description="Default prompts keyed by format type",
    )

    # Subset-specific overrides
    subsets: dict[str, SubsetConfig] = Field(
        default_factory=dict,
        description="Subset-specific configuration overrides",
    )

    model_config = {"protected_namespaces": ()}

    def get_prompt(
        self, format_type: FormatType, subset: Optional[str] = None
    ) -> Optional[PromptConfig]:
        """Get the prompt config for a format type, with optional subset override."""
        format_key = format_type.value

        # Check subset-specific prompt first
        if subset and subset in self.subsets:
            subset_config = self.subsets[subset]
            if format_key in subset_config.prompts:
                return subset_config.prompts[format_key]

        # Fall back to default prompts
        if format_key in self.prompts:
            return self.prompts[format_key]

        return None

    def get_batch_size(self, subset: Optional[str] = None) -> int:
        """Get batch size, with optional subset override."""
        if subset and subset in self.subsets:
            subset_batch = self.subsets[subset].batch_size
            if subset_batch is not None:
                return subset_batch
        return self.batch_size

    def get_temperature(self, subset: Optional[str] = None) -> float:
        """Get temperature, with optional subset override."""
        if subset and subset in self.subsets:
            subset_temp = self.subsets[subset].temperature
            if subset_temp is not None:
                return subset_temp
        return self.temperature

    def get_max_tokens(self, subset: Optional[str] = None) -> int:
        """Get max tokens, with optional subset override."""
        if subset and subset in self.subsets:
            subset_max = self.subsets[subset].max_tokens
            if subset_max is not None:
                return subset_max
        return self.max_tokens

    def get_model_id(self, subset: Optional[str] = None) -> str:
        """Get model ID, with optional subset override."""
        if subset and subset in self.subsets:
            subset_model_id = self.subsets[subset].model_id
            if subset_model_id is not None:
                return subset_model_id
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
        # Parse prompts
        if "prompts" in data:
            data["prompts"] = self._parse_prompts(data["prompts"])

        # Parse subsets
        if "subsets" in data:
            for subset_name, subset_data in data["subsets"].items():
                if "prompts" in subset_data:
                    subset_data["prompts"] = self._parse_prompts(subset_data["prompts"])

        return ModelSpecificConfig(**data)

    def _parse_prompts(
        self, prompts_data: dict[str, Any]
    ) -> dict[str, PromptConfig]:
        """Parse prompts section into PromptConfig objects."""
        result = {}
        for format_type, prompt_data in prompts_data.items():
            if isinstance(prompt_data, str):
                result[format_type] = PromptConfig(prompt=prompt_data)
            else:
                result[format_type] = PromptConfig(**prompt_data)
        return result

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
