"""Evaluation pipeline configuration schemas."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class InferenceBackend(str, Enum):
    """Supported inference backends."""

    # API backends
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    UPSTAGE = "upstage"
    # Local backends
    TRANSFORMERS = "transformers"
    PADDLEOCR = "paddleocr"
    SURYA = "surya"


class EvaluationMode(str, Enum):
    ALL = "all"
    INFERENCE_ONLY = "inference_only"
    EVALUATE_ONLY = "evaluate_only"


class ModelConfig(BaseModel):
    """Model configuration."""

    model_id: str = Field(description="Model identifier (e.g., 'gpt-5-mini', 'Qwen/Qwen3-VL-4B-Instruct')")
    backend: InferenceBackend = Field(description="Inference backend to use")

    # Generation parameters
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    timeout: int = Field(default=120, ge=1, description="Timeout in seconds")
    max_retries: int = Field(default=3, ge=0)

    # API-specific
    api_key: Optional[str] = Field(default=None, description="API key (or use env var)")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    rate_limit_rpm: Optional[int] = Field(default=None, description="Rate limit (requests per minute)")

    # Local model specific
    device: str = Field(default="cuda", description="Device for local inference")
    dtype: str = Field(default="bfloat16", description="Data type for model weights")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallelism size")
    max_model_len: Optional[int] = Field(default=None, description="Maximum model context length")

    model_config = {"protected_namespaces": ()}


class EvaluationConfig(BaseModel):
    """Evaluation pipeline configuration."""

    # Dataset configuration
    dataset_id: str = Field(description="HuggingFace dataset ID")
    split: str = Field(default="train", description="Dataset split")
    language: str = Field(default="ko", description="Dataset language code")

    # Model configuration
    model: ModelConfig = Field(description="Model configuration")

    # Evaluation parameters
    batch_size: int = Field(default=1, ge=1)
    max_samples: Optional[int] = Field(default=None, ge=1, description="Limit number of samples")
    prompt: Optional[str] = Field(default=None, description="Custom prompt override")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    batch_api: bool = Field(default=False, description="Use OpenAI Batch API for inference")
    batch_poll_seconds: int = Field(
        default=60,
        description="Polling interval for batch status",
        ge=5,
    )
    batch_timeout_seconds: int = Field(
        default=86400,
        description="Max wait time for batch completion",
        ge=60,
    )
    batch_completion_window: str = Field(
        default="24h",
        description="Batch completion window",
    )
    execution_mode: EvaluationMode = Field(
        default=EvaluationMode.ALL,
        description="Evaluation execution mode",
    )

    @model_validator(mode="after")
    def validate_batch_api(self) -> "EvaluationConfig":
        if self.batch_api and self.model.backend != InferenceBackend.OPENAI:
            raise ValueError("Batch API is only supported for OpenAI backend")
        if self.batch_api and self.batch_completion_window != "24h":
            raise ValueError("Only '24h' is supported for batch_completion_window")
        return self

    # Output configuration
    output_dir: str = Field(default="./evaluation_results")
    resume_from_checkpoint: bool = Field(default=True)

    # Column mappings
    image_column: str = Field(default="image")
    target_column: str = Field(default="typo_text")

    # Model config file path (optional)
    model_config_path: Optional[str] = Field(
        default=None, description="Path to model-specific config YAML"
    )

    model_config = {"protected_namespaces": ()}


DEFAULT_PROMPT = (
    "Extract the markdown content from this image. "
    "Return the raw markdown text exactly as shown."
)
