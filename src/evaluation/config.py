"""Evaluation pipeline configuration schemas."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FormatType(str, Enum):
    """Supported evaluation format types."""

    SENTENCE = "sentence"
    TABLE = "table"
    DOCUMENT = "document"
    MARKDOWN = "markdown"
    KIE = "kie"


class InferenceBackend(str, Enum):
    """Supported inference backends."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    SGLANG = "sglang"
    OLLAMA = "ollama"


class ModelConfig(BaseModel):
    """Model configuration."""

    model_id: str = Field(description="Model identifier (e.g., 'gpt-4o', 'Qwen/Qwen2-VL-7B')")
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
    subset: str = Field(default="default", description="Dataset subset name")
    split: str = Field(default="test", description="Dataset split")
    format_type: FormatType = Field(description="Evaluation format type")

    # Model configuration
    model: ModelConfig = Field(description="Model configuration")

    # Evaluation parameters
    batch_size: int = Field(default=1, ge=1)
    max_samples: Optional[int] = Field(default=None, ge=1, description="Limit number of samples")
    prompt: Optional[str] = Field(default=None, description="Custom prompt override")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")

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


# Default prompts for each format type
DEFAULT_PROMPTS: dict[FormatType, str] = {
    FormatType.SENTENCE: (
        "Extract all text from the image verbatim, including typos, "
        "without translation or character modification."
    ),
    FormatType.TABLE: (
        "Extract the table from this image. Return the result as HTML table format."
    ),
    FormatType.DOCUMENT: (
        "Extract all text elements from this document image. "
        "Return as JSON with 'elements' array containing objects with "
        "'type', 'text', 'bounding_box', and 'reading_order' fields."
    ),
    FormatType.MARKDOWN: (
        "Extract the markdown content from this image. "
        "Return the raw markdown text exactly as shown."
    ),
    FormatType.KIE: (
        "Extract key information from this document image. "
        "Return as JSON with 'entities' object containing field names as keys "
        "and extracted values as values. For receipts/invoices, also include "
        "'line_items' array with objects containing 'name', 'quantity', "
        "'unit_price', and 'total_price' fields."
    ),
}
