"""Base utilities for vLLM model wrappers."""

from typing import Union

from evaluation.config import InferenceBackend, ModelConfig
from models.local.vllm_vlm import VLLMModel


def create_vllm_model_class(
    default_model_id: str,
    default_temperature: float = 0.0,
    default_max_tokens: int = 1024,
    default_top_p: float = 1.0,
):
    """
    Factory function to create a vLLM model wrapper class.

    Args:
        default_model_id: Default HuggingFace model ID.
        default_temperature: Default temperature for sampling.
        default_max_tokens: Default max tokens for generation.
        default_top_p: Default top_p for nucleus sampling.

    Returns:
        A VLLMModel subclass with the specified defaults.
    """

    class SpecializedVLLMModel(VLLMModel):
        DEFAULT_MODEL_ID = default_model_id

        def __init__(self, config: Union[ModelConfig, None] = None):
            if config is None:
                config = ModelConfig(
                    model_id=default_model_id,
                    backend=InferenceBackend.VLLM,
                    temperature=default_temperature,
                    max_tokens=default_max_tokens,
                    top_p=default_top_p,
                )
            super().__init__(config)

    return SpecializedVLLMModel
