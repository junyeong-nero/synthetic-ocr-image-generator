"""Model registry and factory for VLM models."""

from typing import Dict, Type, Union

from evaluation.config import InferenceBackend, ModelConfig
from models.base import Model, VLMModel


def get_model_class(backend: InferenceBackend) -> Type[VLMModel]:
    """
    Get model class for the specified backend.

    Args:
        backend: Inference backend type.

    Returns:
        Model class.

    Raises:
        ValueError: If backend is not supported.
    """
    # Lazy imports to avoid loading unnecessary dependencies
    if backend == InferenceBackend.OPENAI:
        from models.api.openai_vision import OpenAIVision

        return OpenAIVision

    elif backend == InferenceBackend.ANTHROPIC:
        from models.api.claude_vision import ClaudeVision

        return ClaudeVision

    elif backend == InferenceBackend.GOOGLE:
        from models.api.gemini_vision import GeminiVision

        return GeminiVision

    elif backend == InferenceBackend.TRANSFORMERS:
        from models.local.transformers_vlm import TransformersVLM

        return TransformersVLM

    elif backend == InferenceBackend.VLLM:
        from models.local.vllm_vlm import VLLMModel

        return VLLMModel

    elif backend == InferenceBackend.SGLANG:
        from models.local.sglang_vlm import SGLangModel

        return SGLangModel

    elif backend == InferenceBackend.OLLAMA:
        from models.local.ollama_vlm import OllamaModel

        return OllamaModel

    else:
        raise ValueError(f"Unknown backend: {backend}")


def create_model(config: ModelConfig) -> VLMModel:
    """
    Factory function to create a model from configuration.

    Args:
        config: Model configuration.

    Returns:
        Instantiated model.
    """
    model_class = get_model_class(config.backend)
    return model_class(config)


def create_model_from_args(
    model_id: str,
    backend: str,
    **kwargs,
) -> VLMModel:
    """
    Create a model from arguments.

    Args:
        model_id: Model identifier.
        backend: Backend name (string).
        **kwargs: Additional config options.

    Returns:
        Instantiated model.
    """
    config = ModelConfig(
        model_id=model_id,
        backend=InferenceBackend(backend),
        **kwargs,
    )
    return create_model(config)


# Backend display names for CLI help
BACKEND_DISPLAY_NAMES: Dict[InferenceBackend, str] = {
    InferenceBackend.OPENAI: "OpenAI API (GPT-4o, GPT-4V)",
    InferenceBackend.ANTHROPIC: "Anthropic API (Claude 3.5/4)",
    InferenceBackend.GOOGLE: "Google API (Gemini 1.5/2.0)",
    InferenceBackend.TRANSFORMERS: "HuggingFace Transformers",
    InferenceBackend.VLLM: "vLLM (high-performance local)",
    InferenceBackend.SGLANG: "SGLang (local or server)",
    InferenceBackend.OLLAMA: "Ollama (easy local)",
}


def list_backends() -> Dict[str, str]:
    """
    List available backends with descriptions.

    Returns:
        Dict mapping backend names to descriptions.
    """
    return {
        backend.value: description
        for backend, description in BACKEND_DISPLAY_NAMES.items()
    }
