"""Model registry and factory for VLM models."""

from typing import Dict, Optional, Type

from evaluation.config import InferenceBackend, ModelConfig
from models.base import VLMModel


# Specialized model registry: maps (model_id_pattern, backend) to (module_path, class_name)
SPECIALIZED_MODEL_REGISTRY: list[tuple[str, InferenceBackend, str, str]] = [
    # Transformers backend
    ("stepfun-ai/GOT-OCR", InferenceBackend.TRANSFORMERS, "models.transformers.got_ocr", "GotOCR"),
    ("GOT-OCR", InferenceBackend.TRANSFORMERS, "models.transformers.got_ocr", "GotOCR"),
    ("deepseek-ai/DeepSeek-OCR-2", InferenceBackend.TRANSFORMERS, "models.transformers.deepseek_ocr2", "DeepSeekOCR2"),
    ("DeepSeek-OCR-2", InferenceBackend.TRANSFORMERS, "models.transformers.deepseek_ocr2", "DeepSeekOCR2"),
    ("deepseek-ai/DeepSeek-OCR", InferenceBackend.TRANSFORMERS, "models.transformers.deepseek_ocr", "DeepSeekOCR"),
    ("DeepSeek-OCR", InferenceBackend.TRANSFORMERS, "models.transformers.deepseek_ocr", "DeepSeekOCR"),
    ("Qwen/Qwen3-VL", InferenceBackend.TRANSFORMERS, "models.transformers.qwen3_vl", "Qwen3VL"),
    ("Qwen3-VL", InferenceBackend.TRANSFORMERS, "models.transformers.qwen3_vl", "Qwen3VL"),
    ("NCSOFT/VARCO-VISION", InferenceBackend.TRANSFORMERS, "models.transformers.varco_ocr", "VarcoOCR"),
    ("VARCO-VISION", InferenceBackend.TRANSFORMERS, "models.transformers.varco_ocr", "VarcoOCR"),
    ("tencent/HunyuanOCR", InferenceBackend.TRANSFORMERS, "models.transformers.hunyuan_ocr", "HunYuanOCR"),
    ("HunyuanOCR", InferenceBackend.TRANSFORMERS, "models.transformers.hunyuan_ocr", "HunYuanOCR"),
    ("lightonai/LightOnOCR-2", InferenceBackend.TRANSFORMERS, "models.transformers.light_on_ocr2", "LightOnOCR2"),
    ("LightOnOCR-2", InferenceBackend.TRANSFORMERS, "models.transformers.light_on_ocr2", "LightOnOCR2"),
    ("PaddlePaddle/PaddleOCR", InferenceBackend.TRANSFORMERS, "models.transformers.paddle_ocr", "PaddleOCR"),
    ("paddleocr", InferenceBackend.PADDLEOCR, "models.local.paddle_ocr_engine", "PaddleOCREngine"),
    ("opendatalab/PDF-Extract-Kit-1.0", InferenceBackend.TRANSFORMERS, "models.transformers.paddle_ocr", "PaddleOCR"),
    ("google/gemma-3", InferenceBackend.TRANSFORMERS, "models.transformers.gemma3_4b_it", "Gemma3_4B_IT"),
]

BACKEND_DISPLAY_NAMES: Dict[InferenceBackend, str] = {
    InferenceBackend.OPENAI: "OpenAI API (GPT-4o, GPT-4V)",
    InferenceBackend.ANTHROPIC: "Anthropic API (Claude 3.5/4)",
    InferenceBackend.GOOGLE: "Google API (Gemini 1.5/2.0)",
    InferenceBackend.TRANSFORMERS: "HuggingFace Transformers",
    InferenceBackend.PADDLEOCR: "PaddleOCR",
}


def get_specialized_model_class(
    model_id: str, backend: InferenceBackend
) -> Optional[Type[VLMModel]]:
    """
    Get specialized model class for the given model ID and backend.

    Args:
        model_id: Model identifier.
        backend: Inference backend type.

    Returns:
        Specialized model class if found, None otherwise.
    """
    from importlib import import_module

    for pattern, reg_backend, module_path, class_name in SPECIALIZED_MODEL_REGISTRY:
        if pattern in model_id and reg_backend == backend:
            module = import_module(module_path)
            return getattr(module, class_name)
    return None


def get_model_class(backend: InferenceBackend, model_id: str = "") -> Type[VLMModel]:
    """
    Get model class for the specified backend and model ID.

    Args:
        backend: Inference backend type.
        model_id: Model identifier for specialized model lookup.

    Returns:
        Model class.

    Raises:
        ValueError: If backend is not supported.
    """
    if model_id and backend == InferenceBackend.TRANSFORMERS:
        specialized_class = get_specialized_model_class(model_id, backend)
        if specialized_class is not None:
            return specialized_class

    if backend == InferenceBackend.OPENAI:
        from models.api.openai_vision import OpenAIVision
        return OpenAIVision

    if backend == InferenceBackend.ANTHROPIC:
        from models.api.claude_vision import ClaudeVision
        return ClaudeVision

    if backend == InferenceBackend.GOOGLE:
        from models.api.gemini_vision import GeminiVision
        return GeminiVision

    if backend == InferenceBackend.TRANSFORMERS:
        from models.local.transformers_vlm import TransformersVLM
        return TransformersVLM

    if backend == InferenceBackend.PADDLEOCR:
        from models.local.paddle_ocr_engine import PaddleOCREngine
        return PaddleOCREngine

    raise ValueError(f"Unknown backend: {backend}")


def create_model(config: ModelConfig) -> VLMModel:
    """
    Factory function to create a model from configuration.

    Args:
        config: Model configuration.

    Returns:
        Instantiated model.
    """
    model_class = get_model_class(config.backend, config.model_id)
    return model_class(config)


def create_model_from_args(model_id: str, backend: str, **kwargs) -> VLMModel:
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
