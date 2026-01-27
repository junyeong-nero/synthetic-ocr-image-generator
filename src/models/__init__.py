"""OCR/VLM models.

Model classes are lazily imported to avoid import errors when optional
dependencies (transformers, vllm, etc.) are not installed.
"""

from typing import TYPE_CHECKING

# Base classes (always available)
from models.base import Model, VLMModel, vLLMModel, encode_image_base64, generate_message

# Model registry (always available)
from models.registry import (
    create_model,
    create_model_from_args,
    get_model_class,
    list_backends,
    BACKEND_DISPLAY_NAMES,
)

# Lazy imports for type checking only
if TYPE_CHECKING:
    from models.vllm.dots_ocr import DotsOCR
    from models.vllm.light_on_ocr import LightOnOCR
    from models.vllm.nanonets_ocr import NanonetsOCR
    from models.vllm.olm_ocr import OlmOCR
    from models.transformers.deepseek_ocr import DeepSeekOCR
    from models.transformers.deepseek_ocr2 import DeepSeekOCR2
    from models.transformers.gemma3_4b_it import Gemma3_4B_IT
    from models.transformers.got_ocr import GotOCR
    from models.transformers.paddle_ocr import PaddleOCR
    from models.transformers.qwen3_vl import Qwen3VL
    from models.transformers.varco_ocr import VarcoOCR
    from models.transformers.light_on_ocr2 import LightOnOCR2


def __getattr__(name: str):
    """Lazy import model classes on first access."""
    # vLLM-based models
    if name == "DotsOCR":
        from models.vllm.dots_ocr import DotsOCR
        return DotsOCR
    if name == "LightOnOCR":
        from models.vllm.light_on_ocr import LightOnOCR
        return LightOnOCR
    if name == "NanonetsOCR":
        from models.vllm.nanonets_ocr import NanonetsOCR
        return NanonetsOCR
    if name == "OlmOCR":
        from models.vllm.olm_ocr import OlmOCR
        return OlmOCR

    # Transformers-based models
    if name == "DeepSeekOCR":
        from models.transformers.deepseek_ocr import DeepSeekOCR
        return DeepSeekOCR
    if name == "DeepSeekOCR2":
        from models.transformers.deepseek_ocr2 import DeepSeekOCR2
        return DeepSeekOCR2
    if name == "Gemma3_4B_IT":
        from models.transformers.gemma3_4b_it import Gemma3_4B_IT
        return Gemma3_4B_IT
    if name == "GotOCR":
        from models.transformers.got_ocr import GotOCR
        return GotOCR
    if name == "PaddleOCR":
        from models.transformers.paddle_ocr import PaddleOCR
        return PaddleOCR
    if name == "Qwen3VL":
        from models.transformers.qwen3_vl import Qwen3VL
        return Qwen3VL
    if name == "VarcoOCR":
        from models.transformers.varco_ocr import VarcoOCR
        return VarcoOCR
    if name == "LightOnOCR2":
        from models.transformers.light_on_ocr2 import LightOnOCR2
        return LightOnOCR2

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base
    "Model",
    "VLMModel",
    "vLLMModel",
    "encode_image_base64",
    "generate_message",
    # Registry
    "create_model",
    "create_model_from_args",
    "get_model_class",
    "list_backends",
    "BACKEND_DISPLAY_NAMES",
    # Legacy models (lazily imported)
    "DotsOCR",
    "LightOnOCR",
    "NanonetsOCR",
    "OlmOCR",
    "DeepSeekOCR",
    "DeepSeekOCR2",
    "Gemma3_4B_IT",
    "GotOCR",
    "PaddleOCR",
    "Qwen3VL",
    "VarcoOCR",
    "LightOnOCR2",
]
