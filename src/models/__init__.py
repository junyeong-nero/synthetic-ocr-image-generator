"""OCR/VLM models.

Model classes are lazily imported to avoid import errors when optional
dependencies (transformers, etc.) are not installed.
"""

from typing import TYPE_CHECKING

from src.models.base import Model, VLMModel, encode_image_base64, generate_message
from src.models.registry import (
    BACKEND_DISPLAY_NAMES,
    create_model,
    create_model_from_args,
    get_model_class,
    list_backends,
)

if TYPE_CHECKING:
    from src.models.transformers.deepseek_ocr import DeepSeekOCR
    from src.models.transformers.deepseek_ocr2 import DeepSeekOCR2
    from src.models.transformers.gemma3_4b_it import Gemma3_4B_IT
    from src.models.transformers.got_ocr import GotOCR
    from src.models.transformers.light_on_ocr2 import LightOnOCR2
    from src.models.transformers.nanonets_ocr import NanonetsOCR
    from src.models.transformers.nanonets_ocr2 import NanonetsOCR2
    from src.models.transformers.paddle_ocr import PaddleOCR
    from src.models.transformers.qwen25_vl import Qwen25VL
    from src.models.transformers.qwen3_vl import Qwen3VL
    from src.models.transformers.varco_ocr import VarcoOCR


def __getattr__(name: str):
    if name == "DeepSeekOCR":
        from src.models.transformers.deepseek_ocr import DeepSeekOCR
        return DeepSeekOCR
    if name == "DeepSeekOCR2":
        from src.models.transformers.deepseek_ocr2 import DeepSeekOCR2
        return DeepSeekOCR2
    if name == "Gemma3_4B_IT":
        from src.models.transformers.gemma3_4b_it import Gemma3_4B_IT
        return Gemma3_4B_IT
    if name == "GotOCR":
        from src.models.transformers.got_ocr import GotOCR
        return GotOCR
    if name == "NanonetsOCR":
        from src.models.transformers.nanonets_ocr import NanonetsOCR
        return NanonetsOCR
    if name == "NanonetsOCR2":
        from src.models.transformers.nanonets_ocr2 import NanonetsOCR2
        return NanonetsOCR2
    if name == "PaddleOCR":
        from src.models.transformers.paddle_ocr import PaddleOCR
        return PaddleOCR
    if name == "Qwen25VL":
        from src.models.transformers.qwen25_vl import Qwen25VL
        return Qwen25VL
    if name == "Qwen3VL":
        from src.models.transformers.qwen3_vl import Qwen3VL
        return Qwen3VL
    if name == "VarcoOCR":
        from src.models.transformers.varco_ocr import VarcoOCR
        return VarcoOCR
    if name == "LightOnOCR2":
        from src.models.transformers.light_on_ocr2 import LightOnOCR2
        return LightOnOCR2

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Model",
    "VLMModel",
    "encode_image_base64",
    "generate_message",
    "create_model",
    "create_model_from_args",
    "get_model_class",
    "list_backends",
    "BACKEND_DISPLAY_NAMES",
    "DeepSeekOCR",
    "DeepSeekOCR2",
    "Gemma3_4B_IT",
    "GotOCR",
    "NanonetsOCR",
    "NanonetsOCR2",
    "PaddleOCR",
    "Qwen25VL",
    "Qwen3VL",
    "VarcoOCR",
    "LightOnOCR2",
]