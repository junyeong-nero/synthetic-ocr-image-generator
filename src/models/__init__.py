"""OCR/VLM models."""

# Base classes
from models.base import Model, VLMModel, vLLMModel, encode_image_base64, generate_message

# Model registry
from models.registry import (
    create_model,
    create_model_from_args,
    get_model_class,
    list_backends,
    BACKEND_DISPLAY_NAMES,
)

# Legacy vLLM-based models
from models.vllm.dots_ocr import DotsOCR
from models.vllm.light_on_ocr import LightOnOCR
from models.vllm.nanonets_ocr import NanonetsOCR
from models.vllm.olm_ocr import OlmOCR

# Legacy Transformers-based models
from models.transformers.deepseek_ocr import DeepSeekOCR
from models.transformers.gemma3_4b_it import Gemma3_4B_IT
from models.transformers.got_ocr import GotOCR
from models.transformers.paddle_ocr import PaddleOCR
from models.transformers.qwen3_vl import Qwen3VL
from models.transformers.varco_ocr import VarcoOCR

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
    # Legacy models
    "DotsOCR",
    "LightOnOCR",
    "NanonetsOCR",
    "OlmOCR",
    "DeepSeekOCR",
    "Gemma3_4B_IT",
    "GotOCR",
    "PaddleOCR",
    "Qwen3VL",
    "VarcoOCR",
]
