"""DeepSeek-OCR model wrapper."""

from src.models.transformers.base import StandardTransformersOCR


class DeepSeekOCR(StandardTransformersOCR):
    """Wrapper for the original DeepSeek-OCR model."""

    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
