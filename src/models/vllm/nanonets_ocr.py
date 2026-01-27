"""Nanonets-OCR model wrapper using vLLM."""

from typing import Union

from evaluation.config import InferenceBackend, ModelConfig
from models.local.vllm_vlm import VLLMModel


class NanonetsOCR(VLLMModel):
    """Wrapper for the Nanonets-OCR2-3B model using vLLM."""

    DEFAULT_MODEL_ID = "nanonets/Nanonets-OCR2-3B"

    def __init__(self, config: Union[ModelConfig, None] = None):
        """
        Initialize NanonetsOCR model.

        Args:
            config: ModelConfig object or None for default.
        """
        if config is None:
            config = ModelConfig(
                model_id=self.DEFAULT_MODEL_ID,
                backend=InferenceBackend.VLLM,
                temperature=0.0,
                max_tokens=1024,
            )
        super().__init__(config)
