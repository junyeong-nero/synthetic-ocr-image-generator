"""Dots.OCR model wrapper using vLLM."""

from typing import Union

from evaluation.config import InferenceBackend, ModelConfig
from models.local.vllm_vlm import VLLMModel


class DotsOCR(VLLMModel):
    """Wrapper for the Dots.OCR model using vLLM."""

    DEFAULT_MODEL_ID = "rednote-hilab/dots.ocr"

    def __init__(self, config: Union[ModelConfig, None] = None):
        if config is None:
            config = ModelConfig(
                model_id=self.DEFAULT_MODEL_ID,
                backend=InferenceBackend.VLLM,
                temperature=0.0,
                max_tokens=1024,
            )
        super().__init__(config)
