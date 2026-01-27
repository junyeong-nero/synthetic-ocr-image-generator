"""LightOnOCR model wrapper using vLLM."""

from typing import Union

from evaluation.config import InferenceBackend, ModelConfig
from models.local.vllm_vlm import VLLMModel


class LightOnOCR(VLLMModel):
    """Wrapper for the LightOnOCR-1B-1025 model using vLLM."""

    DEFAULT_MODEL_ID = "lightonai/LightOnOCR-1B-1025"

    def __init__(self, config: Union[ModelConfig, None] = None):
        """
        Initialize LightOnOCR model.

        Args:
            config: ModelConfig object or None for default.
        """
        if config is None:
            config = ModelConfig(
                model_id=self.DEFAULT_MODEL_ID,
                backend=InferenceBackend.VLLM,
                temperature=0.2,
                max_tokens=1024,
                top_p=0.9,
            )
        super().__init__(config)
