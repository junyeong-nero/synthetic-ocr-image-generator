"""olmOCR model wrapper using vLLM."""

from typing import Union

from evaluation.config import InferenceBackend, ModelConfig
from models.local.vllm_vlm import VLLMModel


class OlmOCR(VLLMModel):
    """Wrapper for the olmOCR-2-7B-1025 model using vLLM."""

    DEFAULT_MODEL_ID = "allenai/olmOCR-2-7B-1025"

    def __init__(self, config: Union[ModelConfig, None] = None):
        """
        Initialize OlmOCR model.

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
