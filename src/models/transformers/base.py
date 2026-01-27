"""Base class for transformers-based OCR models."""

from typing import List, Union

from PIL import Image

from evaluation.config import ModelConfig
from models.base import VLMModel


class BaseTransformersOCR(VLMModel):
    """
    Base class for transformers-based OCR models.

    Provides common config parsing and utility methods.
    Subclasses must define DEFAULT_MODEL_ID and implement run().
    """

    DEFAULT_MODEL_ID: str = ""
    DEFAULT_MAX_TOKENS: int = 1024

    def __init__(self, config: Union[ModelConfig, str, None] = None):
        """
        Initialize the model.

        Args:
            config: ModelConfig object, model_id string, or None for default.
        """
        model_id, self.config = self._parse_config(config)
        self._load_model(model_id)

    def _parse_config(
        self, config: Union[ModelConfig, str, None]
    ) -> tuple[str, Union[ModelConfig, None]]:
        """
        Parse config to extract model_id.

        Args:
            config: ModelConfig object, model_id string, or None for default.

        Returns:
            Tuple of (model_id, config or None).
        """
        if config is None:
            return self.DEFAULT_MODEL_ID, None
        if isinstance(config, str):
            return config, None
        return config.model_id, config

    def _load_model(self, model_id: str) -> None:
        """
        Load the model and processor.

        Subclasses must implement this method.

        Args:
            model_id: HuggingFace model ID.
        """
        raise NotImplementedError("Subclasses must implement _load_model")

    def _get_max_tokens(self) -> int:
        """Get max_tokens from config or use default."""
        if self.config is not None:
            return self.config.max_tokens
        return self.DEFAULT_MAX_TOKENS

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        raise NotImplementedError("Subclasses must implement run")
