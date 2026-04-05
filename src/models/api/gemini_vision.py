"""Google Gemini Vision API model."""

import asyncio
import os
from typing import Optional

from PIL import Image

from src.evaluation.config import ModelConfig
from src.models.api.base import APIModel


class GeminiVision(APIModel):
    """
    Google Gemini Vision API model.

    Supports Gemini 3.0 Pro variants.
    """

    SUPPORTED_MODELS = [
        "gemini-3.0-pro",
        "gemini-3.0-pro-latest",
        "gemini-3-flash-preview",
    ]

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package is required for Gemini Vision. "
                "Install with: pip install google-genai"
            )

        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key in config."
            )

        self.client = genai.Client(api_key=api_key)
        self._generation_config = types.GenerateContentConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_tokens,
        )

    async def _call_api(self, prompt: str, image: Image.Image) -> str:
        """
        Make a single API call to Gemini.

        Args:
            prompt: Text prompt.
            image: PIL Image.

        Returns:
            Model response as string.
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.config.model_id,
                contents=[image, prompt],
                config=self._generation_config,
            ),
        )

        if response.text:
            return response.text
        return ""

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "gemini-3.0-pro",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "GeminiVision":
        """
        Create a Gemini Vision model from model ID.

        Args:
            model_id: Google model ID.
            api_key: Optional API key.
            **kwargs: Additional config options.

        Returns:
            GeminiVision instance.
        """
        from src.evaluation.config import InferenceBackend

        config = ModelConfig(
            model_id=model_id,
            backend=InferenceBackend.GOOGLE,
            api_key=api_key,
            **kwargs,
        )
        return cls(config)
