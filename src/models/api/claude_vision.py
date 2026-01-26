"""Claude Vision API model (Claude 3.5/4 Sonnet)."""

import os
from typing import Optional

from PIL import Image

from evaluation.config import ModelConfig
from models.api.base import APIModel


class ClaudeVision(APIModel):
    """
    Anthropic Claude Vision API model.

    Supports Claude 3.5 Sonnet, Claude 4 Sonnet, etc.
    """

    SUPPORTED_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Claude Vision. "
                "Install with: pip install anthropic"
            )

        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key in config."
            )

        self.client = AsyncAnthropic(api_key=api_key)

    async def _call_api(self, prompt: str, image: Image.Image) -> str:
        """
        Make a single API call to Anthropic.

        Args:
            prompt: Text prompt.
            image: PIL Image.

        Returns:
            Model response as string.
        """
        base64_image = self._encode_image(image)

        response = await self.client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "ClaudeVision":
        """
        Create a Claude Vision model from model ID.

        Args:
            model_id: Anthropic model ID.
            api_key: Optional API key.
            **kwargs: Additional config options.

        Returns:
            ClaudeVision instance.
        """
        from evaluation.config import InferenceBackend

        config = ModelConfig(
            model_id=model_id,
            backend=InferenceBackend.ANTHROPIC,
            api_key=api_key,
            **kwargs,
        )
        return cls(config)
