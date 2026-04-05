"""Claude Vision API model (Claude 4.5/4.6 family)."""

import os
from typing import Optional

from PIL import Image

from src.evaluation.config import ModelConfig
from src.models.api.base import APIModel


class ClaudeVision(APIModel):
    """
    Anthropic Claude Vision API model.

    Supports Claude Sonnet 4.5 and Claude Opus 4.6.
    """

    SUPPORTED_MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-5",
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

        if not response.content:
            return ""

        text_chunks: list[str] = []
        for block in response.content:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str) and block_text:
                text_chunks.append(block_text)

        return "\n".join(text_chunks)

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "claude-sonnet-4-5",
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
        from src.evaluation.config import InferenceBackend

        config = ModelConfig(
            model_id=model_id,
            backend=InferenceBackend.ANTHROPIC,
            api_key=api_key,
            **kwargs,
        )
        return cls(config)
