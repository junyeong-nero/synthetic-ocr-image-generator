"""OpenAI Vision API model (GPT-4o, GPT-4V)."""

import os
from typing import List, Optional

from PIL import Image

from evaluation.config import ModelConfig
from models.api.base import APIModel


class OpenAIVision(APIModel):
    """
    OpenAI Vision API model.

    Supports GPT-4o, GPT-4o-mini, GPT-4-turbo, etc.
    """

    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
    ]

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI Vision. "
                "Install with: pip install openai"
            )

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.api_base,
        )

    async def _call_api(self, prompt: str, image: Image.Image) -> str:
        """
        Make a single API call to OpenAI.

        Args:
            prompt: Text prompt.
            image: PIL Image.

        Returns:
            Model response as string.
        """
        base64_image = self._encode_image(image)

        response = await self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        return response.choices[0].message.content or ""

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "gpt-4o",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "OpenAIVision":
        """
        Create an OpenAI Vision model from model ID.

        Args:
            model_id: OpenAI model ID.
            api_key: Optional API key.
            **kwargs: Additional config options.

        Returns:
            OpenAIVision instance.
        """
        from evaluation.config import InferenceBackend

        config = ModelConfig(
            model_id=model_id,
            backend=InferenceBackend.OPENAI,
            api_key=api_key,
            **kwargs,
        )
        return cls(config)
