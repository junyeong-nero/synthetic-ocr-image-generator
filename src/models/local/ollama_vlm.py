"""Ollama-based VLM model."""

import base64
from io import BytesIO
from typing import List, Optional

from PIL import Image

from evaluation.config import ModelConfig
from models.base import VLMModel


class OllamaModel(VLMModel):
    """
    Ollama-based VLM model for easy local inference.

    Supports LLaVA, Moondream, and other vision models available in Ollama.
    """

    VISION_MODELS = [
        "llava",
        "llava-llama3",
        "llava-phi3",
        "bakllava",
        "moondream",
    ]

    def __init__(self, config: ModelConfig):
        self.config = config

        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package is required for OllamaModel. "
                "Install with: pip install ollama"
            )

        self._ollama = ollama
        self.client = ollama.Client(
            host=config.api_base or "http://localhost:11434"
        )

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image = image.convert("RGB")
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        results = []
        for prompt, image in zip(prompts, images):
            response = self.client.chat(
                model=self.config.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [self._image_to_base64(image)],
                    }
                ],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_p": self.config.top_p,
                },
            )
            results.append(response["message"]["content"])

        return results

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "llava",
        api_base: Optional[str] = None,
        **kwargs,
    ) -> "OllamaModel":
        """
        Create an Ollama model from model ID.

        Args:
            model_id: Ollama model name (e.g., "llava", "moondream").
            api_base: Optional Ollama server URL.
            **kwargs: Additional config options.

        Returns:
            OllamaModel instance.
        """
        from evaluation.config import InferenceBackend

        config = ModelConfig(
            model_id=model_id,
            backend=InferenceBackend.OLLAMA,
            api_base=api_base,
            **kwargs,
        )
        return cls(config)
