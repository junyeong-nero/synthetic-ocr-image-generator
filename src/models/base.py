"""Base classes for OCR/VLM models."""

import asyncio
import base64
import io
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from PIL import Image


def generate_message(
    image: Image.Image,
    prompt: str = "OCR this image",
) -> List[Dict]:
    """
    Generates a message payload for a multimodal model.

    Args:
        image: The PIL Image to be included.
        prompt: The text prompt to accompany the image.

    Returns:
        A list of dictionaries representing the message structure.
    """
    buf = io.BytesIO()
    image = image.convert("RGB")
    image.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode a PIL Image to base64 string.

    Args:
        image: The PIL Image to encode.
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 encoded string.
    """
    buf = io.BytesIO()
    image = image.convert("RGB")
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode()


class Model:
    """Base class for OCR models (legacy compatibility)."""

    def __init__(self) -> None:
        pass

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Runs the OCR model on a batch of images and prompts.

        Args:
            prompts: A list of text prompts.
            images: A list of PIL Images.

        Returns:
            A list of OCR results as strings.
        """
        assert len(prompts) == len(images)
        return ["empty"] * len(prompts)


class VLMModel(ABC):
    """
    Abstract base class for Vision Language Models.

    Supports both synchronous and asynchronous inference.
    All new model implementations should inherit from this class.
    """

    @abstractmethod
    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run synchronous inference on a batch of images.

        Args:
            prompts: A list of text prompts.
            images: A list of PIL Images.

        Returns:
            A list of model outputs as strings.
        """
        pass

    async def run_async(
        self, prompts: List[str], images: List[Image.Image]
    ) -> List[str]:
        """
        Run asynchronous inference on a batch of images.

        Default implementation wraps the synchronous run() method.
        Override this for true async support (e.g., API models).

        Args:
            prompts: A list of text prompts.
            images: A list of PIL Images.

        Returns:
            A list of model outputs as strings.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, prompts, images)

    async def run_batch_async(
        self,
        prompts: List[str],
        images: List[Image.Image],
        custom_ids: List[str],
        output_dir,
        completion_window: str,
        poll_interval: int,
        timeout: int,
    ) -> Dict[str, str]:
        raise NotImplementedError

    async def resume_batch_async(
        self,
        batch_id: str,
        output_dir,
        poll_interval: int,
        timeout: int,
    ) -> Dict[str, str]:
        raise NotImplementedError

    def run_batch(
        self,
        prompts: List[str],
        images: List[Image.Image],
        batch_size: int = 1,
    ) -> List[str]:
        """
        Run inference with batching.

        Args:
            prompts: A list of text prompts.
            images: A list of PIL Images.
            batch_size: Number of samples per batch.

        Returns:
            A list of model outputs as strings.
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_images = images[i : i + batch_size]
            results.extend(self.run(batch_prompts, batch_images))
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }
