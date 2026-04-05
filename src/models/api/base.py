"""Base class for API-based VLM models."""

import asyncio
import time
from abc import abstractmethod
from typing import List, Optional

from PIL import Image

from src.evaluation.config import ModelConfig
from src.models.base import VLMModel, encode_image_base64


class RateLimiter:
    """Simple async rate limiter."""

    def __init__(self, requests_per_minute: Optional[int] = None):
        self.rpm = requests_per_minute
        self.tokens: float = float(requests_per_minute) if requests_per_minute else float("inf")
        self.max_tokens = self.tokens
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a rate limit token."""
        if self.rpm is None:
            return

        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * (self.rpm / 60.0))
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.rpm / 60.0)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class APIModel(VLMModel):
    """
    Base class for API-based VLM models.

    Provides rate limiting, retry logic, and async support.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._rate_limiter = RateLimiter(config.rate_limit_rpm)

    @abstractmethod
    async def _call_api(self, prompt: str, image: Image.Image) -> str:
        """
        Make a single API call.

        Args:
            prompt: Text prompt.
            image: PIL Image.

        Returns:
            Model response as string.
        """
        pass

    async def _call_with_retry(self, prompt: str, image: Image.Image) -> str:
        """
        Call API with exponential backoff retry.

        Args:
            prompt: Text prompt.
            image: PIL Image.

        Returns:
            Model response as string.

        Raises:
            Exception: If all retries fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                await self._rate_limiter.acquire()
                return await asyncio.wait_for(
                    self._call_api(prompt, image),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"API call timed out after {self.config.timeout}s")
            except Exception as e:
                last_error = e

            if attempt < self.config.max_retries:
                wait_time = min(2**attempt, 60)
                await asyncio.sleep(wait_time)

        raise last_error or Exception("Unknown error during API call")

    async def run_async(
        self, prompts: List[str], images: List[Image.Image]
    ) -> List[str]:
        """
        Run async inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        tasks = [
            self._call_with_retry(prompt, image)
            for prompt, image in zip(prompts, images)
        ]
        return await asyncio.gather(*tasks)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run synchronous inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        return asyncio.run(self.run_async(prompts, images))

    def _encode_image(self, image: Image.Image) -> str:
        """Encode image to base64."""
        return encode_image_base64(image)
