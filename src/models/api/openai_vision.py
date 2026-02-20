"""OpenAI Vision API model (GPT-5 family)."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, cast

from PIL import Image

from evaluation.config import ModelConfig
from models.api.base import APIModel


class OpenAIVision(APIModel):
    """
    OpenAI Vision API model.

    Supports GPT-5, GPT-5-mini, and GPT-5-nano.
    """

    SUPPORTED_MODELS = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ]

    MAX_BATCH_MB = int(os.getenv("OPENAI_BATCH_MAX_MB", "100"))

    def _is_gpt5_family(self) -> bool:
        return (self.config.model_id or "").lower().startswith("gpt-5")

    @staticmethod
    def _normalize_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
            return "".join(parts)
        return ""

    def _extract_text_from_response(self, response: Any) -> str:
        output_text = None
        if isinstance(response, dict):
            output_text = response.get("output_text")
        else:
            output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text

        choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = first_choice.get("message") if isinstance(first_choice, dict) else getattr(first_choice, "message", None)
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            if content is not None:
                return self._normalize_text_content(content)

        output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                item_content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                if isinstance(item_content, list):
                    for content_item in item_content:
                        text = (
                            content_item.get("text")
                            if isinstance(content_item, dict)
                            else getattr(content_item, "text", None)
                        )
                        if isinstance(text, str) and text:
                            parts.append(text)
            if parts:
                return "".join(parts)

        return ""

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

        if self._is_gpt5_family():
            response = await self.client.responses.create(
                model=self.config.model_id,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                max_output_tokens=self.config.max_tokens,
            )
        else:
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

        return self._extract_text_from_response(response)

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
        batch_dir = Path(output_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)

        if completion_window != "24h":
            raise ValueError("Only '24h' is supported for OpenAI batch completion_window")
        if len(prompts) > 50000:
            raise ValueError("Batch request limit exceeded (max 50,000)")

        request_path = batch_dir / "requests.jsonl"
        with open(request_path, "w", encoding="utf-8") as f:
            for prompt, image, custom_id in zip(prompts, images, custom_ids):
                base64_image = self._encode_image(image)
                body = {
                    "model": self.config.model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                }
                if self._is_gpt5_family():
                    body["max_completion_tokens"] = self.config.max_tokens
                else:
                    body["max_tokens"] = self.config.max_tokens
                    body["temperature"] = self.config.temperature
                    body["top_p"] = self.config.top_p

                payload = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        request_size_mb = request_path.stat().st_size / (1024 * 1024)
        if request_size_mb > self.MAX_BATCH_MB:
            raise ValueError(
                f"Batch request file too large ({request_size_mb:.1f}MB). "
                "Reduce samples or batch size."
            )

        with open(request_path, "rb") as f:
            batch_file = await self.client.files.create(file=f, purpose="batch")

        completion_window_value: Literal["24h"] = (
            completion_window if completion_window == "24h" else "24h"
        )

        batch = await self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window_value,
        )

        await self._write_batch_info(
            batch_dir,
            {
                "batch_id": batch.id,
                "input_file_id": batch_file.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completion_window": completion_window_value,
            },
        )

        status = await self._poll_batch(
            batch.id,
            batch_dir,
            poll_interval,
            timeout,
            completion_window_value,
        )

        output_file_id = status.output_file_id
        if not output_file_id:
            raise RuntimeError("Batch completed without output file")
        return await self._fetch_batch_results(cast(str, output_file_id))

    async def resume_batch_async(
        self,
        batch_id: str,
        output_dir,
        poll_interval: int,
        timeout: int,
    ) -> Dict[str, str]:
        batch_dir = Path(output_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        status = await self._poll_batch(
            batch_id,
            batch_dir,
            poll_interval,
            timeout,
            "24h",
        )
        output_file_id = status.output_file_id
        if not output_file_id:
            raise RuntimeError("Batch completed without output file")
        return await self._fetch_batch_results(cast(str, output_file_id))

    async def _poll_batch(
        self,
        batch_id: str,
        batch_dir: Path,
        poll_interval: int,
        timeout: int,
        completion_window: str,
    ):
        start = time.time()
        status = await self.client.batches.retrieve(batch_id)
        while status.status not in {"completed", "failed", "cancelled", "expired"}:
            if time.time() - start > timeout:
                raise TimeoutError("Batch processing timed out")
            await asyncio.sleep(poll_interval)
            status = await self.client.batches.retrieve(batch_id)

        await self._write_batch_info(
            batch_dir,
            {
                "batch_id": status.id,
                "input_file_id": status.input_file_id,
                "output_file_id": status.output_file_id,
                "error_file_id": status.error_file_id,
                "status": status.status,
                "completed_at": status.completed_at,
                "completion_window": completion_window,
            },
        )

        if status.error_file_id:
            error_map = await self._fetch_batch_errors(status.error_file_id)
            error_path = batch_dir / "batch_errors.json"
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(error_map, f, ensure_ascii=False, indent=2)

        if status.status != "completed":
            raise RuntimeError(f"Batch failed with status: {status.status}")

        if not status.output_file_id:
            raise RuntimeError("Batch completed without output file")

        return status

    async def _fetch_batch_results(self, output_file_id: str) -> Dict[str, str]:
        response = await self.client.files.content(output_file_id)
        content = response.text if hasattr(response, "text") else response.read().decode()

        results: Dict[str, str] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id = data.get("custom_id")
            body = (data.get("response") or {}).get("body") or {}
            message = self._extract_text_from_response(body)
            if custom_id is not None:
                results[str(custom_id)] = message

        return results

    async def _fetch_batch_errors(self, error_file_id: str) -> Dict[str, str]:
        response = await self.client.files.content(error_file_id)
        content = response.text if hasattr(response, "text") else response.read().decode()
        errors: Dict[str, str] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id = data.get("custom_id")
            error = (data.get("error") or {}).get("message") or ""
            if custom_id is not None:
                errors[str(custom_id)] = error
        return errors

    async def _write_batch_info(self, batch_dir: Path, info: Dict[str, object]) -> None:
        batch_info_path = batch_dir / "batch_info.json"
        with open(batch_info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "gpt-5-mini",
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
