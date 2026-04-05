import io
import json
import os
from typing import Any, Optional

from PIL import Image

from src.evaluation.config import ModelConfig
from src.models.api.base import APIModel


class UpstageDocumentParse(APIModel):
    DEFAULT_URL = "https://api.upstage.ai/v1/document-digitization"

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp package is required for Upstage Document Parse. "
                "Install with: uv sync --group api"
            )

        api_key = config.api_key or os.environ.get("UPSTAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "Upstage API key is required. Set UPSTAGE_API_KEY environment variable "
                "or pass api_key in config."
            )

        self._aiohttp = aiohttp
        self.api_key = api_key

    async def _call_api(self, prompt: str, image: Image.Image) -> str:
        del prompt

        image_bytes = self._to_png_bytes(image)
        form = self._aiohttp.FormData()
        form.add_field(
            "document",
            image_bytes,
            filename="document.png",
            content_type="image/png",
        )
        form.add_field("model", self.config.model_id or "document-parse-260128")
        form.add_field("ocr", "auto")
        form.add_field("chart_recognition", "true")
        form.add_field("coordinates", "true")
        form.add_field("output_formats", json.dumps(["html", "markdown"]))
        form.add_field("base64_encoding", json.dumps(["figure"]))

        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout = self._aiohttp.ClientTimeout(total=self.config.timeout)

        async with self._aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.config.api_base or self.DEFAULT_URL,
                headers=headers,
                data=form,
            ) as response:
                raw_body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"Upstage API request failed ({response.status}): {raw_body[:500]}"
                    )

                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Upstage API returned non-JSON response") from exc

        return self._extract_markdown(payload)

    def _to_png_bytes(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    def _extract_markdown(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload

        if isinstance(payload, dict):
            direct = self._extract_direct_content(payload)
            if direct:
                return direct

            pages = payload.get("pages")
            if isinstance(pages, list):
                chunks = []
                for page in pages:
                    if not isinstance(page, dict):
                        continue
                    chunk = self._extract_direct_content(page)
                    if chunk:
                        chunks.append(chunk)
                if chunks:
                    return "\n\n".join(chunks)

        return json.dumps(payload, ensure_ascii=False)

    def _extract_direct_content(self, content: dict[str, Any]) -> str:
        for key in ("markdown", "text", "content", "html"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                nested = self._extract_direct_content(value)
                if nested:
                    return nested
        return ""

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "document-parse-260128",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "UpstageDocumentParse":
        from src.evaluation.config import InferenceBackend

        config = ModelConfig(
            model_id=model_id,
            backend=InferenceBackend.UPSTAGE,
            api_key=api_key,
            **kwargs,
        )
        return cls(config)
