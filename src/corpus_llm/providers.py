import os
from typing import List, Optional, Protocol

from src.corpus_llm.constants import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL


class LLMProvider(Protocol):
    async def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        ...


class OpenAIProvider:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_OPENAI_MODEL):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _is_gpt5_family(self) -> bool:
        return self.model.lower().startswith("gpt-5")

    @staticmethod
    def _extract_text(response: object) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text

        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content

        output = getattr(response, "output", None)
        if isinstance(output, list):
            parts: List[str] = []
            for item in output:
                item_content = getattr(item, "content", None)
                if not isinstance(item_content, list):
                    continue
                for content_item in item_content:
                    text = getattr(content_item, "text", None)
                    if isinstance(text, str) and text:
                        parts.append(text)
            if parts:
                return "".join(parts)

        return ""

    async def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        if self._is_gpt5_family():
            response = await self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max_tokens,
            )
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.9,
            )
        return self._extract_text(response)


class AnthropicProvider:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_ANTHROPIC_MODEL):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError("anthropic package required. Install with: pip install anthropic") from exc

        self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    async def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return ""

        text_chunks: List[str] = []
        for block in response.content:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str) and block_text:
                text_chunks.append(block_text)
        return "\n".join(text_chunks)


def get_provider(provider_name: str, model: Optional[str] = None) -> LLMProvider:
    if provider_name == "openai":
        return OpenAIProvider(model=model or DEFAULT_OPENAI_MODEL)
    if provider_name == "anthropic":
        return AnthropicProvider(model=model or DEFAULT_ANTHROPIC_MODEL)
    raise ValueError(f"Unknown provider: {provider_name}")
