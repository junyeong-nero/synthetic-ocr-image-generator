import os
from typing import List, Optional, Protocol

from corpus_llm.constants import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL


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

    async def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.9,
        )
        return response.choices[0].message.content or ""


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
