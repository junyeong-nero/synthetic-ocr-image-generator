"""Text section generator for markdown documents."""

import random
import textwrap
from typing import Callable, List

from generator.data_provider import DataProvider


class TextGenerator:
    """Builds text-only markdown sections."""

    def __init__(
        self,
        *,
        data: DataProvider,
        clip_text: Callable[[str, int], str],
        max_paragraph_chars: int = 220,
        max_line_chars: int = 72,
    ) -> None:
        self.data = data
        self.clip_text = clip_text
        self.max_paragraph_chars = max_paragraph_chars
        self.max_line_chars = max(1, max_line_chars)

    def _format_text_block(self, text: str) -> str:
        clipped = self.clip_text(text, self.max_paragraph_chars)
        if not clipped:
            return ""

        wrapped_lines = textwrap.wrap(
            clipped,
            width=self.max_line_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not wrapped_lines:
            return clipped

        return "  \n".join(wrapped_lines)

    def _next_context_lines(self, count: int) -> List[str]:
        if count <= 0:
            return []

        return [
            self._format_text_block(sentence)
            for sentence in self.data.sentences(count)
            if str(sentence).strip()
        ]

    def generate_sections(self, section_count: int) -> List[str]:
        sections: List[str] = []
        for _ in range(max(0, section_count)):
            heading = self.clip_text(self.data.title(), 96)
            paragraph_count = random.randint(1, 2)
            context_line_count = random.randint(2, 4)
            lines: List[str] = [f"## {heading}", ""]
            lines.extend(self._next_context_lines(context_line_count))
            lines.append("")
            for index in range(paragraph_count):
                lines.append(self._format_text_block(self.data.paragraph()))
                if index < paragraph_count - 1:
                    lines.append("")
            sections.append("\n".join(lines).strip())
        return sections
