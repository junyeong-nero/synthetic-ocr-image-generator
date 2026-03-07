"""Text section generator for markdown documents."""

import random
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
    ) -> None:
        self.data = data
        self.clip_text = clip_text
        self.max_paragraph_chars = max_paragraph_chars

    def _next_context_lines(self, count: int) -> List[str]:
        if count <= 0:
            return []

        return [
            self.clip_text(sentence, self.max_paragraph_chars)
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
                lines.append(self.clip_text(self.data.paragraph(), self.max_paragraph_chars))
                if index < paragraph_count - 1:
                    lines.append("")
            sections.append("\n".join(lines).strip())
        return sections
