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

    def generate_sections(self, section_count: int) -> List[str]:
        sections: List[str] = []
        for _ in range(max(0, section_count)):
            heading = self.clip_text(self.data.title(), 96)
            paragraph_count = random.randint(1, 2)
            lines: List[str] = [f"## {heading}", ""]
            for index in range(paragraph_count):
                lines.append(self.clip_text(self.data.paragraph(), self.max_paragraph_chars))
                if index < paragraph_count - 1:
                    lines.append("")
            sections.append("\n".join(lines).strip())
        return sections
