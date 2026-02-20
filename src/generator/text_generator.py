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
        self._context_line_builders = [
            lambda: f"- Person: {self.data.name()}",
            lambda: f"- Company: {self.data.company()}",
            lambda: f"- Position: {self.data.position()}",
            lambda: f"- Department: {self.data.department()}",
            lambda: f"- Address: {self.data.address()}",
            lambda: f"- Store: {self.data.store_name()}",
            lambda: f"- Product: {self.data.product_name()}",
            lambda: f"- Feature: {self.data.feature()}",
            lambda: f"- Requirement: {self.data.requirement_line()}",
            lambda: f"- API Endpoint: `{self.data.api_endpoint()}`",
            lambda: f"- Config: `{self.data.config_line()}`",
            lambda: f"- Install: `{self.data.install_command()}`",
            lambda: f"- Usage: `{self.data.usage_command()}`",
        ]
        random.shuffle(self._context_line_builders)
        self._context_line_index = 0

    def _next_context_lines(self, count: int) -> List[str]:
        if count <= 0:
            return []

        lines: List[str] = []
        for _ in range(count):
            if self._context_line_index >= len(self._context_line_builders):
                self._context_line_index = 0
                random.shuffle(self._context_line_builders)

            line = self._context_line_builders[self._context_line_index]()
            self._context_line_index += 1
            lines.append(self.clip_text(line, self.max_paragraph_chars))
        return lines

    def generate_sections(self, section_count: int) -> List[str]:
        sections: List[str] = []
        for _ in range(max(0, section_count)):
            heading = self.clip_text(self.data.title(), 96)
            paragraph_count = random.randint(1, 2)
            context_line_count = random.randint(4, 6)
            lines: List[str] = [f"## {heading}", ""]
            lines.extend(self._next_context_lines(context_line_count))
            lines.append("")
            for index in range(paragraph_count):
                lines.append(self.clip_text(self.data.paragraph(), self.max_paragraph_chars))
                if index < paragraph_count - 1:
                    lines.append("")
            sections.append("\n".join(lines).strip())
        return sections
