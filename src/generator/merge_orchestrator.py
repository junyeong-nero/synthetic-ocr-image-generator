"""Section merge orchestration for markdown documents."""

import random
from typing import Callable, List, Tuple

from src.generator.data_provider import DataProvider


class MergeOrchestrator:
    """Merges and shuffles markdown sections from component generators."""

    def __init__(self, *, data: DataProvider, clip_text: Callable[[str, int], str]) -> None:
        self.data = data
        self.clip_text = clip_text

    def merge(
        self,
        *,
        text_sections: List[str],
        table_sections: List[str],
        formula_sections: List[str],
    ) -> Tuple[str, List[str]]:
        grouped_sections: List[Tuple[str, str]] = []
        grouped_sections.extend(("text", section) for section in text_sections)
        grouped_sections.extend(("table", section) for section in table_sections)
        grouped_sections.extend(("formula", section) for section in formula_sections)
        random.shuffle(grouped_sections)

        lines: List[str] = [f"# {self.clip_text(self.data.title(), 110)}", ""]
        section_order: List[str] = []
        for section_type, section in grouped_sections:
            section_order.append(section_type)
            lines.extend(section.splitlines())
            lines.append("")

        return "\n".join(lines).strip(), section_order
