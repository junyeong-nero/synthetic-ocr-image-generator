"""Table section generator for markdown documents."""

import random
from typing import Callable, List, Tuple

from generator.data_provider import DataProvider


class TableGenerator:
    """Builds markdown table sections."""

    def __init__(self, *, data: DataProvider, clip_text: Callable[[str, int], str]) -> None:
        self.data = data
        self.clip_text = clip_text

    def _resolve_headers(self, column_count: int) -> List[str]:
        template_name = random.choice(["invoice", "schedule", "product", "contact"])
        base_headers = list(self.data.headers(template_name, count=column_count))
        if not base_headers:
            base_headers = [str(idx + 1) for idx in range(column_count)]

        headers: List[str] = []
        for idx in range(column_count):
            if idx < len(base_headers):
                headers.append(self.clip_text(str(base_headers[idx]), 24))
            else:
                headers.append(str(idx + 1))
        return headers

    def _build_cell_value(self, column_index: int, column_count: int) -> str:
        if column_index == 0:
            return self.clip_text(self.data.product_name(), 32)
        if column_index == 1:
            return str(self.data.quantity())
        if column_index == 2 and column_count >= 4:
            return self.data.format_currency(self.data.random_price())
        return self.clip_text(self.data.feature(), 36)

    def generate_sections(
        self,
        *,
        section_count: int,
        row_range: Tuple[int, int],
        column_range: Tuple[int, int],
    ) -> List[str]:
        sections: List[str] = []
        row_min, row_max = row_range
        col_min, col_max = column_range

        for _ in range(max(0, section_count)):
            row_count = random.randint(row_min, row_max)
            column_count = random.randint(col_min, col_max)
            headers = self._resolve_headers(column_count)

            lines: List[str] = [f"## {self.clip_text(self.data.title(), 96)}", ""]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for _ in range(row_count):
                row_values = [
                    self._build_cell_value(column_index=idx, column_count=column_count)
                    for idx in range(column_count)
                ]
                lines.append("| " + " | ".join(row_values) + " |")

            sections.append("\n".join(lines).strip())

        return sections
