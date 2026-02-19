"""Formula section generator for markdown documents."""

from typing import Callable, List

from generator.data_provider import DataProvider


class FormularGenerator:
    """Builds formula markdown sections."""

    def __init__(
        self,
        *,
        data: DataProvider,
        clip_text: Callable[[str, int], str],
        formula_supplier: Callable[[], str],
    ) -> None:
        self.data = data
        self.clip_text = clip_text
        self.formula_supplier = formula_supplier

    def generate_sections(self, section_count: int) -> List[str]:
        sections: List[str] = []
        for _ in range(max(0, section_count)):
            formula = str(self.formula_supplier() or "").strip() or "E = mc^2"
            heading = self.clip_text(self.data.title(), 96)
            lines = [f"## {heading}", "", f"$$ {formula} $$"]
            sections.append("\n".join(lines))
        return sections
