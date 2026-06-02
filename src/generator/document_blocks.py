"""Markdown block composition for structurally diverse synthetic documents."""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

from src.generator.data_provider import DataProvider
from src.generator.table_generator import TableGenerator

DEFAULT_BLOCK_TYPES = (
    "paragraph",
    "bullet_list",
    "numbered_list",
    "checklist",
    "table",
    "formula",
    "quote",
    "code",
    "command",
    "image",
    "rule",
)
PROSE_BLOCK_TYPES = frozenset(
    {"text", "paragraph", "bullet_list", "numbered_list", "checklist", "quote"}
)

_BLOCK_TYPE_ALIASES = {"text": "paragraph"}


@dataclass(frozen=True)
class GeneratedBlock:
    """A rendered markdown block and its normalized structural type."""

    block_type: str
    markdown: str


@dataclass
class DocumentCompositionMetadata:
    """Structural metadata emitted alongside composed markdown."""

    document_shape: str
    block_types: List[str]
    block_type_counts: Dict[str, int]
    section_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_shape": self.document_shape,
            "block_types": list(self.block_types),
            "block_type_counts": dict(self.block_type_counts),
            "section_count": self.section_count,
        }


@dataclass(frozen=True)
class BlockBlueprint:
    """Normalized document block composition settings."""

    document_shape: str
    section_count: Tuple[int, int]
    blocks_per_section: Tuple[int, int]
    allowed_blocks: Tuple[str, ...]
    required_blocks: Tuple[str, ...]
    table_rows: Tuple[int, int]
    table_columns: Tuple[int, int]


def coerce_int_range(
    value: Any,
    default_min: int,
    default_max: int,
    *,
    minimum: int = 0,
) -> Tuple[int, int]:
    """Coerce an integer or two-value range into an ordered bounded tuple."""

    lower = default_min
    upper = default_max

    if isinstance(value, bool):
        value = None

    if isinstance(value, int):
        lower = value
        upper = value
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            lower = int(value[0])
            upper = int(value[1])
        except (TypeError, ValueError):
            lower = default_min
            upper = default_max

    if lower > upper:
        lower, upper = upper, lower

    lower = max(minimum, lower)
    upper = max(lower, upper)
    return lower, upper


def normalize_block_types(
    block_types: Iterable[Any] | None,
    *,
    fallback_to_default: bool = True,
) -> List[str]:
    """Normalize supported block type names, dropping unknown values."""

    if isinstance(block_types, str):
        raw_values: Iterable[Any] = [block_types]
    else:
        raw_values = block_types or []

    supported = set(DEFAULT_BLOCK_TYPES)
    normalized: List[str] = []
    seen: set[str] = set()

    for raw_value in raw_values:
        block_type = str(raw_value).strip().lower().replace("-", "_")
        block_type = _BLOCK_TYPE_ALIASES.get(block_type, block_type)
        if block_type not in supported or block_type in seen:
            continue
        normalized.append(block_type)
        seen.add(block_type)

    if normalized or not fallback_to_default:
        return normalized
    return list(DEFAULT_BLOCK_TYPES)


def parse_block_blueprint(blueprint: Mapping[str, Any] | None) -> BlockBlueprint:
    """Parse a raw document-block blueprint into normalized settings."""

    raw: Mapping[str, Any] = blueprint if isinstance(blueprint, Mapping) else {}
    document_shape = str(raw.get("document_shape") or raw.get("shape") or "document").strip()
    if not document_shape:
        document_shape = "document"

    table_raw = raw.get("table")
    table_cfg: Mapping[str, Any] = table_raw if isinstance(table_raw, Mapping) else {}
    table_rows = table_cfg.get("rows", table_cfg.get("row_count"))
    table_columns = table_cfg.get(
        "columns",
        table_cfg.get("cols", table_cfg.get("column_count")),
    )

    return BlockBlueprint(
        document_shape=document_shape,
        section_count=coerce_int_range(raw.get("section_count"), 3, 6, minimum=0),
        blocks_per_section=coerce_int_range(raw.get("blocks_per_section"), 1, 3, minimum=1),
        allowed_blocks=tuple(
            normalize_block_types(raw.get("allowed_blocks", DEFAULT_BLOCK_TYPES))
        ),
        required_blocks=tuple(
            normalize_block_types(raw.get("required_blocks"), fallback_to_default=False)
        ),
        table_rows=coerce_int_range(table_rows, 2, 4, minimum=1),
        table_columns=coerce_int_range(table_columns, 3, 5, minimum=1),
    )


class DocumentBlockBuilder:
    """Build individual markdown blocks from normalized block types."""

    def __init__(
        self,
        *,
        data: DataProvider,
        clip_text: Callable[[str, int], str],
        formula_supplier: Callable[[], str],
        table_rows: Tuple[int, int] = (2, 4),
        table_columns: Tuple[int, int] = (3, 5),
    ) -> None:
        self.data = data
        self.clip_text = clip_text
        self.formula_supplier = formula_supplier
        self.table_rows = table_rows
        self.table_columns = table_columns
        self.table_generator = TableGenerator(data=data, clip_text=clip_text)

    def build(
        self,
        block_type: str,
        *,
        block_index: int = 0,
        section_index: int = 0,
    ) -> GeneratedBlock:
        normalized = _BLOCK_TYPE_ALIASES.get(block_type, block_type)
        builder = getattr(self, f"_build_{normalized}", None)
        if builder is None:
            builder = self._build_paragraph
            normalized = "paragraph"
        return builder(block_index=block_index, section_index=section_index)

    def _build_paragraph(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = block_index, section_index
        paragraph = self.clip_text(self.data.paragraph(), 320)
        if not paragraph:
            paragraph = self.clip_text(self.data.sentence(), 160)
        return GeneratedBlock("paragraph", paragraph)

    def _build_bullet_list(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = block_index, section_index
        item_count = random.randint(3, 5)
        items = [self.clip_text(self.data.feature(), 96) for _ in range(item_count)]
        return GeneratedBlock("bullet_list", "\n".join(f"- {item}" for item in items if item))

    def _build_numbered_list(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = block_index, section_index
        item_count = random.randint(3, 5)
        items = [self.clip_text(self.data.requirement_line(), 96) for _ in range(item_count)]
        lines = [f"{index}. {item}" for index, item in enumerate(items, start=1) if item]
        return GeneratedBlock("numbered_list", "\n".join(lines))

    def _build_checklist(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = block_index, section_index
        item_count = random.randint(3, 5)
        lines: List[str] = []
        for _ in range(item_count):
            marker = random.choice([" ", "x"])
            item = self.clip_text(self.data.requirement_line(), 96)
            if item:
                lines.append(f"- [{marker}] {item}")
        return GeneratedBlock("checklist", "\n".join(lines))

    def _build_table(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = block_index, section_index
        sections = self.table_generator.generate_sections(
            section_count=1,
            row_range=self.table_rows,
            column_range=self.table_columns,
        )
        table = sections[0] if sections else ""
        table_lines = table.splitlines()
        if table_lines and table_lines[0].startswith("## "):
            table_lines = table_lines[1:]
            if table_lines and not table_lines[0].strip():
                table_lines = table_lines[1:]
            table = "\n".join(table_lines).strip()
        return GeneratedBlock("table", table)

    def _build_formula(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = block_index, section_index
        expression = " ".join(str(self.formula_supplier()).split()).strip()
        if not expression:
            expression = "x = y"
        return GeneratedBlock("formula", f"$$ {expression} $$")

    def _build_quote(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = block_index, section_index
        sentence_count = random.randint(1, 2)
        lines = [
            f"> {self.clip_text(self.data.sentence(), 120)}"
            for _ in range(sentence_count)
        ]
        return GeneratedBlock("quote", "\n".join(line for line in lines if line.strip() != ">"))

    def _build_code(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = block_index
        service_name = self._slugify(self.data.title(), fallback=f"service-{section_index}")
        lines = [
            "```yaml",
            f"name: {service_name}",
            f"endpoint: {self.data.api_endpoint()}",
            f"replicas: {random.randint(1, 4)}",
            f"metrics_enabled: {random.choice(['true', 'false'])}",
            "```",
        ]
        return GeneratedBlock("code", "\n".join(lines))

    def _build_command(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = section_index
        package_name = self._slugify(self.data.title(), fallback=f"sample-app-{block_index}")
        endpoint = self.data.api_endpoint()
        lines = [
            "```bash",
            self.data.install_command(package_name=package_name),
            self.data.usage_command(entrypoint="main.py"),
            f"curl -X GET https://example.test{endpoint}",
            "```",
        ]
        return GeneratedBlock("command", "\n".join(lines))

    def _build_image(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = section_index
        slug = self._slugify(self.data.title(), fallback="figure")
        return GeneratedBlock("image", f"![Figure](placeholder://{slug}-{block_index})")

    def _build_rule(self, *, block_index: int, section_index: int) -> GeneratedBlock:
        _ = block_index, section_index
        return GeneratedBlock("rule", "---")

    @staticmethod
    def _slugify(text: str, *, fallback: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
        return slug or fallback


class DocumentComposer:
    """Compose a markdown document from diverse block-level structures."""

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

    def compose(
        self,
        blueprint: Mapping[str, Any] | None = None,
    ) -> Tuple[str, DocumentCompositionMetadata]:
        parsed = parse_block_blueprint(blueprint)
        section_count = random.randint(*parsed.section_count)
        section_count = max(section_count, len(parsed.required_blocks))

        if section_count <= 0:
            section_count = 1

        block_counts = [
            random.randint(*parsed.blocks_per_section) for _ in range(section_count)
        ]
        block_plan = self._plan_block_types(parsed, total_slots=sum(block_counts))
        if not block_plan:
            block_counts = [1]
            section_count = 1
            block_plan = ["paragraph"]

        builder = DocumentBlockBuilder(
            data=self.data,
            clip_text=self.clip_text,
            formula_supplier=self.formula_supplier,
            table_rows=parsed.table_rows,
            table_columns=parsed.table_columns,
        )

        lines: List[str] = [f"# {self.clip_text(self.data.title(), 96)}"]
        blocks: List[GeneratedBlock] = []
        cursor = 0

        for section_index, block_count in enumerate(block_counts):
            lines.append("")
            lines.append(f"## {self.clip_text(self.data.title(), 96)}")

            for _ in range(block_count):
                if cursor >= len(block_plan):
                    break
                block_type = block_plan[cursor]
                block = builder.build(
                    block_type,
                    block_index=cursor,
                    section_index=section_index,
                )
                cursor += 1
                markdown = block.markdown.strip()
                if not markdown:
                    continue
                lines.append("")
                lines.append(markdown)
                blocks.append(block)

        if not blocks:
            fallback = builder.build("paragraph")
            lines = [
                f"# {self.clip_text(self.data.title(), 96)}",
                "",
                f"## {self.clip_text(self.data.title(), 96)}",
                "",
                fallback.markdown.strip(),
            ]
            blocks = [fallback]
            section_count = 1

        block_types = [block.block_type for block in blocks]
        metadata = DocumentCompositionMetadata(
            document_shape=parsed.document_shape,
            block_types=block_types,
            block_type_counts=dict(Counter(block_types)),
            section_count=section_count,
        )
        return "\n".join(lines).strip() + "\n", metadata

    @staticmethod
    def _plan_block_types(parsed: BlockBlueprint, *, total_slots: int) -> List[str]:
        if total_slots <= 0:
            return []

        required = list(parsed.required_blocks)
        required_set = set(required)
        allowed = list(parsed.allowed_blocks) or list(DEFAULT_BLOCK_TYPES)
        filler_candidates = [block for block in allowed if block not in required_set]
        if not filler_candidates:
            filler_candidates = allowed

        plan: List[str] = []
        for required_block in required[:total_slots]:
            plan.append(required_block)

        while len(plan) < total_slots:
            plan.append(random.choice(filler_candidates))

        return plan
