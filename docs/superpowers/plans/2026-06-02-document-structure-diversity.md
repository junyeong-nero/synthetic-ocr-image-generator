# Document Structure Diversity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand markdown OCR generation from a text/table/formula shuffle into metadata-auditable document families with richer markdown block structures.

**Architecture:** Add a focused block composer under `src/generator/` that builds markdown sections and returns structural metadata. Wire `MarkdownDataGenerator` and `Generator.generate_single()` to preserve the legacy section-count path while using the richer composer when templates declare `allowed_blocks`, `required_blocks`, or `document_shape`. Expand the YAML template catalog and docs after behavior is covered by tests.

**Tech Stack:** Python 3.11, dataclasses, `collections.Counter`, `random`, existing PIL markdown renderers, pytest, uv.

---

## File Structure

- Create `src/generator/document_blocks.py`: owns block names, block composition dataclasses, blueprint parsing helpers, `DocumentBlockBuilder`, and `DocumentComposer`.
- Create `tests/generator/test_document_blocks.py`: focused unit tests for rich block generation, invalid block fallback, required block inclusion, and deterministic output under seeded `random`.
- Modify `src/generator/markdown_content.py`: route rich blueprints through `DocumentComposer`, keep legacy text/table/formula composition, and expose last composition metadata.
- Modify `src/generator/text_mutation.py`: allow prose-like block types to receive similar-character mutation while preserving table/formula/code/image/rule blocks.
- Modify `src/generator/generator.py`: copy composer metadata into final sample metadata.
- Modify `tests/generator/test_dynamic_templates.py`: update legacy compatibility expectations and add generate-single metadata coverage.
- Modify `configs/generator/templates/default.yaml`: replace the single flat template with a multi-template catalog using `sections` mode and richer block blueprints.
- Modify `docs/generation.md`: document structure-diversity templates and usage recipes.

---

### Task 1: Add Rich Document Block Composer

**Files:**
- Create: `src/generator/document_blocks.py`
- Create: `tests/generator/test_document_blocks.py`

- [ ] **Step 1: Write failing tests for block composition**

Create `tests/generator/test_document_blocks.py`:

```python
import random
from collections import Counter

from generator.data_provider import DataProvider
from generator.document_blocks import (
    DEFAULT_BLOCK_TYPES,
    DocumentComposer,
    normalize_block_types,
)


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def test_normalize_block_types_ignores_unknown_values() -> None:
    assert normalize_block_types(["paragraph", "unknown", "table"]) == ["paragraph", "table"]


def test_normalize_block_types_falls_back_when_all_values_invalid() -> None:
    assert normalize_block_types(["unknown"]) == list(DEFAULT_BLOCK_TYPES)


def test_composer_satisfies_required_blocks_and_records_metadata() -> None:
    random.seed(11)
    composer = DocumentComposer(
        data=DataProvider(lang="en", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "E = mc^2",
    )
    markdown, metadata = composer.compose(
        {
            "document_shape": "technical_manual",
            "section_count": [4, 4],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["paragraph", "bullet_list", "code", "table"],
            "required_blocks": ["code", "table"],
            "table": {"rows": [2, 2], "columns": [3, 3]},
        }
    )

    assert markdown.startswith("# ")
    assert "```" in markdown
    assert "| " in markdown
    assert metadata.document_shape == "technical_manual"
    assert metadata.section_count == 4
    assert "code" in metadata.block_types
    assert "table" in metadata.block_types
    assert metadata.block_type_counts["code"] == 1
    assert metadata.block_type_counts["table"] == 1


def test_composer_emits_each_supported_block_type() -> None:
    composer = DocumentComposer(
        data=DataProvider(lang="en", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "x^2 + y^2 = z^2",
    )

    markdown, metadata = composer.compose(
        {
            "document_shape": "all_blocks",
            "section_count": [11, 11],
            "blocks_per_section": [1, 1],
            "allowed_blocks": list(DEFAULT_BLOCK_TYPES),
            "required_blocks": list(DEFAULT_BLOCK_TYPES),
            "table": {"rows": [2, 2], "columns": [3, 3]},
        }
    )

    counts = Counter(metadata.block_types)
    assert set(DEFAULT_BLOCK_TYPES).issubset(counts)
    assert "- " in markdown
    assert "1. " in markdown
    assert "- [ ]" in markdown or "- [x]" in markdown
    assert "> " in markdown
    assert "```" in markdown
    assert "![Figure]" in markdown
    assert "---" in markdown
    assert "$$ x^2 + y^2 = z^2 $$" in markdown


def test_composer_output_is_deterministic_with_seed() -> None:
    blueprint = {
        "document_shape": "release_note",
        "section_count": [5, 5],
        "blocks_per_section": [1, 1],
        "allowed_blocks": ["paragraph", "bullet_list", "checklist", "rule"],
        "required_blocks": ["checklist"],
    }

    random.seed(123)
    first_composer = DocumentComposer(
        data=DataProvider(lang="en", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "a=b",
    )
    first_markdown, first_metadata = first_composer.compose(blueprint)

    random.seed(123)
    second_composer = DocumentComposer(
        data=DataProvider(lang="en", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "a=b",
    )
    second_markdown, second_metadata = second_composer.compose(blueprint)

    assert first_markdown == second_markdown
    assert first_metadata.to_dict() == second_metadata.to_dict()
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
uv run pytest tests/generator/test_document_blocks.py -q
```

Expected: failure during import with `ModuleNotFoundError: No module named 'generator.document_blocks'`.

- [ ] **Step 3: Implement `document_blocks.py`**

Create `src/generator/document_blocks.py`:

```python
from __future__ import annotations

import random
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from src.generator.data_provider import DataProvider
from src.generator.table_generator import TableGenerator

DEFAULT_BLOCK_TYPES: tuple[str, ...] = (
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

PROSE_BLOCK_TYPES: frozenset[str] = frozenset(
    {"text", "paragraph", "bullet_list", "numbered_list", "checklist", "quote"}
)


@dataclass(frozen=True)
class GeneratedBlock:
    block_type: str
    markdown: str


@dataclass(frozen=True)
class DocumentCompositionMetadata:
    document_shape: str
    block_types: list[str]
    block_type_counts: dict[str, int]
    section_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_shape": self.document_shape,
            "block_types": list(self.block_types),
            "block_type_counts": dict(self.block_type_counts),
            "section_count": self.section_count,
        }


@dataclass(frozen=True)
class CompositionResult:
    markdown: str
    metadata: DocumentCompositionMetadata


@dataclass
class BlockBlueprint:
    document_shape: str = "sections"
    section_count: tuple[int, int] = (3, 5)
    blocks_per_section: tuple[int, int] = (1, 2)
    allowed_blocks: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCK_TYPES))
    required_blocks: list[str] = field(default_factory=list)
    max_line_chars: int = 72
    table_rows: tuple[int, int] = (2, 4)
    table_columns: tuple[int, int] = (3, 5)


def coerce_int_range(value: Any, default_min: int, default_max: int) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            lower = int(value[0])
            upper = int(value[1])
        except (TypeError, ValueError):
            return default_min, default_max
        if lower > upper:
            lower, upper = upper, lower
        return max(0, lower), max(0, upper)

    if isinstance(value, int):
        parsed = max(0, value)
        return parsed, parsed

    return default_min, default_max


def normalize_block_types(values: Any, fallback: Sequence[str] = DEFAULT_BLOCK_TYPES) -> list[str]:
    if isinstance(values, str):
        raw_values: Sequence[Any] = [values]
    elif isinstance(values, Sequence):
        raw_values = values
    else:
        raw_values = []

    normalized: list[str] = []
    for value in raw_values:
        block_type = str(value).strip().lower().replace("-", "_")
        if block_type == "text":
            block_type = "paragraph"
        if block_type in DEFAULT_BLOCK_TYPES and block_type not in normalized:
            normalized.append(block_type)

    if normalized:
        return normalized
    return list(fallback)


def parse_block_blueprint(blueprint: Mapping[str, Any]) -> BlockBlueprint:
    text_cfg = blueprint.get("text") if isinstance(blueprint.get("text"), Mapping) else {}
    table_cfg = blueprint.get("table") if isinstance(blueprint.get("table"), Mapping) else {}

    section_count = coerce_int_range(blueprint.get("section_count"), 3, 5)
    blocks_per_section = coerce_int_range(blueprint.get("blocks_per_section"), 1, 2)
    if blocks_per_section == (0, 0):
        blocks_per_section = (1, 1)

    row_value = table_cfg.get("rows", table_cfg.get("row_count")) if isinstance(table_cfg, Mapping) else None
    column_value = (
        table_cfg.get("columns", table_cfg.get("cols", table_cfg.get("column_count")))
        if isinstance(table_cfg, Mapping)
        else None
    )

    max_line_chars = 72
    if isinstance(text_cfg, Mapping):
        try:
            max_line_chars = max(1, int(text_cfg.get("max_line_chars", 72)))
        except (TypeError, ValueError):
            max_line_chars = 72

    allowed_blocks = normalize_block_types(blueprint.get("allowed_blocks"))
    required_blocks = normalize_block_types(blueprint.get("required_blocks"), fallback=[])
    required_blocks = [block_type for block_type in required_blocks if block_type in allowed_blocks]

    document_shape = str(blueprint.get("document_shape") or "sections").strip().lower() or "sections"

    return BlockBlueprint(
        document_shape=document_shape,
        section_count=section_count,
        blocks_per_section=blocks_per_section,
        allowed_blocks=allowed_blocks,
        required_blocks=required_blocks,
        max_line_chars=max_line_chars,
        table_rows=coerce_int_range(row_value, 2, 4),
        table_columns=coerce_int_range(column_value, 3, 5),
    )


class DocumentBlockBuilder:
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

    def build(self, block_type: str, blueprint: BlockBlueprint, index: int) -> GeneratedBlock:
        builders = {
            "paragraph": self._paragraph,
            "bullet_list": self._bullet_list,
            "numbered_list": self._numbered_list,
            "checklist": self._checklist,
            "table": self._table,
            "formula": self._formula,
            "quote": self._quote,
            "code": self._code,
            "command": self._command,
            "image": self._image,
            "rule": self._rule,
        }
        builder = builders.get(block_type, self._paragraph)
        return GeneratedBlock(block_type=block_type if block_type in builders else "paragraph", markdown=builder(blueprint, index))

    def _heading(self) -> str:
        return f"## {self.clip_text(self.data.title(), 96)}"

    def _wrap(self, text: str, width: int) -> str:
        wrapped = textwrap.wrap(
            self.clip_text(text, 260),
            width=max(1, width),
            break_long_words=True,
            break_on_hyphens=False,
        )
        return "  \n".join(wrapped) if wrapped else self.clip_text(text, 260)

    def _paragraph(self, blueprint: BlockBlueprint, _index: int) -> str:
        lines = [self._heading(), ""]
        lines.extend(self._wrap(sentence, blueprint.max_line_chars) for sentence in self.data.sentences(random.randint(1, 3)))
        lines.append("")
        lines.append(self._wrap(self.data.paragraph(), blueprint.max_line_chars))
        return "\n".join(line for line in lines if line is not None).strip()

    def _bullet_list(self, _blueprint: BlockBlueprint, _index: int) -> str:
        items = [self.clip_text(self.data.feature(), 90) for _ in range(random.randint(3, 5))]
        return "\n".join([self._heading(), "", *[f"- {item}" for item in items]]).strip()

    def _numbered_list(self, _blueprint: BlockBlueprint, _index: int) -> str:
        items = [self.clip_text(self.data.requirement_line(), 90) for _ in range(random.randint(3, 5))]
        return "\n".join([self._heading(), "", *[f"{idx}. {item}" for idx, item in enumerate(items, start=1)]]).strip()

    def _checklist(self, _blueprint: BlockBlueprint, _index: int) -> str:
        items = []
        for _ in range(random.randint(3, 5)):
            marker = "x" if random.random() < 0.35 else " "
            items.append(f"- [{marker}] {self.clip_text(self.data.feature(), 90)}")
        return "\n".join([self._heading(), "", *items]).strip()

    def _table(self, blueprint: BlockBlueprint, _index: int) -> str:
        section = TableGenerator(data=self.data, clip_text=self.clip_text).generate_sections(
            section_count=1,
            row_range=blueprint.table_rows,
            column_range=blueprint.table_columns,
        )
        return section[0] if section else self._paragraph(blueprint, _index)

    def _formula(self, _blueprint: BlockBlueprint, _index: int) -> str:
        formula = str(self.formula_supplier() or "").strip() or "E = mc^2"
        return "\n".join([self._heading(), "", f"$$ {formula} $$"]).strip()

    def _quote(self, _blueprint: BlockBlueprint, _index: int) -> str:
        lines = [f"> {self.clip_text(sentence, 120)}" for sentence in self.data.sentences(random.randint(1, 2))]
        return "\n".join([self._heading(), "", *lines]).strip()

    def _code(self, _blueprint: BlockBlueprint, _index: int) -> str:
        lines = [self.data.config_line() for _ in range(random.randint(3, 6))]
        return "\n".join([self._heading(), "", "```yaml", *lines, "```"]).strip()

    def _command(self, _blueprint: BlockBlueprint, _index: int) -> str:
        lines = [self.data.install_command(), self.data.usage_command(), f"curl {self.data.api_endpoint()}"]
        return "\n".join([self._heading(), "", "```bash", *lines, "```"]).strip()

    def _image(self, _blueprint: BlockBlueprint, index: int) -> str:
        alt = self.clip_text(self.data.title(), 48) or "Figure"
        slug = alt.lower().replace(" ", "-")
        return "\n".join([self._heading(), "", f"![Figure](placeholder://{slug}-{index})"]).strip()

    def _rule(self, _blueprint: BlockBlueprint, _index: int) -> str:
        return "\n".join([self._heading(), "", "---"]).strip()


class DocumentComposer:
    def __init__(
        self,
        *,
        data: DataProvider,
        clip_text: Callable[[str, int], str],
        formula_supplier: Callable[[], str],
    ) -> None:
        self.data = data
        self.clip_text = clip_text
        self.builder = DocumentBlockBuilder(
            data=data,
            clip_text=clip_text,
            formula_supplier=formula_supplier,
        )

    def compose(self, blueprint: Mapping[str, Any]) -> tuple[str, DocumentCompositionMetadata]:
        parsed = parse_block_blueprint(blueprint)
        section_count = random.randint(*parsed.section_count)
        section_count = max(section_count, len(parsed.required_blocks), 1)

        block_plan: list[str] = list(parsed.required_blocks)
        while len(block_plan) < section_count:
            block_plan.append(random.choice(parsed.allowed_blocks))
        random.shuffle(block_plan)

        blocks: list[GeneratedBlock] = []
        for index, block_type in enumerate(block_plan):
            block = self.builder.build(block_type, parsed, index)
            if block.markdown.strip():
                blocks.append(block)

        if not blocks:
            blocks.append(self.builder.build("paragraph", parsed, 0))

        title = f"# {self.clip_text(self.data.title(), 110)}"
        markdown_parts = [title, *[block.markdown.strip() for block in blocks]]
        block_types = [block.block_type for block in blocks]
        counts = Counter(block_types)
        metadata = DocumentCompositionMetadata(
            document_shape=parsed.document_shape,
            block_types=block_types,
            block_type_counts=dict(counts),
            section_count=len(blocks),
        )
        return "\n\n".join(markdown_parts).strip(), metadata
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/generator/test_document_blocks.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/generator/document_blocks.py tests/generator/test_document_blocks.py
git commit -m "[add] Compose rich markdown document blocks"
```

Expected: commit succeeds with only the new module and focused tests.

---

### Task 2: Wire Rich Composer Into Markdown Content Generation

**Files:**
- Modify: `src/generator/markdown_content.py`
- Modify: `tests/generator/test_dynamic_templates.py`

- [ ] **Step 1: Add failing tests for rich blueprint routing and legacy compatibility**

Append these tests near the existing section-generation tests in `tests/generator/test_dynamic_templates.py`:

```python
def test_rich_blueprint_generation_records_block_metadata() -> None:
    random.seed(101)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="rich_blocks_test",
        family="technical",
        complexity=3,
        mode="sections",
        blueprint={
            "document_shape": "technical_manual",
            "section_count": [4, 4],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["paragraph", "code", "table", "checklist"],
            "required_blocks": ["code", "table"],
            "table": {"rows": [2, 2], "columns": [3, 3]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    merge_order = data_generator.pop_merge_order()
    composition = data_generator.pop_composition_metadata()

    assert "```" in markdown
    assert "| " in markdown
    assert len(merge_order) == 4
    assert "code" in merge_order
    assert "table" in merge_order
    assert composition["document_shape"] == "technical_manual"
    assert composition["block_type_counts"]["code"] == 1
    assert composition["block_type_counts"]["table"] == 1


def test_legacy_sections_generation_keeps_existing_merge_order() -> None:
    random.seed(102)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="legacy_sections_still_work",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "text": {"section_count": [1, 1]},
            "table": {"section_count": [1, 1], "rows": [2, 2], "columns": [3, 3]},
            "formula": {"section_count": [1, 1]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    merge_order = data_generator.pop_merge_order()
    composition = data_generator.pop_composition_metadata()

    assert "$$" in markdown
    assert "| " in markdown
    assert sorted(merge_order) == ["formula", "table", "text"]
    assert composition["document_shape"] == "sections"
    assert composition["block_type_counts"] == {"text": 1, "table": 1, "formula": 1}
```

Update `test_sections_mode_ignores_blueprint_only_controls` because rich controls should now be honored. Replace it with:

```python
def test_sections_mode_honors_rich_blueprint_controls() -> None:
    random.seed(19)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="sections_honor_blueprint_controls",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "document_shape": "mixed_controls",
            "section_count": [3, 3],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["paragraph", "image", "checklist"],
            "required_blocks": ["image", "checklist"],
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    composition = data_generator.pop_composition_metadata()

    assert "![" in markdown
    assert "- [ ]" in markdown or "- [x]" in markdown
    assert composition["document_shape"] == "mixed_controls"
    assert "image" in composition["block_types"]
    assert "checklist" in composition["block_types"]
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
uv run pytest tests/generator/test_dynamic_templates.py::test_rich_blueprint_generation_records_block_metadata tests/generator/test_dynamic_templates.py::test_legacy_sections_generation_keeps_existing_merge_order tests/generator/test_dynamic_templates.py::test_sections_mode_honors_rich_blueprint_controls -q
```

Expected: failures because `pop_composition_metadata()` does not exist and rich controls are not routed.

- [ ] **Step 3: Import composer types in `markdown_content.py`**

Add these imports near existing generator imports in `src/generator/markdown_content.py`:

```python
from src.generator.document_blocks import DocumentComposer, DocumentCompositionMetadata
```

- [ ] **Step 4: Track composition metadata**

In `MarkdownDataGenerator.__init__`, add:

```python
        self._last_composition_metadata: Dict[str, Any] = {
            "document_shape": "sections",
            "block_types": [],
            "block_type_counts": {},
            "section_count": 0,
        }
```

Add this method after `pop_merge_order()`:

```python
    def pop_composition_metadata(self) -> Dict[str, Any]:
        metadata = dict(self._last_composition_metadata)
        metadata["block_types"] = list(metadata.get("block_types", []))
        metadata["block_type_counts"] = dict(metadata.get("block_type_counts", {}))
        self._last_composition_metadata = {
            "document_shape": "sections",
            "block_types": [],
            "block_type_counts": {},
            "section_count": 0,
        }
        return metadata
```

- [ ] **Step 5: Add rich blueprint detection**

Add this static method to `MarkdownDataGenerator` before `_generate_from_sections()`:

```python
    @staticmethod
    def _uses_rich_block_composition(blueprint: Dict[str, Any]) -> bool:
        return any(
            key in blueprint
            for key in (
                "document_shape",
                "section_count",
                "blocks_per_section",
                "allowed_blocks",
                "required_blocks",
            )
        )
```

- [ ] **Step 6: Route rich blueprints through `DocumentComposer`**

At the start of `_generate_from_sections()`, before reading `text_cfg_raw`, add:

```python
        if self._uses_rich_block_composition(blueprint):
            composer = DocumentComposer(
                data=self.data,
                clip_text=self._clip_text,
                formula_supplier=self._generate_formula_expression,
            )
            markdown_text, composition_metadata = composer.compose(blueprint)
            self._last_merge_order = list(composition_metadata.block_types)
            self._last_composition_metadata = composition_metadata.to_dict()
            return markdown_text
```

- [ ] **Step 7: Record legacy composition metadata**

After legacy `markdown_text, merge_order = orchestrator.merge(...)` in `_generate_from_sections()`, replace the current return tail with:

```python
        self._last_merge_order = merge_order
        block_counts = Counter(merge_order)
        self._last_composition_metadata = {
            "document_shape": str(blueprint.get("document_shape") or "sections"),
            "block_types": list(merge_order),
            "block_type_counts": dict(block_counts),
            "section_count": len(merge_order),
        }
        return markdown_text
```

Add `Counter` to the imports at the top of `markdown_content.py`:

```python
from collections import Counter
```

- [ ] **Step 8: Run focused tests**

Run:

```bash
uv run pytest tests/generator/test_document_blocks.py tests/generator/test_dynamic_templates.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

Run:

```bash
git add src/generator/markdown_content.py tests/generator/test_dynamic_templates.py
git commit -m "[feat] Route markdown generation through rich block composer"
```

Expected: commit succeeds with composer wiring and tests.

---

### Task 3: Add Diversity Metadata And Prose-Block Mutation

**Files:**
- Modify: `src/generator/generator.py`
- Modify: `src/generator/text_mutation.py`
- Modify: `tests/generator/test_dynamic_templates.py`

- [ ] **Step 1: Add failing test for metadata fields**

Append this test near `test_generate_single_metadata_does_not_include_a4_clipping_flags` in `tests/generator/test_dynamic_templates.py`:

```python
def test_generate_single_includes_document_structure_metadata(monkeypatch) -> None:
    generator = Generator.__new__(Generator)
    generator.template_specs = [
        TemplateSpec(
            template_id="rich-template",
            family="technical",
            mode="sections",
            complexity=3,
            source="test",
            weight=1.0,
            version="1",
            blueprint={},
        )
    ]
    generator.template_catalog = None
    generator.template_counts = Counter()
    generator.family_counts = Counter()
    generator.novelty_window = 8
    generator.novelty_threshold = 1.0
    generator.novelty_max_attempts = 1
    generator._recent_signatures = deque(maxlen=generator.novelty_window)
    generator.base_seed = None
    generator.noise_ratio = 0.0
    generator.blur_ratio = 0.0
    generator.style_profile = "balanced"
    generator.markdown_renderer = "pil"
    generator.similar_char_ratio = 0.0
    generator._seed_for_sample = lambda _seed: None
    generator._derive_sample_seed = lambda _sample_index, _attempt: None
    generator._select_template_spec = lambda: (generator.template_specs[0], 1.0)
    generator._mutate_text_generator_sections = lambda markdown, _ratio, merge_order: (markdown, 0)

    class _StubDataGenerator:
        @staticmethod
        def generate_markdown(template_id: str, template_spec: TemplateSpec) -> str:
            return "# Heading\n\n## Section\n\n```bash\nuv run main.py\n```"

        @staticmethod
        def pop_merge_order() -> list[str]:
            return ["command"]

        @staticmethod
        def pop_composition_metadata() -> dict:
            return {
                "document_shape": "technical_manual",
                "block_types": ["command"],
                "block_type_counts": {"command": 1},
                "section_count": 1,
            }

    generator.data_generator = _StubDataGenerator()
    monkeypatch.setattr(generator_module, "random_style", lambda _profile: generator_module.MarkdownStyle())
    monkeypatch.setattr(generator_module, "markdown_to_json_ast", lambda markdown_text: [{"raw": markdown_text}])

    class _StubRenderer:
        def __init__(self, _font_path, style):
            self.style = style

        def render(self, markdown_text: str):
            return Image.new("RGB", (320, 480), color=(255, 255, 255))

    monkeypatch.setattr(generator_module, "MarkdownRenderer", _StubRenderer)
    generator.font_paths = ["/tmp/dummy-font.ttf"]

    _image, metadata = generator.generate_single(sample_index=3)

    assert metadata["document_family"] == "technical"
    assert metadata["document_shape"] == "technical_manual"
    assert metadata["block_types"] == ["command"]
    assert metadata["block_type_counts"] == {"command": 1}
    assert metadata["section_count"] == 1
```

- [ ] **Step 2: Add failing test for prose-block mutation**

Append this test next to the existing text mutation tests:

```python
def test_text_section_typos_apply_to_rich_prose_blocks_only() -> None:
    generator = Generator.__new__(Generator)

    def fake_mutate(section_text: str, _ratio: float):
        return section_text.replace("Alpha", "A1pha"), 1 if "Alpha" in section_text else 0

    generator._mutate_similar_text = fake_mutate

    markdown = (
        "# Report\n\n"
        "## Paragraph\n"
        "Alpha paragraph content.\n\n"
        "## Command\n"
        "```bash\n"
        "echo Alpha\n"
        "```\n\n"
        "## Formula\n"
        "$$ Alpha = beta $$"
    )

    mutated, mutation_count = generator._mutate_text_generator_sections(
        markdown,
        0.2,
        ["paragraph", "command", "formula"],
    )

    assert "A1pha paragraph content." in mutated
    assert "echo Alpha" in mutated
    assert "$$ Alpha = beta $$" in mutated
    assert mutation_count == 1
```

- [ ] **Step 3: Run tests and confirm they fail**

Run:

```bash
uv run pytest tests/generator/test_dynamic_templates.py::test_generate_single_includes_document_structure_metadata tests/generator/test_dynamic_templates.py::test_text_section_typos_apply_to_rich_prose_blocks_only -q
```

Expected: metadata key failure and mutation-count failure.

- [ ] **Step 4: Update prose mutation block types**

In `src/generator/text_mutation.py`, add near the imports:

```python
PROSE_SECTION_TYPES = {"text", "paragraph", "bullet_list", "numbered_list", "checklist", "quote"}
```

Change this condition inside `mutate_text_generator_sections()`:

```python
        if section_type == "text":
```

to:

```python
        if section_type in PROSE_SECTION_TYPES:
```

- [ ] **Step 5: Read composition metadata in `Generator.generate_single()`**

In `src/generator/generator.py`, after:

```python
            merge_order = self.data_generator.pop_merge_order()
```

add:

```python
            if hasattr(self.data_generator, "pop_composition_metadata"):
                composition_metadata = self.data_generator.pop_composition_metadata()
            else:
                composition_metadata = {
                    "document_shape": selected_template.family,
                    "block_types": list(merge_order),
                    "block_type_counts": dict(Counter(merge_order)),
                    "section_count": len(merge_order),
                }
```

Initialize `composition_metadata` before the novelty loop with:

```python
        composition_metadata: Dict[str, Any] = {
            "document_shape": selected_template.family,
            "block_types": [],
            "block_type_counts": {},
            "section_count": 0,
        }
```

- [ ] **Step 6: Attach metadata fields**

In the final `metadata = { ... }` dict in `Generator.generate_single()`, add these keys after `template_family`:

```python
            "document_family": selected_template.family,
            "document_shape": composition_metadata.get("document_shape", selected_template.family),
            "block_types": list(composition_metadata.get("block_types", merge_order)),
            "block_type_counts": dict(composition_metadata.get("block_type_counts", Counter(merge_order))),
            "section_count": int(composition_metadata.get("section_count", len(merge_order))),
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest tests/generator/test_dynamic_templates.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add src/generator/generator.py src/generator/text_mutation.py tests/generator/test_dynamic_templates.py
git commit -m "[feat] Record document structure metadata"
```

Expected: commit succeeds with metadata and mutation changes.

---

### Task 4: Expand The Template Catalog

**Files:**
- Modify: `configs/generator/templates/default.yaml`
- Modify: `tests/generator/test_dynamic_templates.py`

- [ ] **Step 1: Add failing catalog coverage test**

Append this test near `test_template_catalog_builtin_default_limits_table_columns`:

```python
def test_default_template_catalog_exposes_document_families() -> None:
    catalog = TemplateCatalog()
    specs = catalog.all_specs()
    families = {spec.family for spec in specs}
    template_ids = {spec.template_id for spec in specs}

    assert len(specs) >= 10
    assert {"business", "technical", "academic", "operations", "forms"}.issubset(families)
    assert {
        "business_report",
        "meeting_minutes",
        "technical_manual",
        "api_reference",
        "academic_note",
        "release_note",
        "policy_document",
        "form_like",
        "table_heavy",
        "formula_heavy",
    }.issubset(template_ids)
    assert all(spec.mode == "sections" for spec in specs)
```

- [ ] **Step 2: Run catalog test and confirm it fails**

Run:

```bash
uv run pytest tests/generator/test_dynamic_templates.py::test_default_template_catalog_exposes_document_families -q
```

Expected: failure because the default catalog has only the current default template.

- [ ] **Step 3: Replace `configs/generator/templates/default.yaml`**

Replace the file content with:

```yaml
version: 3
templates:
  - id: business_report
    family: business
    complexity: 2
    weight: 1.2
    mode: sections
    aliases: [report, business-report]
    tags: [paragraphs, bullets, tables]
    blueprint:
      document_shape: business_report
      section_count: [4, 7]
      blocks_per_section: [1, 1]
      allowed_blocks: [paragraph, bullet_list, table, quote, rule]
      required_blocks: [paragraph, table]
      text:
        max_line_chars: 72
      table:
        rows: [2, 5]
        columns: [3, 5]

  - id: meeting_minutes
    family: operations
    complexity: 2
    weight: 1.1
    mode: sections
    aliases: [minutes, meeting]
    tags: [checklist, numbered, table]
    blueprint:
      document_shape: meeting_minutes
      section_count: [4, 6]
      blocks_per_section: [1, 1]
      allowed_blocks: [paragraph, checklist, numbered_list, table, quote]
      required_blocks: [checklist, numbered_list]
      text:
        max_line_chars: 68
      table:
        rows: [2, 4]
        columns: [3, 4]

  - id: technical_manual
    family: technical
    complexity: 3
    weight: 1.1
    mode: sections
    aliases: [manual, technical-manual]
    tags: [commands, code, procedures]
    blueprint:
      document_shape: technical_manual
      section_count: [5, 8]
      blocks_per_section: [1, 1]
      allowed_blocks: [paragraph, numbered_list, command, code, table, checklist]
      required_blocks: [numbered_list, command]
      text:
        max_line_chars: 76
      table:
        rows: [2, 5]
        columns: [3, 5]

  - id: api_reference
    family: technical
    complexity: 3
    weight: 1.0
    mode: sections
    aliases: [api, api-reference]
    tags: [api, commands, code]
    blueprint:
      document_shape: api_reference
      section_count: [4, 7]
      blocks_per_section: [1, 1]
      allowed_blocks: [paragraph, command, code, table, bullet_list]
      required_blocks: [command, code]
      text:
        max_line_chars: 78
      table:
        rows: [2, 4]
        columns: [3, 5]

  - id: academic_note
    family: academic
    complexity: 3
    weight: 1.0
    mode: sections
    aliases: [academic, research-note]
    tags: [formulas, quote, image]
    blueprint:
      document_shape: academic_note
      section_count: [4, 7]
      blocks_per_section: [1, 1]
      allowed_blocks: [paragraph, formula, quote, image, bullet_list]
      required_blocks: [formula]
      text:
        max_line_chars: 74

  - id: release_note
    family: technical
    complexity: 2
    weight: 0.9
    mode: sections
    aliases: [release, changelog]
    tags: [bullets, checklist, code]
    blueprint:
      document_shape: release_note
      section_count: [4, 6]
      blocks_per_section: [1, 1]
      allowed_blocks: [paragraph, bullet_list, checklist, code, rule]
      required_blocks: [bullet_list, checklist]
      text:
        max_line_chars: 72

  - id: policy_document
    family: business
    complexity: 3
    weight: 0.9
    mode: sections
    aliases: [policy]
    tags: [numbered, quote, checklist]
    blueprint:
      document_shape: policy_document
      section_count: [5, 8]
      blocks_per_section: [1, 1]
      allowed_blocks: [paragraph, numbered_list, quote, checklist, rule]
      required_blocks: [numbered_list]
      text:
        max_line_chars: 70

  - id: form_like
    family: forms
    complexity: 2
    weight: 1.0
    mode: sections
    aliases: [form, form-like]
    tags: [tables, checklist]
    blueprint:
      document_shape: form_like
      section_count: [3, 5]
      blocks_per_section: [1, 1]
      allowed_blocks: [table, checklist, paragraph, rule]
      required_blocks: [table, checklist]
      text:
        max_line_chars: 60
      table:
        rows: [2, 6]
        columns: [3, 4]

  - id: table_heavy
    family: forms
    complexity: 4
    weight: 0.8
    mode: sections
    aliases: [tables, table-heavy]
    tags: [tables]
    blueprint:
      document_shape: table_heavy
      section_count: [4, 7]
      blocks_per_section: [1, 1]
      allowed_blocks: [table, paragraph, rule]
      required_blocks: [table]
      text:
        max_line_chars: 64
      table:
        rows: [3, 7]
        columns: [4, 6]

  - id: formula_heavy
    family: academic
    complexity: 4
    weight: 0.8
    mode: sections
    aliases: [formulas, formula-heavy]
    tags: [formulas]
    blueprint:
      document_shape: formula_heavy
      section_count: [4, 7]
      blocks_per_section: [1, 1]
      allowed_blocks: [formula, paragraph, quote, rule]
      required_blocks: [formula]
      text:
        max_line_chars: 72

  - id: default
    family: sections
    complexity: 2
    weight: 0.7
    mode: sections
    aliases: [sections]
    tags: [legacy-compatible]
    blueprint:
      document_shape: sections
      section_count: [3, 5]
      blocks_per_section: [1, 1]
      allowed_blocks: [paragraph, table, formula]
      required_blocks: [paragraph]
      text:
        max_line_chars: 72
      table:
        rows: [2, 4]
        columns: [3, 4]
```

- [ ] **Step 4: Run catalog and generation tests**

Run:

```bash
uv run pytest tests/generator/test_document_blocks.py tests/generator/test_dynamic_templates.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add configs/generator/templates/default.yaml tests/generator/test_dynamic_templates.py
git commit -m "[add] Seed diverse markdown document templates"
```

Expected: commit succeeds with template catalog expansion.

---

### Task 5: Update Generation Docs And Run Final Verification

**Files:**
- Modify: `docs/generation.md`

- [ ] **Step 1: Update docs with structure-diversity recipe**

In `docs/generation.md`, add this subsection under `## Practical Recipes` before the existing recipe list:

````markdown
### Structure-diverse documents

The default template catalog now includes multiple document families such as `business`, `technical`, `academic`, `operations`, `forms`, and legacy-compatible `sections`.

Use coverage targets when you want family balance:

```bash
uv run main.py generate \
  --lang "ko" \
  --size 1000 \
  --style-profile aggressive \
  --coverage-target business=0.2 \
  --coverage-target technical=0.25 \
  --coverage-target academic=0.2 \
  --coverage-target operations=0.15 \
  --coverage-target forms=0.2 \
  --novelty-threshold 0.92 \
  --novelty-max-attempts 6
```

Each sample records `document_family`, `document_shape`, `block_types`, `block_type_counts`, and `section_count` in metadata so generated diversity can be audited from `metadata.jsonl`.
````

Also update the existing “Template Catalog Format” prose so it states `mode: sections` is the supported block-composition mode, and remove claims that unsupported modes are active if those claims remain in the file.

- [ ] **Step 2: Run documentation diff check**

Run:

```bash
git diff -- docs/generation.md
```

Expected: diff shows only the new structure-diversity docs and any corrected mode wording.

- [ ] **Step 3: Run focused pytest suite**

Run:

```bash
uv run pytest tests/generator/test_document_blocks.py tests/generator/test_dynamic_templates.py tests/generator/test_phase4_generator_helpers.py tests/test_wave2_cli_pipeline_helpers.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run compile check**

Run:

```bash
uv run python -m compileall -q src tests
```

Expected: command exits with status 0 and no output.

- [ ] **Step 5: Run a small generation smoke test**

Run:

```bash
uv run main.py generate \
  --lang en \
  --size 3 \
  --seed 42 \
  --markdown-renderer pil \
  --style-profile aggressive \
  --output-dir /private/tmp/synthetic-ocr-diversity-smoke
```

Expected: command exits with status 0 and writes generated images plus metadata under `/private/tmp/synthetic-ocr-diversity-smoke`.

- [ ] **Step 6: Inspect smoke metadata keys**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/private/tmp/synthetic-ocr-diversity-smoke")
rows = []
for path in root.rglob("metadata.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

assert rows, "no metadata rows found"
for row in rows:
    assert "document_family" in row
    assert "document_shape" in row
    assert "block_types" in row
    assert "block_type_counts" in row
    assert "section_count" in row
print({"rows": len(rows), "families": sorted({row["document_family"] for row in rows})})
PY
```

Expected: prints a dictionary with `rows` greater than zero and at least one family.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add docs/generation.md
git commit -m "[docs] Document structure-diverse generation"
```

Expected: commit succeeds with documentation only.

---

## Final Verification

Run the full focused verification set after all task commits:

```bash
uv run pytest tests/generator/test_document_blocks.py tests/generator/test_dynamic_templates.py tests/generator/test_phase4_generator_helpers.py tests/test_wave2_cli_pipeline_helpers.py -q
uv run python -m compileall -q src tests
git status --short
```

Expected:

- pytest exits with status 0.
- compileall exits with status 0 and no output.
- `git status --short` is empty.

If a generated smoke-test directory under `/private/tmp/synthetic-ocr-diversity-smoke` exists, leave it outside the repo.
