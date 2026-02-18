"""Markdown Generator Module for synthetic OCR markdown image generation.

This module provides comprehensive markdown document generation capabilities including:
- Various markdown elements (headers, paragraphs, lists, code blocks, tables, blockquotes)
- Multiple markdown templates (readme, technical_doc, blog_post, api_doc, tutorial)
- Layout variations (backgrounds, noise effects)
- Ground truth format with raw markdown text and rendered image
"""

import logging
import random
import tempfile
import importlib
import re
from collections import Counter, deque
from difflib import SequenceMatcher
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from character_similarity import find_similar_chars
from generator.base import BaseGenerator
from generator.data_provider import DataProvider
from utils import markdown_to_json_ast, read_json

logger = logging.getLogger(__name__)


DEFAULT_NOVELTY_WINDOW = 80
DEFAULT_NOVELTY_THRESHOLD = 0.95
DEFAULT_NOVELTY_MAX_ATTEMPTS = 4
DEFAULT_BLUEPRINT_MAX_TOTAL_LINES = 115
DEFAULT_BLUEPRINT_MAX_PARAGRAPH_CHARS = 220
A4_MAX_WIDTH_PX = 2480
A4_MAX_HEIGHT_PX = 3508

_MARKDOWN_IMAGE_PATTERN = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)$")
_MARKDOWN_FORMULA_PATTERN = re.compile(r"^\$\$\s*(?P<formula>.+?)\s*\$\$$")


def parse_markdown_image_line(line: str) -> Optional[Tuple[str, str]]:
    match = _MARKDOWN_IMAGE_PATTERN.match(line.strip())
    if not match:
        return None
    return match.group("alt").strip(), match.group("src").strip()


def parse_markdown_formula_line(line: str) -> Optional[str]:
    match = _MARKDOWN_FORMULA_PATTERN.match(line.strip())
    if not match:
        return None
    return match.group("formula").strip()

_DEFAULT_TEMPLATE_DIR = Path("configs") / "generator" / "templates"
_DEFAULT_LEGACY_TEMPLATE_SPECS: List[Dict[str, Any]] = [
    {
        "id": "readme",
        "family": "legacy",
        "complexity": 1,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "readme",
    },
    {
        "id": "technical_doc",
        "family": "legacy",
        "complexity": 2,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "technical_doc",
    },
    {
        "id": "blog_post",
        "family": "legacy",
        "complexity": 1,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "blog_post",
    },
    {
        "id": "api_doc",
        "family": "legacy",
        "complexity": 2,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "api_doc",
    },
    {
        "id": "tutorial",
        "family": "legacy",
        "complexity": 2,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "tutorial",
    },
    {
        "id": "changelog",
        "family": "legacy",
        "complexity": 2,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "changelog",
    },
    {
        "id": "meeting_notes",
        "family": "legacy",
        "complexity": 2,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "meeting_notes",
    },
    {
        "id": "incident_report",
        "family": "legacy",
        "complexity": 3,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "incident_report",
    },
    {
        "id": "release_note",
        "family": "legacy",
        "complexity": 2,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "release_note",
    },
    {
        "id": "compliance_checklist",
        "family": "legacy",
        "complexity": 3,
        "weight": 1.0,
        "mode": "legacy",
        "legacy_method": "compliance_checklist",
    },
]


def _canonicalize_template_ref(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass
class TemplateSpec:
    template_id: str
    family: str = "legacy"
    complexity: int = 1
    weight: float = 1.0
    mode: str = "legacy"
    legacy_method: Optional[str] = None
    blueprint: Optional[Dict[str, Any]] = None
    aliases: Optional[List[str]] = None
    version: str = "1"
    source: str = "builtin"
    tags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.blueprint is None:
            self.blueprint = {}
        if self.aliases is None:
            self.aliases = []
        if self.tags is None:
            self.tags = []

    def refs(self) -> List[str]:
        refs: List[str] = [_canonicalize_template_ref(self.template_id)]
        for alias in self.aliases or []:
            normalized = _canonicalize_template_ref(alias)
            if normalized and normalized not in refs:
                refs.append(normalized)
        return refs


class TemplateCatalog:
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else _DEFAULT_TEMPLATE_DIR
        self.templates: Dict[str, TemplateSpec] = {}
        self.alias_to_id: Dict[str, str] = {}
        self._loaded = False

    @staticmethod
    def _extract_entries(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            templates = data.get("templates")
            if isinstance(templates, list):
                return [item for item in templates if isinstance(item, dict)]
            if "id" in data:
                return [data]
        return []

    @staticmethod
    def _coerce_spec(raw: Dict[str, Any], source: str) -> Optional[TemplateSpec]:
        template_id = _canonicalize_template_ref(str(raw.get("id", "")))
        if not template_id:
            return None

        mode = str(raw.get("mode", "legacy")).strip().lower()
        if mode in {"dynamic", "procedural"}:
            mode = "blueprint"
        if mode not in {"legacy", "blueprint"}:
            logger.warning("Unknown template mode '%s' for '%s'. Falling back to legacy.", mode, template_id)
            mode = "legacy"

        family = str(raw.get("family") or mode).strip().lower() or mode

        try:
            complexity = int(raw.get("complexity", 1))
        except (TypeError, ValueError):
            complexity = 1
        complexity = max(1, min(5, complexity))

        try:
            weight = float(raw.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        weight = max(0.01, weight)

        aliases_raw = raw.get("aliases", [])
        aliases: List[str] = []
        if isinstance(aliases_raw, list):
            for item in aliases_raw:
                if isinstance(item, str) and item.strip():
                    aliases.append(item.strip())

        tags_raw = raw.get("tags", [])
        tags: List[str] = []
        if isinstance(tags_raw, list):
            for item in tags_raw:
                if isinstance(item, str) and item.strip():
                    tags.append(item.strip())

        blueprint_raw = raw.get("blueprint")
        blueprint: Dict[str, Any] = blueprint_raw if isinstance(blueprint_raw, dict) else {}

        legacy_method_raw = raw.get("legacy_method")
        legacy_method = str(legacy_method_raw).strip() if isinstance(legacy_method_raw, str) else None
        if mode == "legacy" and not legacy_method:
            legacy_method = template_id

        return TemplateSpec(
            template_id=template_id,
            family=family,
            complexity=complexity,
            weight=weight,
            mode=mode,
            legacy_method=legacy_method,
            blueprint=blueprint,
            aliases=aliases,
            version=str(raw.get("version", "1")),
            source=source,
            tags=tags,
        )

    def load(self) -> None:
        template_by_id: Dict[str, TemplateSpec] = {}
        for item in _DEFAULT_LEGACY_TEMPLATE_SPECS:
            spec = self._coerce_spec(item, source="builtin")
            if spec is not None:
                template_by_id[spec.template_id] = spec

        if self.config_dir.exists():
            import yaml

            for yaml_path in sorted(self.config_dir.glob("*.y*ml")):
                try:
                    with open(yaml_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except Exception as exc:
                    logger.warning("Failed to read template catalog '%s': %s", yaml_path, exc)
                    continue

                for raw in self._extract_entries(data):
                    spec = self._coerce_spec(raw, source=str(yaml_path))
                    if spec is not None:
                        template_by_id[spec.template_id] = spec

        self.templates = template_by_id
        self.alias_to_id = {}
        for template_id, spec in self.templates.items():
            for ref in spec.refs():
                self.alias_to_id[ref] = template_id

        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def all_specs(self) -> List[TemplateSpec]:
        self._ensure_loaded()
        return [self.templates[key] for key in sorted(self.templates)]

    def get(self, template_ref: str) -> Optional[TemplateSpec]:
        self._ensure_loaded()
        template_id = self.alias_to_id.get(_canonicalize_template_ref(template_ref))
        if not template_id:
            return None
        return self.templates.get(template_id)

    def resolve(
        self,
        template: Optional[str],
        template_family: Optional[str],
        min_complexity: Optional[int],
        max_complexity: Optional[int],
    ) -> List[TemplateSpec]:
        self._ensure_loaded()

        if template:
            resolved = self.get(template)
            if resolved is not None:
                return [resolved]
            logger.warning("Unknown template '%s'; applying filters over full catalog.", template)

        candidates = self.all_specs()
        if template_family:
            family = template_family.strip().lower()
            candidates = [spec for spec in candidates if spec.family == family]
        if min_complexity is not None:
            candidates = [spec for spec in candidates if spec.complexity >= min_complexity]
        if max_complexity is not None:
            candidates = [spec for spec in candidates if spec.complexity <= max_complexity]

        if not candidates:
            logger.warning("Template filters returned no candidates; falling back to full catalog.")
            return self.all_specs()

        return candidates


def parse_coverage_targets(raw: Any) -> Dict[str, float]:
    if raw is None:
        return {}

    parsed: Dict[str, float] = {}

    def put(key: str, value: Any) -> None:
        normalized = key.strip().lower()
        if not normalized:
            return
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return
        parsed[normalized] = max(0.0, min(1.0, ratio))

    if isinstance(raw, dict):
        for key, value in raw.items():
            put(str(key), value)
        return parsed

    items: List[str] = []
    if isinstance(raw, str):
        items.extend(token for token in raw.split(",") if token.strip())
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, str):
                items.extend(token for token in item.split(",") if token.strip())

    for item in items:
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            continue
        put(key, value)

    return parsed


@dataclass
class MarkdownStyle:
    """Markdown rendering style options."""
    # Layout
    margin_top: int = 40
    margin_bottom: int = 40
    margin_left: int = 40
    margin_right: int = 40
    content_width: int = 600
    line_spacing: float = 1.5

    # Typography
    h1_font_size: int = 28
    h2_font_size: int = 22
    h3_font_size: int = 18
    body_font_size: int = 14
    code_font_size: int = 12

    # Colors
    text_color: Tuple[int, int, int] = (33, 33, 33)
    h1_color: Tuple[int, int, int] = (0, 0, 0)
    h2_color: Tuple[int, int, int] = (50, 50, 50)
    h3_color: Tuple[int, int, int] = (70, 70, 70)
    link_color: Tuple[int, int, int] = (0, 102, 204)
    code_bg_color: Tuple[int, int, int] = (245, 245, 245)
    code_text_color: Tuple[int, int, int] = (0, 0, 0)
    blockquote_color: Tuple[int, int, int] = (100, 100, 100)
    blockquote_border_color: Tuple[int, int, int] = (200, 200, 200)

    # Background
    background_color: Tuple[int, int, int] = (255, 255, 255)

    # Effects
    add_noise: bool = True
    add_blur: bool = False
    add_contrast: bool = False


class MarkdownDataGenerator:
    """Generates markdown content for various template types."""

    def __init__(self, lang: str = "ko", data_provider: Optional[DataProvider] = None):
        self.lang = lang
        self.data = data_provider or DataProvider(lang=lang)

    def generate_markdown(
        self,
        template_id: str = "readme",
        template_spec: Optional[TemplateSpec] = None,
    ) -> str:
        if template_spec and template_spec.mode == "blueprint":
            return self._generate_from_blueprint(template_spec.template_id, template_spec.blueprint or {})

        legacy_method_value = template_id
        if template_spec and template_spec.legacy_method:
            legacy_method_value = template_spec.legacy_method
        legacy_method = legacy_method_value.strip().lower()
        gen_func = getattr(self, f"_generate_{legacy_method}", None)
        if callable(gen_func):
            return str(gen_func())

        logger.warning("Unknown legacy template method '%s'. Falling back to readme.", legacy_method)
        return self._generate_readme()

    @staticmethod
    def _slugify(text: str, max_parts: int = 3) -> str:
        chunks: List[str] = []
        for token in text.lower().replace("_", "-").split():
            cleaned = "".join(ch for ch in token if ch.isalnum() or ch == "-").strip("-")
            if cleaned:
                chunks.append(cleaned)
            if len(chunks) >= max_parts:
                break
        return "-".join(chunks) if chunks else "sample-app"

    def _project_slug(self) -> str:
        title = self.data.title()
        return self._slugify(title)

    def _sample_requirements(self, count: int = 3) -> List[str]:
        items = set()
        while len(items) < count:
            items.add(self.data.requirement_line())
        return list(items)

    def _sample_config_lines(self, count: int = 3) -> List[str]:
        items = set()
        while len(items) < count:
            items.add(self.data.config_line())
        return list(items)

    @staticmethod
    def _to_config_entry(line: str) -> Tuple[str, str]:
        if ":" in line:
            key, value = line.split(":", 1)
            return key.strip(), value.strip()
        token = "".join(ch for ch in line.lower().replace(" ", "_") if ch.isalnum() or ch == "_")
        return token or "option", "true"

    @staticmethod
    def _to_runtime_version(requirement: str) -> Tuple[str, str]:
        if ">=" in requirement:
            name, version = requirement.split(">=", 1)
            return name.strip(), version.strip()
        if "==" in requirement:
            name, version = requirement.split("==", 1)
            return name.strip(), version.strip()
        words = requirement.split()
        if not words:
            return "Runtime", "1.0"
        if len(words) == 1:
            return words[0], "1.0"
        return words[0], words[-1]

    @staticmethod
    def _coerce_positive_int(value: Any, default: int, lower: int, upper: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(lower, min(upper, parsed))

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        normalized = " ".join(text.split())
        if max_chars <= 0 or len(normalized) <= max_chars:
            return normalized
        clipped = normalized[:max_chars].rstrip()
        split_at = clipped.rfind(" ")
        if split_at >= int(max_chars * 0.6):
            clipped = clipped[:split_at].rstrip()
        return clipped.rstrip(" ,;:") + "..."

    @staticmethod
    def _coerce_int_range(value: Any, default_min: int, default_max: int) -> Tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                lower = int(value[0])
                upper = int(value[1])
            except (TypeError, ValueError):
                return default_min, default_max
            if lower > upper:
                lower, upper = upper, lower
            return lower, upper

        if isinstance(value, int):
            return value, value

        return default_min, default_max

    @staticmethod
    def _coerce_probability(value: Any, default: float) -> float:
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, ratio))

    def _build_blueprint_table(self, min_rows: int, max_rows: int) -> List[str]:
        template_name = random.choice(["invoice", "schedule", "product", "contact"])
        headers = self.data.headers(template_name)
        row_count = random.randint(min_rows, max_rows)
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for _ in range(row_count):
            row_values: List[str] = []
            for index, _ in enumerate(headers):
                if index == 0:
                    row_values.append(self.data.product_name())
                elif index == 1:
                    row_values.append(str(self.data.quantity()))
                elif index == 2:
                    row_values.append(self.data.format_currency(self.data.random_price()))
                else:
                    row_values.append(self.data.feature())
            lines.append("| " + " | ".join(row_values) + " |")
        return lines

    def _build_blueprint_code(self) -> List[str]:
        code_kind = random.choice(["python", "bash", "json", "yaml"])
        if code_kind == "python":
            return [
                "```python",
                self.data.code_comment(),
                "def run():",
                f"    return '{self.data.word()}'",
                "```",
            ]
        if code_kind == "bash":
            return [
                "```bash",
                self.data.install_command(package_name=self._project_slug()),
                self.data.usage_command(entrypoint="main.py"),
                "```",
            ]
        if code_kind == "json":
            return [
                "```json",
                "{",
                f"  \"id\": \"{self.data.word()}-{random.randint(100, 999)}\",",
                f"  \"status\": \"{random.choice(['ok', 'pending', 'failed'])}\"",
                "}",
                "```",
            ]
        return [
            "```yaml",
            "config:",
            f"  {self.data.config_line()}",
            f"  {self.data.config_line()}",
            "```",
        ]

    @staticmethod
    def _normalize_blueprint_block_type(block_type: str) -> str:
        normalized = block_type.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "bullet": "bullet_points",
            "bullets": "bullet_points",
            "bullet_list": "bullet_points",
            "bulletlist": "bullet_points",
            "points": "bullet_points",
            "ordered_list": "numbered_list",
            "ordered": "numbered_list",
            "numbered": "numbered_list",
            "subheading": "subtitle",
            "heading": "subtitle",
            "table_of_contents": "contents",
            "toc": "contents",
            "equation": "formula",
            "math": "formula",
            "latex": "formula",
            "figure": "image",
            "images": "image",
            "img": "image",
        }
        return aliases.get(normalized, normalized)

    def _build_blueprint_formula(self) -> List[str]:
        variable = random.choice(["x", "y", "n", "t", "k"])
        expressions = [
            f"f({variable}) = {random.randint(2, 9)}{variable} + {random.randint(1, 20)}",
            "a^2 + b^2 = c^2",
            "E = mc^2",
            "\\frac{n(n+1)}{2}",
            "P(A|B) = \\frac{P(B|A)P(A)}{P(B)}",
            "\\sum_{i=1}^n i = \\frac{n(n+1)}{2}",
        ]
        expression = random.choice(expressions)
        return [f"$$ {expression} $$"]

    def _build_blueprint_image(self) -> List[str]:
        alt_text = self.data.title()
        image_ref = f"placeholder://{self._slugify(alt_text, max_parts=6)}-{random.randint(100, 999)}"
        return [f"![{alt_text}]({image_ref})", f"*Figure: {self.data.sentence()}*"]

    def _build_blueprint_contents(
        self,
        section_titles: Optional[List[str]],
        list_item_range: Tuple[int, int],
        max_title_chars: int,
    ) -> List[str]:
        lower_items, upper_items = list_item_range
        if section_titles:
            entries = list(section_titles)
            random.shuffle(entries)
            max_items = min(len(entries), upper_items)
            min_items = min(len(entries), max(1, lower_items))
            if max_items < min_items:
                max_items = min_items
            item_count = random.randint(min_items, max_items)
            selected_titles = entries[:item_count]
        else:
            item_count = random.randint(lower_items, upper_items)
            selected_titles = [self.data.title() for _ in range(item_count)]

        lines = ["## Contents"]
        for index, title in enumerate(selected_titles, start=1):
            clipped_title = self._clip_text(title, max_title_chars)
            anchor = self._slugify(clipped_title, max_parts=8)
            lines.append(f"{index}. [{clipped_title}](#{anchor})")
        return lines

    def _build_blueprint_block(
        self,
        block_type: str,
        list_item_range: Tuple[int, int],
        table_row_range: Tuple[int, int],
        section_titles: Optional[List[str]] = None,
        max_paragraph_chars: int = DEFAULT_BLUEPRINT_MAX_PARAGRAPH_CHARS,
    ) -> List[str]:
        block_type = self._normalize_blueprint_block_type(block_type)
        lower_items, upper_items = list_item_range
        lower_rows, upper_rows = table_row_range

        if block_type == "title":
            return ["## " + self._clip_text(self.data.title(), 90)]

        if block_type == "subtitle":
            return ["### " + self._clip_text(self.data.title(), 90)]

        if block_type == "contents":
            return self._build_blueprint_contents(section_titles, list_item_range, max_title_chars=80)

        if block_type == "bullet_points":
            item_count = random.randint(lower_items, upper_items)
            return [f"- {item}" for item in self.data.features(item_count)]

        if block_type == "numbered_list":
            item_count = random.randint(lower_items, upper_items)
            return [f"{idx}. {item}" for idx, item in enumerate(self.data.features(item_count), start=1)]

        if block_type == "checklist":
            item_count = random.randint(lower_items, upper_items)
            lines: List[str] = []
            for item in self.data.features(item_count):
                marker = "x" if random.random() < 0.5 else " "
                lines.append(f"- [{marker}] {item}")
            return lines

        if block_type == "table":
            return self._build_blueprint_table(lower_rows, upper_rows)

        if block_type == "formula":
            return self._build_blueprint_formula()

        if block_type == "image":
            return self._build_blueprint_image()

        if block_type == "code":
            return self._build_blueprint_code()

        if block_type == "quote":
            return ["> " + self._clip_text(self.data.paragraph(), max_paragraph_chars)]

        if block_type == "rule":
            return ["---"]

        if block_type == "command":
            return ["`" + self.data.usage_command(entrypoint="main.py") + "`"]

        return [self._clip_text(self.data.paragraph(), max_paragraph_chars)]

    def _generate_from_blueprint(self, template_id: str, blueprint: Dict[str, Any]) -> str:
        section_min, section_max = self._coerce_int_range(blueprint.get("section_count"), 2, 5)
        block_min, block_max = self._coerce_int_range(blueprint.get("blocks_per_section"), 1, 3)
        paragraph_min, paragraph_max = self._coerce_int_range(blueprint.get("paragraphs_per_section"), 1, 2)
        list_item_range = self._coerce_int_range(blueprint.get("list_items"), 2, 5)
        table_row_range = self._coerce_int_range(blueprint.get("table_rows"), 2, 4)
        max_total_lines = self._coerce_positive_int(
            blueprint.get("max_total_lines"),
            DEFAULT_BLUEPRINT_MAX_TOTAL_LINES,
            40,
            260,
        )
        max_paragraph_chars = self._coerce_positive_int(
            blueprint.get("max_paragraph_chars"),
            DEFAULT_BLUEPRINT_MAX_PARAGRAPH_CHARS,
            80,
            900,
        )

        heading_level = int(blueprint.get("section_heading_level", 2))
        heading_level = max(2, min(3, heading_level))

        frontmatter_probability = self._coerce_probability(blueprint.get("frontmatter_probability"), 0.0)
        section_rule_probability = self._coerce_probability(blueprint.get("section_rule_probability"), 0.2)

        allowed_blocks_raw = blueprint.get("allowed_blocks")
        if isinstance(allowed_blocks_raw, list) and allowed_blocks_raw:
            allowed_blocks = []
            for item in allowed_blocks_raw:
                normalized = self._normalize_blueprint_block_type(str(item))
                if normalized and normalized not in allowed_blocks:
                    allowed_blocks.append(normalized)
        else:
            allowed_blocks = [
                "subtitle",
                "contents",
                "bullet_points",
                "numbered_list",
                "checklist",
                "table",
                "formula",
                "image",
                "code",
                "quote",
                "command",
                "rule",
            ]

        required_raw = blueprint.get("required_blocks")
        required_blocks: List[str] = []
        if isinstance(required_raw, list):
            for item in required_raw:
                normalized = self._normalize_blueprint_block_type(str(item))
                if normalized:
                    required_blocks.append(normalized)
        pending_required = [item for item in required_blocks if item in allowed_blocks]

        title_prefix = str(blueprint.get("title_prefix") or "Document")
        lines: List[str] = []

        def _append_lines(chunk: List[str], trailing_blank: bool = False) -> bool:
            required_slots = len(chunk) + (1 if trailing_blank else 0)
            if len(lines) + required_slots > max_total_lines:
                return False
            lines.extend(chunk)
            if trailing_blank:
                lines.append("")
            return True

        if random.random() < frontmatter_probability:
            _append_lines(
                [
                    "---",
                    f"template_id: {template_id}",
                    f"generated_at: {self.data.date()}",
                    "---",
                    "",
                ],
                trailing_blank=False,
            )

        root_heading = self._clip_text(f"{title_prefix}: {self.data.title()}", 110)
        if not _append_lines([f"# {root_heading}"], trailing_blank=True):
            return "\n".join(lines)

        intro_paragraph = self._clip_text(self.data.paragraph(), max_paragraph_chars)
        if not _append_lines([intro_paragraph], trailing_blank=True):
            return "\n".join(lines)

        section_count = random.randint(section_min, section_max)
        section_titles = [self.data.title() for _ in range(section_count)]
        section_prefix = "#" * heading_level
        stop_generation = False

        for section_idx in range(section_count):
            section_title = self._clip_text(section_titles[section_idx], 100)
            if not _append_lines([f"{section_prefix} {section_title}"], trailing_blank=True):
                stop_generation = True
                break

            paragraph_count = random.randint(paragraph_min, paragraph_max)
            for _ in range(paragraph_count):
                paragraph = self._clip_text(self.data.paragraph(), max_paragraph_chars)
                if not _append_lines([paragraph], trailing_blank=True):
                    stop_generation = True
                    break

            if stop_generation:
                break

            block_count = random.randint(block_min, block_max)
            for _ in range(block_count):
                if pending_required:
                    block_type = pending_required.pop(0)
                else:
                    block_type = random.choice(allowed_blocks)

                block_lines = self._build_blueprint_block(
                    block_type,
                    list_item_range,
                    table_row_range,
                    section_titles=section_titles,
                    max_paragraph_chars=max_paragraph_chars,
                )
                if not _append_lines(block_lines, trailing_blank=True):
                    stop_generation = True
                    break

            if stop_generation:
                break

            if section_idx < section_count - 1 and random.random() < section_rule_probability:
                if not _append_lines(["---"], trailing_blank=True):
                    stop_generation = True
                    break

        if pending_required and not stop_generation:
            for block_type in pending_required:
                block_lines = self._build_blueprint_block(
                    block_type,
                    list_item_range,
                    table_row_range,
                    section_titles=section_titles,
                    max_paragraph_chars=max_paragraph_chars,
                )
                if not _append_lines(block_lines, trailing_blank=True):
                    break

        return "\n".join(lines[:max_total_lines])

    def _generate_readme(self) -> str:
        title = self.data.title()
        project_slug = self._slugify(title)
        install_command = self.data.install_command(package_name=project_slug)
        usage_command = self.data.usage_command(entrypoint="main.py")
        lines = [
            f"# {title}",
            "",
            self.data.paragraph(),
            "",
            "## " + ("기능" if self.lang == "ko" else "Features"),
            "",
        ]

        # Add feature list
        num_features = random.randint(3, 5)
        for feature in self.data.features(num_features):
            lines.append(f"- {feature}")
        lines.append("")

        # Add installation section
        lines.extend([
            "## " + ("설치" if self.lang == "ko" else "Installation"),
            "",
            "```bash",
            install_command,
            "```",
            "",
        ])

        # Add usage section
        lines.extend([
            "## " + ("사용법" if self.lang == "ko" else "Usage"),
            "",
            "```python",
            self.data.code_comment(),
            f"from {project_slug.replace('-', '_')} import Client",
            "",
            "client = Client()",
            "result = client.run()",
            "```",
            "",
            "```bash",
            usage_command,
            "```",
            "",
        ])

        # Add quote
        lines.extend([
            "> " + self.data.paragraph(),
            "",
        ])

        return "\n".join(lines)

    def _generate_technical_doc(self) -> str:
        title = self.data.title()
        requirements = self._sample_requirements(random.randint(3, 5))
        cfg_lines = self._sample_config_lines(3)
        lines = [
            f"# {title}",
            "",
            "## " + ("개요" if self.lang == "ko" else "Overview"),
            "",
            self.data.paragraph(),
            "",
            "## " + ("요구사항" if self.lang == "ko" else "Requirements"),
            "",
        ]

        # Add requirements list
        for i, req in enumerate(requirements, 1):
            lines.append(f"{i}. {req}")
        lines.append("")

        # Add table
        lines.extend([
            "## " + ("지원 버전" if self.lang == "ko" else "Supported Versions"),
            "",
            "| " + ("버전" if self.lang == "ko" else "Version") + " | " + ("상태" if self.lang == "ko" else "Status") + " |",
            "|--------|--------|",
            "| 1.0.x | " + ("지원됨" if self.lang == "ko" else "Supported") + " |",
            "| 2.0.x | " + ("지원됨" if self.lang == "ko" else "Supported") + " |",
            "| 3.0.x | " + ("개발중" if self.lang == "ko" else "In Development") + " |",
            "",
        ])

        # Add code block
        lines.extend([
            "## " + ("설정" if self.lang == "ko" else "Configuration"),
            "",
            "```yaml",
            "config:",
            f"  {cfg_lines[0]}",
            f"  {cfg_lines[1]}",
            f"  {cfg_lines[2]}",
            "```",
            "",
        ])

        return "\n".join(lines)

    def _generate_blog_post(self) -> str:
        title = self.data.title()
        date = self.data.date()

        lines = [
            f"# {title}",
            "",
            "*" + ("작성일" if self.lang == "ko" else "Published") + f": {date}*",
            "",
            "---",
            "",
            self.data.paragraph(),
            "",
            "## " + ("주요 내용" if self.lang == "ko" else "Key Points"),
            "",
        ]

        # Add bullet points
        for feature in self.data.features(3):
            lines.append(f"- **{feature}**: " + self.data.paragraph()[:50] + "...")
        lines.append("")

        # Add blockquote
        lines.extend([
            "> " + ("중요" if self.lang == "ko" else "Important") + ": " + self.data.paragraph(),
            "",
        ])

        # Add inline code
        lines.extend([
            ("이 기능은 " if self.lang == "ko" else "This feature uses ") + "`config.yaml`" + (" 파일을 사용합니다." if self.lang == "ko" else " file."),
            "",
        ])

        # Add link
        lines.extend([
            ("자세한 내용은 " if self.lang == "ko" else "For more details, see ") + "[" + ("공식 문서" if self.lang == "ko" else "official docs") + "](https://example.com)" + ("를 참조하세요." if self.lang == "ko" else "."),
            "",
        ])

        return "\n".join(lines)

    def _generate_api_doc(self) -> str:
        endpoint_get = self.data.api_endpoint()
        endpoint_post = self.data.api_endpoint()
        if endpoint_post == endpoint_get:
            endpoint_post = endpoint_get.rstrip("s") + "s"

        user_one = self.data.name()
        user_two = self.data.name()
        user_email = self.data.email()
        page_name = "페이지 번호" if self.lang == "ko" else "Page number"
        limit_name = "페이지당 항목 수" if self.lang == "ko" else "Items per page"
        lines = [
            "# API " + ("레퍼런스" if self.lang == "ko" else "Reference"),
            "",
            "## " + ("엔드포인트" if self.lang == "ko" else "Endpoints"),
            "",
            f"### GET {endpoint_get}",
            "",
            ("사용자 목록을 조회합니다." if self.lang == "ko" else "Retrieve a list of users."),
            "",
            "**" + ("파라미터" if self.lang == "ko" else "Parameters") + ":**",
            "",
            "| " + ("이름" if self.lang == "ko" else "Name") + " | " + ("타입" if self.lang == "ko" else "Type") + " | " + ("설명" if self.lang == "ko" else "Description") + " |",
            "|------|------|-------------|",
            f"| page | int | {page_name} |",
            f"| limit | int | {limit_name} |",
            "",
            "**" + ("응답 예시" if self.lang == "ko" else "Example Response") + ":**",
            "",
            "```json",
            "{",
            '  "users": [',
            f'    {{"id": 1, "name": "{user_one}"}},',
            f'    {{"id": 2, "name": "{user_two}"}}',
            "  ],",
            f'  "total": {random.randint(20, 500)}',
            "}",
            "```",
            "",
            f"### POST {endpoint_post}",
            "",
            ("새 사용자를 생성합니다." if self.lang == "ko" else "Create a new user."),
            "",
            "**" + ("요청 본문" if self.lang == "ko" else "Request Body") + ":**",
            "",
            "```json",
            "{",
            f'  "name": "{self.data.name()}",',
            f'  "email": "{user_email}"',
            "}",
            "```",
            "",
        ]

        return "\n".join(lines)

    def _generate_tutorial(self) -> str:
        title = self.data.title()
        project_slug = self._slugify(title)
        install_command = self.data.install_command(package_name=project_slug)
        run_command = self.data.usage_command(entrypoint="main.py")
        requirements = self._sample_requirements(3)
        cfg_lines = self._sample_config_lines(2)
        cfg_1_key, cfg_1_value = self._to_config_entry(cfg_lines[0])
        cfg_2_key, cfg_2_value = self._to_config_entry(cfg_lines[1])
        lines = [
            "# " + ("튜토리얼" if self.lang == "ko" else "Tutorial") + f": {title}",
            "",
            self.data.paragraph(),
            "",
            "## " + ("시작하기 전에" if self.lang == "ko" else "Before You Begin"),
            "",
            ("다음 항목이 필요합니다:" if self.lang == "ko" else "You will need:"),
            "",
            f"- [ ] {requirements[0]}",
            f"- [ ] {requirements[1]}",
            f"- [ ] {requirements[2]}",
            "",
            "## " + ("1단계" if self.lang == "ko" else "Step 1") + ": " + ("설치" if self.lang == "ko" else "Installation"),
            "",
            ("먼저 패키지를 설치합니다:" if self.lang == "ko" else "First, install the package:"),
            "",
            "```bash",
            install_command,
            "```",
            "",
            "## " + ("2단계" if self.lang == "ko" else "Step 2") + ": " + ("설정" if self.lang == "ko" else "Configuration"),
            "",
            ("설정 파일을 생성합니다:" if self.lang == "ko" else "Create a configuration file:"),
            "",
            "```python",
            self.data.code_comment(),
            "config = {",
            f'    "{cfg_1_key}": "{cfg_1_value}",',
            f'    "{cfg_2_key}": "{cfg_2_value}"',
            "}",
            "```",
            "",
            "> **" + ("팁" if self.lang == "ko" else "Tip") + "**: " + self.data.paragraph()[:60],
            "",
            "## " + ("3단계" if self.lang == "ko" else "Step 3") + ": " + ("실행" if self.lang == "ko" else "Run"),
            "",
            "```bash",
            run_command,
            "```",
            "",
            ("예상 출력:" if self.lang == "ko" else "Expected output:"),
            "",
            "```",
            f"Success! Service started for {project_slug}",
            "```",
            "",
        ]

        return "\n".join(lines)

    def _generate_changelog(self) -> str:
        version = f"v{random.randint(1, 4)}.{random.randint(0, 9)}.{random.randint(0, 12)}"
        lines = [
            "# " + ("변경 이력" if self.lang == "ko" else "Changelog"),
            "",
            f"## {version} - {self.data.date()}",
            "",
            "### " + ("추가" if self.lang == "ko" else "Added"),
            "",
        ]
        for feature in self.data.features(3):
            lines.append(f"- {feature}")
        lines.extend([
            "",
            "### " + ("수정" if self.lang == "ko" else "Fixed"),
            "",
            f"- {self.data.paragraph()[:70]}",
            f"- {self.data.paragraph()[:70]}",
            "",
            "### " + ("변경" if self.lang == "ko" else "Changed"),
            "",
            "| " + ("항목" if self.lang == "ko" else "Item") + " | " + ("영향도" if self.lang == "ko" else "Impact") + " |",
            "|---|---|",
            "| API | High |",
            "| UI | Medium |",
            "| Docs | Low |",
            "",
        ])
        return "\n".join(lines)

    def _generate_meeting_notes(self) -> str:
        lines = [
            "# " + ("회의록" if self.lang == "ko" else "Meeting Notes"),
            "",
            "- " + ("일시" if self.lang == "ko" else "Date") + f": {self.data.date()}",
            "- " + ("참석자" if self.lang == "ko" else "Attendees") + f": {self.data.name()}, {self.data.name()}, {self.data.name()}",
            "",
            "## " + ("안건" if self.lang == "ko" else "Agenda"),
            "",
            "1. " + self.data.title(),
            "2. " + self.data.title(),
            "3. " + self.data.title(),
            "",
            "## " + ("결정사항" if self.lang == "ko" else "Decisions"),
            "",
            "- [x] " + self.data.features(1)[0],
            "- [x] " + self.data.features(1)[0],
            "- [ ] " + self.data.features(1)[0],
            "",
            "## " + ("액션 아이템" if self.lang == "ko" else "Action Items"),
            "",
            "| " + ("담당" if self.lang == "ko" else "Owner") + " | " + ("작업" if self.lang == "ko" else "Task") + " | " + ("기한" if self.lang == "ko" else "Due") + " |",
            "|---|---|---|",
            f"| {self.data.name()} | {self.data.paragraph()[:24]} | {self.data.date()} |",
            f"| {self.data.name()} | {self.data.paragraph()[:24]} | {self.data.date()} |",
            "",
        ]
        return "\n".join(lines)

    def _generate_incident_report(self) -> str:
        severity = random.choice(["SEV-1", "SEV-2", "SEV-3"])
        lines = [
            "# " + ("장애 보고서" if self.lang == "ko" else "Incident Report"),
            "",
            f"- Incident ID: INC-{random.randint(1000, 9999)}",
            f"- Severity: {severity}",
            "- " + ("발생 시각" if self.lang == "ko" else "Start Time") + f": {self.data.date()}",
            "",
            "## " + ("요약" if self.lang == "ko" else "Summary"),
            "",
            self.data.paragraph(),
            "",
            "## " + ("타임라인" if self.lang == "ko" else "Timeline"),
            "",
            f"- 09:10 - {self.data.paragraph()[:48]}",
            f"- 09:25 - {self.data.paragraph()[:48]}",
            f"- 09:41 - {self.data.paragraph()[:48]}",
            "",
            "## " + ("영향 범위" if self.lang == "ko" else "Impact"),
            "",
            "```json",
            "{",
            f"  \"affected_users\": {random.randint(100, 5000)},",
            f"  \"region\": \"{random.choice(['ap-northeast-2', 'us-east-1', 'eu-west-1'])}\",",
            f"  \"duration_min\": {random.randint(10, 180)}",
            "}",
            "```",
            "",
            "## " + ("재발 방지" if self.lang == "ko" else "Preventive Actions"),
            "",
            "- [ ] " + self.data.features(1)[0],
            "- [ ] " + self.data.features(1)[0],
            "",
        ]
        return "\n".join(lines)

    def _generate_release_note(self) -> str:
        release = f"{random.randint(2024, 2027)}.{random.randint(1, 12)}.{random.randint(1, 28)}"
        requirements = self._sample_requirements(3)
        runtime_1, min_1 = self._to_runtime_version(requirements[0])
        runtime_2, min_2 = self._to_runtime_version(requirements[1])
        runtime_3, min_3 = self._to_runtime_version(requirements[2])
        install_cmd = self.data.install_command(package_name="synthetic-ocr")
        run_cmd = self.data.usage_command(entrypoint="main.py")
        lines = [
            "# " + ("릴리즈 노트" if self.lang == "ko" else "Release Notes") + f" {release}",
            "",
            "## " + ("하이라이트" if self.lang == "ko" else "Highlights"),
            "",
            f"- **{self.data.features(1)[0]}**",
            f"- **{self.data.features(1)[0]}**",
            f"- **{self.data.features(1)[0]}**",
            "",
            "## " + ("호환성" if self.lang == "ko" else "Compatibility"),
            "",
            "| Runtime | Minimum | Recommended |",
            "|---|---|---|",
            f"| {runtime_1} | {min_1} | {min_1}+ |",
            f"| {runtime_2} | {min_2} | {min_2}+ |",
            f"| {runtime_3} | {min_3} | {min_3}+ |",
            "",
            "## " + ("업그레이드 가이드" if self.lang == "ko" else "Upgrade Guide"),
            "",
            "```bash",
            install_cmd.replace(" install ", " install -U "),
            f"{run_cmd} generate --lang en --size 10" if run_cmd.startswith("python") else run_cmd,
            "```",
            "",
        ]
        return "\n".join(lines)

    def _generate_compliance_checklist(self) -> str:
        lines = [
            "# " + ("컴플라이언스 체크리스트" if self.lang == "ko" else "Compliance Checklist"),
            "",
            "## " + ("데이터 보안" if self.lang == "ko" else "Data Security"),
            "",
            "- [x] Encryption at rest",
            "- [x] Encryption in transit",
            "- [ ] Data retention policy review",
            "",
            "## " + ("접근 통제" if self.lang == "ko" else "Access Control"),
            "",
            "1. MFA enabled for admins",
            "2. Role-based access matrix updated",
            "3. Quarterly permission audit completed",
            "",
            "## " + ("감사 로그" if self.lang == "ko" else "Audit Logs"),
            "",
            "```yaml",
            "audit:",
            "  enabled: true",
            "  retention_days: 365",
            "  export: s3://compliance-logs",
            "```",
            "",
            "> " + ("주의" if self.lang == "ko" else "Note") + ": " + self.data.paragraph(),
            "",
        ]
        return "\n".join(lines)


class MarkdownRenderer:
    """Renders markdown content to images."""

    _FONT_CACHE: Dict[Tuple[str, int], Any] = {}

    def __init__(self, font_path: str, style: Optional[MarkdownStyle] = None):
        self.style = style or MarkdownStyle()
        self.font_path = font_path

        try:
            self.body_font = self._get_font(font_path, self.style.body_font_size)
            self.h1_font = self._get_font(font_path, self.style.h1_font_size)
            self.h2_font = self._get_font(font_path, self.style.h2_font_size)
            self.h3_font = self._get_font(font_path, self.style.h3_font_size)
            self.code_font = self._get_font(font_path, self.style.code_font_size)
        except IOError:
            logger.warning("Font '%s' not found. Using default.", font_path)
            self.body_font = ImageFont.load_default()
            self.h1_font = self.body_font
            self.h2_font = self.body_font
            self.h3_font = self.body_font
            self.code_font = self.body_font

    @classmethod
    def _get_font(cls, font_path: str, size: int) -> ImageFont.ImageFont:
        key = (font_path, size)
        if key not in cls._FONT_CACHE:
            cls._FONT_CACHE[key] = ImageFont.truetype(font_path, size)
        return cls._FONT_CACHE[key]

    @staticmethod
    def _is_ordered_list_item(stripped: str) -> bool:
        return bool(stripped) and stripped[0].isdigit() and ". " in stripped

    @staticmethod
    def _parse_image_line(stripped: str) -> Optional[Tuple[str, str]]:
        return parse_markdown_image_line(stripped)

    @staticmethod
    def _parse_formula_line(stripped: str) -> Optional[str]:
        return parse_markdown_formula_line(stripped)

    @staticmethod
    def _image_placeholder_height(style: MarkdownStyle) -> int:
        return int(max(110, style.body_font_size * 7.0))

    def render(self, markdown_text: str) -> Image.Image:
        """Render markdown text to image."""
        lines = markdown_text.split("\n")
        style = self.style

        # First pass: calculate required height
        total_height = style.margin_top + style.margin_bottom
        line_heights = []

        for line in lines:
            height = self._get_line_height(line)
            line_heights.append(height)
            total_height += height

        # Create image
        width = style.margin_left + style.content_width + style.margin_right
        height = max(total_height, 200)

        img = Image.new("RGB", (width, int(height)), style.background_color)
        draw = ImageDraw.Draw(img)

        # Second pass: render content
        current_y = style.margin_top
        in_code_block = False
        code_block_start_y = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Handle code block markers
            if stripped.startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_block_start_y = current_y
                else:
                    # Draw code block background
                    draw.rectangle(
                        [
                            style.margin_left - 5,
                            code_block_start_y - 5,
                            style.margin_left + style.content_width + 5,
                            current_y + 5,
                        ],
                        fill=style.code_bg_color,
                    )
                    # Redraw code lines on top of background
                    in_code_block = False

                current_y += line_heights[i]
                continue

            if in_code_block:
                current_y = self._draw_code_line(draw, line, current_y, style)
            elif (image_payload := self._parse_image_line(stripped)) is not None:
                current_y = self._draw_image_placeholder(draw, image_payload[0], current_y, style)
            elif (formula_text := self._parse_formula_line(stripped)) is not None:
                current_y = self._draw_formula_line(draw, formula_text, current_y, style)
            elif stripped.startswith("# "):
                current_y = self._draw_h1(draw, stripped[2:], current_y, style)
            elif stripped.startswith("## "):
                current_y = self._draw_h2(draw, stripped[3:], current_y, style)
            elif stripped.startswith("### "):
                current_y = self._draw_h3(draw, stripped[4:], current_y, style)
            elif stripped.startswith("> "):
                current_y = self._draw_blockquote(draw, stripped[2:], current_y, style)
            elif stripped.startswith("- [ ]") or stripped.startswith("- [x]"):
                checked = stripped.startswith("- [x]")
                current_y = self._draw_checkbox_item(draw, stripped[6:], current_y, style, checked)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                current_y = self._draw_list_item(draw, stripped[2:], current_y, style, ordered=False)
            elif self._is_ordered_list_item(stripped):
                idx = stripped.index(". ")
                current_y = self._draw_list_item(
                    draw,
                    stripped[idx + 2 :],
                    current_y,
                    style,
                    ordered=True,
                    number=stripped[:idx],
                )
            elif stripped.startswith("|"):
                current_y = self._draw_table_row(draw, stripped, current_y, style)
            elif stripped == "---" or stripped == "***":
                current_y = self._draw_horizontal_rule(draw, current_y, style)
            elif stripped.startswith("*") and stripped.endswith("*"):
                current_y = self._draw_italic(draw, stripped.strip("*"), current_y, style)
            elif stripped:
                current_y = self._draw_paragraph(draw, stripped, current_y, style)
            else:
                current_y += int(self.style.body_font_size * 0.5)

        # Apply effects
        img = self._apply_effects(img, style)

        return img

    def _get_line_height(self, line: str) -> int:
        """Calculate height needed for a line."""
        stripped = line.strip()
        base_spacing = int(self.style.line_spacing * self.style.body_font_size)

        if stripped.startswith("# "):
            return int(self.style.h1_font_size * self.style.line_spacing) + 10
        if stripped.startswith("## "):
            return int(self.style.h2_font_size * self.style.line_spacing) + 8
        if stripped.startswith("### "):
            return int(self.style.h3_font_size * self.style.line_spacing) + 6
        if self._parse_image_line(stripped):
            return self._image_placeholder_height(self.style) + 14
        if self._parse_formula_line(stripped):
            return base_spacing + 18
        if stripped.startswith("```"):
            return 5
        if stripped.startswith("> "):
            return base_spacing + 10
        if stripped == "---" or stripped == "***":
            return 20
        if stripped:
            return base_spacing
        return int(self.style.body_font_size * 0.5)

    def _draw_h1(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw H1 header."""
        draw.text((style.margin_left, y), text, font=self.h1_font, fill=style.h1_color)
        # Draw underline
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h1_font)
        line_y = int(bbox[3]) + 5
        draw.line([(style.margin_left, line_y), (style.margin_left + style.content_width, line_y)],
                  fill=style.h2_color, width=2)
        return int(line_y + 15)

    def _draw_h2(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw H2 header."""
        draw.text((style.margin_left, y), text, font=self.h2_font, fill=style.h2_color)
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h2_font)
        return int(bbox[3] + 12)

    def _draw_h3(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw H3 header."""
        draw.text((style.margin_left, y), text, font=self.h3_font, fill=style.h3_color)
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h3_font)
        return int(bbox[3] + 10)

    def _draw_paragraph(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw paragraph text with word wrapping."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=self.body_font)
            if bbox[2] - bbox[0] <= style.content_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        for line in lines:
            # Handle inline code
            if "`" in line:
                y = self._draw_inline_code_line(draw, line, y, style)
            else:
                draw.text((style.margin_left, y), line, font=self.body_font, fill=style.text_color)
                y += int(style.body_font_size * style.line_spacing)

        return y + 5

    def _draw_inline_code_line(self, draw: ImageDraw.ImageDraw, line: str, y: int, style: MarkdownStyle) -> int:
        """Draw a line that may contain inline code."""
        x = style.margin_left
        parts = line.split("`")

        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code part
                bbox = draw.textbbox((x, y), part, font=self.code_font)
                draw.rectangle([x - 2, y - 1, bbox[2] + 2, bbox[3] + 1], fill=style.code_bg_color)
                draw.text((x, y), part, font=self.code_font, fill=style.code_text_color)
                x = bbox[2] + 4
            else:  # Normal text
                draw.text((x, y), part, font=self.body_font, fill=style.text_color)
                bbox = draw.textbbox((x, y), part, font=self.body_font)
                x = bbox[2]

        return y + int(style.body_font_size * style.line_spacing)

    def _draw_code_line(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw a line of code."""
        draw.text((style.margin_left + 10, y), text, font=self.code_font, fill=style.code_text_color)
        return y + int(style.code_font_size * style.line_spacing)

    def _draw_blockquote(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw blockquote."""
        # Draw left border
        draw.line(
            [(style.margin_left, y), (style.margin_left, y + style.body_font_size + 10)],
            fill=style.blockquote_border_color,
            width=3,
        )
        # Draw text
        draw.text(
            (style.margin_left + 15, y),
            text,
            font=self.body_font,
            fill=style.blockquote_color,
        )
        return y + int(style.body_font_size * style.line_spacing) + 10

    def _draw_list_item(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        y: int,
        style: MarkdownStyle,
        ordered: bool = False,
        number: Optional[str] = None,
    ) -> int:
        """Draw list item."""
        marker = f"{number}." if ordered and number else "•"
        draw.text((style.margin_left, y), marker, font=self.body_font, fill=style.text_color)
        bbox = draw.textbbox((style.margin_left, y), marker + " ", font=self.body_font)
        draw.text((bbox[2], y), text, font=self.body_font, fill=style.text_color)
        return y + int(style.body_font_size * style.line_spacing)

    def _draw_checkbox_item(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        y: int,
        style: MarkdownStyle,
        checked: bool = False,
    ) -> int:
        """Draw checkbox list item."""
        box_size = style.body_font_size - 2
        box_x = style.margin_left
        box_y = y + 2

        # Draw checkbox
        draw.rectangle([box_x, box_y, box_x + box_size, box_y + box_size], outline=style.text_color)
        if checked:
            draw.line([(box_x + 2, box_y + box_size // 2), (box_x + box_size // 2, box_y + box_size - 2)],
                      fill=style.text_color, width=2)
            draw.line([(box_x + box_size // 2, box_y + box_size - 2), (box_x + box_size - 2, box_y + 2)],
                      fill=style.text_color, width=2)

        draw.text((box_x + box_size + 8, y), text.strip(), font=self.body_font, fill=style.text_color)
        return y + int(style.body_font_size * style.line_spacing)

    def _draw_table_row(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw table row."""
        if text.replace("|", "").replace("-", "").strip() == "":
            # Separator row
            draw.line(
                [(style.margin_left, y + 5), (style.margin_left + style.content_width, y + 5)],
                fill=style.text_color,
                width=1,
            )
            return y + 10

        cells = [cell.strip() for cell in text.split("|") if cell.strip()]
        if not cells:
            return y + int(style.body_font_size * style.line_spacing)

        cell_width = style.content_width // max(len(cells), 1)
        for i, cell in enumerate(cells):
            x = style.margin_left + i * cell_width
            draw.text((x, y), cell, font=self.body_font, fill=style.text_color)

        return y + int(style.body_font_size * style.line_spacing) + 2

    def _draw_horizontal_rule(self, draw: ImageDraw.ImageDraw, y: int, style: MarkdownStyle) -> int:
        """Draw horizontal rule."""
        draw.line(
            [(style.margin_left, y + 10), (style.margin_left + style.content_width, y + 10)],
            fill=(200, 200, 200),
            width=1,
        )
        return y + 20

    def _draw_formula_line(self, draw: ImageDraw.ImageDraw, formula_text: str, y: int, style: MarkdownStyle) -> int:
        text = formula_text.strip()
        x = style.margin_left + 8
        text_y = y + 4
        bbox = draw.textbbox((x, text_y), text, font=self.code_font)
        box_left = style.margin_left
        box_top = y + 1
        box_right = min(style.margin_left + style.content_width, bbox[2] + 10)
        box_bottom = bbox[3] + 5

        draw.rectangle(
            [box_left, box_top, box_right, box_bottom],
            fill=style.code_bg_color,
            outline=style.blockquote_border_color,
            width=1,
        )
        draw.text((x, text_y), text, font=self.code_font, fill=style.code_text_color)
        return int(box_bottom + 8)

    def _draw_image_placeholder(self, draw: ImageDraw.ImageDraw, alt_text: str, y: int, style: MarkdownStyle) -> int:
        placeholder_height = self._image_placeholder_height(style)
        left = style.margin_left
        top = y + 4
        right = style.margin_left + style.content_width
        bottom = top + placeholder_height

        draw.rectangle(
            [left, top, right, bottom],
            fill=(245, 245, 245),
            outline=(170, 170, 170),
            width=2,
        )
        draw.line([(left + 8, top + 8), (right - 8, bottom - 8)], fill=(190, 190, 190), width=1)
        draw.line([(left + 8, bottom - 8), (right - 8, top + 8)], fill=(190, 190, 190), width=1)

        label = f"Image: {alt_text}" if alt_text else "Image"
        label_bbox = draw.textbbox((0, 0), label, font=self.body_font)
        label_x = left + max(8, (style.content_width - (label_bbox[2] - label_bbox[0])) // 2)
        label_y = top + max(8, (placeholder_height - (label_bbox[3] - label_bbox[1])) // 2)
        draw.rectangle(
            [label_x - 6, label_y - 3, label_x + (label_bbox[2] - label_bbox[0]) + 6, label_y + (label_bbox[3] - label_bbox[1]) + 3],
            fill=(255, 255, 255),
        )
        draw.text((label_x, label_y), label, font=self.body_font, fill=(90, 90, 90))

        return int(bottom + 10)

    def _draw_italic(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw italic text (simulated)."""
        draw.text((style.margin_left, y), text, font=self.body_font, fill=(100, 100, 100))
        return y + int(style.body_font_size * style.line_spacing)

    def _apply_effects(self, img: Image.Image, style: MarkdownStyle) -> Image.Image:
        """Apply noise and other effects."""
        if style.add_noise:
            img = self._add_noise(img)

        if style.add_blur:
            blur_radius = random.uniform(0.3, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if style.add_contrast:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.9, 1.1)
            img = enhancer.enhance(factor)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise to image."""
        width, height = img.size
        noise = np.zeros((height, width, 3), dtype=np.uint8)
        sample_count = 300
        xs = np.random.randint(0, width, size=sample_count)
        ys = np.random.randint(0, height, size=sample_count)
        grays = np.random.randint(0, 256, size=sample_count, dtype=np.uint8)
        noise[ys, xs] = np.stack([grays, grays, grays], axis=1)
        noise_img = Image.fromarray(noise, mode="RGB")

        return Image.blend(img, noise_img, 0.03)


class HtmlMarkdownRenderer:
    """Renders markdown through HTML and captures it as an image."""

    def __init__(self, font_path: str, style: Optional[MarkdownStyle] = None):
        self.style = style or MarkdownStyle()
        self.font_path = str(Path(font_path).resolve())

    @staticmethod
    def _coerce_markdown_html(markdown_text: str) -> str:
        try:
            markdown_pkg = importlib.import_module("markdown")
        except ImportError as exc:
            raise RuntimeError(
                "markdown package is required for markdown->html rendering. "
                "Install with: uv sync --group generate"
            ) from exc

        return markdown_pkg.markdown(
            markdown_text,
            extensions=["extra", "tables", "fenced_code", "sane_lists"],
        )

    @staticmethod
    def _prepare_component_markdown(markdown_text: str) -> str:
        prepared_lines: List[str] = []
        for raw_line in markdown_text.splitlines():
            stripped = raw_line.strip()

            image_payload = parse_markdown_image_line(stripped)
            if image_payload is not None:
                alt_text = image_payload[0] or "Image"
                safe_alt = escape(alt_text)
                prepared_lines.extend(
                    [
                        '<figure class="md-image-placeholder">',
                        f'  <div class="md-image-box" aria-label="{safe_alt}"><span>{safe_alt}</span></div>',
                        f"  <figcaption>{safe_alt}</figcaption>",
                        "</figure>",
                    ]
                )
                continue

            formula_text = parse_markdown_formula_line(stripped)
            if formula_text is not None:
                prepared_lines.append(f'<div class="md-formula">{escape(formula_text)}</div>')
                continue

            prepared_lines.append(raw_line)

        return "\n".join(prepared_lines)

    def _estimate_viewport_height(self, markdown_text: str) -> int:
        lines = markdown_text.splitlines() or [""]
        body_line_px = int(self.style.body_font_size * self.style.line_spacing)
        chars_per_line = max(18, self.style.content_width // max(self.style.body_font_size - 1, 8))

        wrapped_line_count = 0
        header_bonus = 0
        code_bonus = 0
        table_bonus = 0
        image_bonus = 0
        formula_bonus = 0
        for raw in lines:
            line = raw.strip()
            wrapped_line_count += max(1, (len(raw) // chars_per_line) + 1)
            if line.startswith("# "):
                header_bonus += self.style.h1_font_size
            elif line.startswith("## "):
                header_bonus += self.style.h2_font_size
            elif line.startswith("### "):
                header_bonus += self.style.h3_font_size
            if line.startswith("```"):
                code_bonus += int(self.style.code_font_size * self.style.line_spacing * 2)
            if line.startswith("|"):
                table_bonus += int(body_line_px * 0.6)
            if parse_markdown_image_line(line):
                image_bonus += max(120, int(self.style.body_font_size * 8.5))
            if parse_markdown_formula_line(line):
                formula_bonus += int(body_line_px * 1.6)

        estimated = (
            self.style.margin_top
            + self.style.margin_bottom
            + wrapped_line_count * body_line_px
            + header_bonus
            + code_bonus
            + table_bonus
            + image_bonus
            + formula_bonus
            + 120
        )
        return max(300, min(9000, int(estimated)))

    def _build_html_document(self, markdown_text: str) -> str:
        prepared_markdown = self._prepare_component_markdown(markdown_text)
        rendered_html = self._coerce_markdown_html(prepared_markdown)
        css = f"""
@font-face {{
  font-family: 'RenderFont';
  src: url('file://{escape(self.font_path)}') format('truetype');
}}
html, body {{
  margin: 0;
  padding: 0;
  background: rgb{self.style.background_color};
}}
body {{
  width: {self.style.margin_left + self.style.content_width + self.style.margin_right}px;
}}
.markdown-body {{
  width: {self.style.content_width}px;
  padding: {self.style.margin_top}px {self.style.margin_right}px {self.style.margin_bottom}px {self.style.margin_left}px;
  color: rgb{self.style.text_color};
  font-family: 'RenderFont', sans-serif;
  font-size: {self.style.body_font_size}px;
  line-height: {self.style.line_spacing};
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}}
.markdown-body h1 {{ font-size: {self.style.h1_font_size}px; color: rgb{self.style.h1_color}; margin: 0 0 16px 0; }}
.markdown-body h2 {{ font-size: {self.style.h2_font_size}px; color: rgb{self.style.h2_color}; margin: 18px 0 12px 0; }}
.markdown-body h3 {{ font-size: {self.style.h3_font_size}px; color: rgb{self.style.h3_color}; margin: 16px 0 8px 0; }}
.markdown-body a {{ color: rgb{self.style.link_color}; text-decoration: none; }}
.markdown-body p {{ margin: 0 0 10px 0; }}
.markdown-body ul, .markdown-body ol {{ margin: 0 0 12px 18px; padding: 0; }}
.markdown-body blockquote {{
  margin: 0 0 12px 0;
  padding: 0 0 0 12px;
  border-left: 3px solid rgb{self.style.blockquote_border_color};
  color: rgb{self.style.blockquote_color};
}}
.markdown-body pre, .markdown-body code {{
  font-family: 'RenderFont', monospace;
  font-size: {self.style.code_font_size}px;
}}
.markdown-body pre {{
  margin: 0 0 12px 0;
  padding: 8px 10px;
  background: rgb{self.style.code_bg_color};
  color: rgb{self.style.code_text_color};
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.markdown-body code {{
  background: rgb{self.style.code_bg_color};
  color: rgb{self.style.code_text_color};
  padding: 1px 3px;
}}
.markdown-body table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 12px 0;
  table-layout: fixed;
}}
.markdown-body th, .markdown-body td {{
  border: 1px solid rgba(0, 0, 0, 0.25);
  text-align: left;
  padding: 6px;
  overflow-wrap: anywhere;
}}
.markdown-body .md-formula {{
  margin: 0 0 12px 0;
  padding: 8px 10px;
  background: rgb{self.style.code_bg_color};
  color: rgb{self.style.code_text_color};
  border: 1px solid rgba(0, 0, 0, 0.2);
  font-family: 'RenderFont', monospace;
  font-size: {self.style.code_font_size}px;
  overflow-wrap: anywhere;
}}
.markdown-body .md-image-placeholder {{
  margin: 0 0 12px 0;
}}
.markdown-body .md-image-box {{
  width: 100%;
  min-height: 130px;
  border: 2px solid rgba(0, 0, 0, 0.28);
  background: rgba(240, 240, 240, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(0, 0, 0, 0.6);
  font-weight: 600;
}}
.markdown-body .md-image-placeholder figcaption {{
  margin-top: 6px;
  color: rgba(0, 0, 0, 0.65);
  font-size: {max(10, self.style.body_font_size - 1)}px;
}}
"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>{css}</style>
</head>
<body>
  <div class="markdown-body">{rendered_html}</div>
</body>
</html>"""

    def _trim_bottom_whitespace(self, image: Image.Image) -> Image.Image:
        background = Image.new("RGB", image.size, self.style.background_color)
        diff = ImageChops.difference(image, background)
        bbox = diff.getbbox()
        if not bbox:
            return image
        cropped_bottom = min(image.height, int(bbox[3] + self.style.margin_bottom))
        return image.crop((0, 0, image.width, max(cropped_bottom, 200)))

    def _apply_effects(self, img: Image.Image) -> Image.Image:
        if self.style.add_noise:
            img = self._add_noise(img)

        if self.style.add_blur:
            blur_radius = random.uniform(0.3, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if self.style.add_contrast:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.9, 1.1)
            img = enhancer.enhance(factor)

        return img

    @staticmethod
    def _add_noise(img: Image.Image) -> Image.Image:
        width, height = img.size
        noise = np.zeros((height, width, 3), dtype=np.uint8)
        sample_count = 300
        xs = np.random.randint(0, width, size=sample_count)
        ys = np.random.randint(0, height, size=sample_count)
        grays = np.random.randint(0, 256, size=sample_count, dtype=np.uint8)
        noise[ys, xs] = np.stack([grays, grays, grays], axis=1)
        noise_img = Image.fromarray(noise, mode="RGB")
        return Image.blend(img, noise_img, 0.03)

    def render(self, markdown_text: str) -> Image.Image:
        try:
            Html2Image = importlib.import_module("html2image").Html2Image
        except ImportError as exc:
            raise RuntimeError(
                "html2image package is required for html->image rendering. "
                "Install with: uv sync --group generate"
            ) from exc

        width = self.style.margin_left + self.style.content_width + self.style.margin_right
        height = self._estimate_viewport_height(markdown_text)
        html_doc = self._build_html_document(markdown_text)

        with tempfile.TemporaryDirectory(prefix="markdown-html2image-") as temp_dir:
            hti = Html2Image(
                output_path=temp_dir,
                size=(width, height),
                custom_flags=[
                    "--headless=new",
                    "--hide-scrollbars",
                    "--disable-gpu",
                    "--force-device-scale-factor=1",
                ],
            )
            out_name = "rendered.png"
            hti.screenshot(html_str=html_doc, save_as=out_name)
            rendered_path = Path(temp_dir) / out_name
            image = Image.open(rendered_path).convert("RGB")
            image.load()

        image = self._trim_bottom_whitespace(image)
        return self._apply_effects(image)


class Generator(BaseGenerator):
    """Main generator class for markdown image generation."""

    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str = "ko",
    ):
        super().__init__(output_dir, font_dir, lang)
        self.data_generator = MarkdownDataGenerator(lang)
        self.similarity_db: Dict[str, Any] = {}
        self.similarity_db_path = ""
        self._similarity_db_source: Optional[str] = None
        self._protected_chars = set("#`|[](){}<>!+-=_~*/\\")
        self.template_catalog = TemplateCatalog()
        self.template_specs: List[TemplateSpec] = self.template_catalog.all_specs()
        self.template_counts: Counter[str] = Counter()
        self.family_counts: Counter[str] = Counter()
        self.coverage_targets: Dict[str, float] = {}
        self.template_family: Optional[str] = None
        self.min_template_complexity: Optional[int] = None
        self.max_template_complexity: Optional[int] = None
        self.template_config_dir: Optional[str] = None
        self.add_noise = True
        self.add_blur = False
        self.noise_ratio = 0.1
        self.blur_ratio = 0.1
        self.similar_char_ratio = 0.08
        self.markdown_renderer = "pil"
        self.style_profile = "balanced"
        self.novelty_window = DEFAULT_NOVELTY_WINDOW
        self.novelty_threshold = DEFAULT_NOVELTY_THRESHOLD
        self.novelty_max_attempts = DEFAULT_NOVELTY_MAX_ATTEMPTS
        self._recent_signatures: deque[str] = deque(maxlen=self.novelty_window)
        self.base_seed: Optional[int] = None
        self.max_render_width = A4_MAX_WIDTH_PX
        self.max_render_height = A4_MAX_HEIGHT_PX

    def _load_similarity_db(self, db_path: Optional[str]) -> None:
        source_key = db_path or "__auto__"
        if self._similarity_db_source == source_key:
            return

        if db_path:
            candidates = [Path(db_path)]
        else:
            candidates = [
                Path("data") / self.lang / f"char_similarity_db_{self.lang}.json",
                Path("data") / f"char_similarity_db_{self.lang}.json",
                Path("data") / self.lang / "char_similarity_db.json",
                Path("data") / "char_similarity_db.json",
            ]

        self._similarity_db_source = source_key
        resolved = next((p for p in candidates if p.exists()), None)
        if resolved is None:
            self.similarity_db = {}
            self.similarity_db_path = ""
            return

        loaded = read_json(str(resolved))
        if isinstance(loaded, dict):
            self.similarity_db = loaded
            self.similarity_db_path = str(resolved)
            return

        self.similarity_db = {}
        self.similarity_db_path = ""

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_ratio(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, ratio))

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    @staticmethod
    def _normalize_choice(
        value: Any,
        allowed: set[str],
        fallback: str,
        warning_label: str,
    ) -> str:
        normalized = str(value).strip().lower()
        if normalized in allowed:
            return normalized
        logger.warning(
            "Unknown %s '%s'. Falling back to '%s'.",
            warning_label,
            normalized,
            fallback,
        )
        return fallback

    def _resolve_effect_settings(
        self,
        *,
        enabled_key: str,
        ratio_key: str,
        enabled_default: bool,
        ratio_default: float,
        kwargs: dict[str, Any],
    ) -> tuple[bool, float]:
        enabled = self._coerce_bool(kwargs.get(enabled_key), enabled_default)
        ratio = self._coerce_ratio(kwargs.get(ratio_key), ratio_default)
        if enabled_key in kwargs and kwargs.get(enabled_key) is not None:
            ratio = 1.0 if enabled else 0.0
        return enabled, ratio

    def _resolve_template_specs(self, template: Optional[str]) -> List[TemplateSpec]:
        return self.template_catalog.resolve(
            template=template,
            template_family=self.template_family,
            min_complexity=self.min_template_complexity,
            max_complexity=self.max_template_complexity,
        )

    def _configure_template_selection(self, **kwargs) -> None:
        self.template_family = kwargs.get("template_family")
        self.min_template_complexity = self._coerce_optional_int(
            kwargs.get("min_template_complexity")
        )
        self.max_template_complexity = self._coerce_optional_int(
            kwargs.get("max_template_complexity")
        )
        if (
            self.min_template_complexity is not None
            and self.max_template_complexity is not None
            and self.min_template_complexity > self.max_template_complexity
        ):
            self.min_template_complexity, self.max_template_complexity = (
                self.max_template_complexity,
                self.min_template_complexity,
            )

        requested_catalog_dir = kwargs.get("template_config_dir")
        if requested_catalog_dir != self.template_config_dir:
            self.template_catalog = TemplateCatalog(config_dir=requested_catalog_dir)
            self.template_config_dir = requested_catalog_dir

        template = kwargs.get("template")
        self.template_specs = self._resolve_template_specs(template)
        self.coverage_targets = parse_coverage_targets(kwargs.get("coverage_targets"))

    def _configure_rendering(self, **kwargs) -> None:
        self.add_noise, self.noise_ratio = self._resolve_effect_settings(
            enabled_key="add_noise",
            ratio_key="noise_ratio",
            enabled_default=True,
            ratio_default=0.1,
            kwargs=kwargs,
        )
        self.add_blur, self.blur_ratio = self._resolve_effect_settings(
            enabled_key="add_blur",
            ratio_key="blur_ratio",
            enabled_default=False,
            ratio_default=0.1,
            kwargs=kwargs,
        )
        self.similar_char_ratio = float(kwargs.get("similar_char_ratio", 0.08))

        self.markdown_renderer = self._normalize_choice(
            kwargs.get("markdown_renderer", self.markdown_renderer),
            {"pil", "html2image"},
            "pil",
            "markdown renderer",
        )

        self.style_profile = self._normalize_choice(
            kwargs.get("style_profile", self.style_profile),
            {"legacy", "balanced", "aggressive"},
            "balanced",
            "style profile",
        )

    def _configure_novelty(self, **kwargs) -> None:
        self.novelty_window = max(
            5,
            self._coerce_optional_int(kwargs.get("novelty_window")) or self.novelty_window,
        )
        self.novelty_threshold = self._coerce_ratio(
            kwargs.get("novelty_threshold"),
            self.novelty_threshold,
        )
        self.novelty_max_attempts = max(
            1,
            self._coerce_optional_int(kwargs.get("novelty_max_attempts"))
            or self.novelty_max_attempts,
        )
        self._recent_signatures = deque(
            self._recent_signatures,
            maxlen=self.novelty_window,
        )

    def _configure_generation(self, **kwargs) -> None:
        if "seed" in kwargs:
            self.base_seed = self._coerce_optional_int(kwargs.get("seed"))

        self._configure_template_selection(**kwargs)
        self._configure_rendering(**kwargs)
        self._configure_novelty(**kwargs)

        self._load_similarity_db(kwargs.get("similarity_db_path"))

    def _mutate_similar_text(self, text: str, ratio: float) -> Tuple[str, int]:
        if ratio <= 0 or not self.similarity_db:
            return text, 0

        chars = list(text)
        candidate_indices: List[int] = []
        cached_candidates: Dict[str, List[Tuple[str, float]]] = {}

        def get_candidates(ch: str) -> List[Tuple[str, float]]:
            if ch not in cached_candidates:
                cached_candidates[ch] = find_similar_chars(ch, self.similarity_db, top_n=5)
            return cached_candidates[ch]

        for idx, ch in enumerate(chars):
            if ch in self._protected_chars or ch.isspace():
                continue
            if get_candidates(ch):
                candidate_indices.append(idx)

        if not candidate_indices:
            return text, 0

        target = int(len(candidate_indices) * ratio)
        if target == 0:
            target = 1
        target = min(target, len(candidate_indices))

        mutated_count = 0
        for idx in random.sample(candidate_indices, target):
            source = chars[idx]
            candidates = get_candidates(source)
            if not candidates:
                continue
            replacement, _ = random.choice(candidates)
            if (
                not replacement
                or any(c in self._protected_chars or c.isspace() for c in replacement)
            ):
                continue
            if replacement == source:
                continue
            chars[idx] = replacement
            mutated_count += 1

        return "".join(chars), mutated_count

    def _derive_sample_seed(self, sample_index: int, attempt: int) -> Optional[int]:
        if self.base_seed is None:
            return None
        return int(self.base_seed + sample_index * 1009 + attempt * 9176)

    def _seed_for_sample(self, sample_seed: Optional[int]) -> None:
        if sample_seed is None:
            return

        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32 - 1))

        faker = getattr(self.data_generator.data, "faker", None)
        if faker is not None:
            try:
                faker.seed_instance(sample_seed)
            except Exception:
                pass

    def _fit_image_to_a4(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        width, height = image.size
        if width <= self.max_render_width and height <= self.max_render_height:
            return image, False

        ratio = min(self.max_render_width / max(1, width), self.max_render_height / max(1, height))
        new_width = max(1, int(width * ratio))
        new_height = max(1, int(height * ratio))
        if hasattr(Image, "Resampling"):
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            resized = image.resize((new_width, new_height), getattr(Image, "LANCZOS", 1))
        return resized, True

    @staticmethod
    def _structure_signature(markdown_text: str) -> str:
        tokens: List[str] = []
        for raw_line in markdown_text.splitlines():
            line = raw_line.strip()
            if not line:
                tokens.append("blank")
            elif parse_markdown_image_line(line):
                tokens.append("image")
            elif parse_markdown_formula_line(line):
                tokens.append("formula")
            elif line.startswith("# "):
                tokens.append("h1")
            elif line.startswith("## "):
                tokens.append("h2")
            elif line.startswith("### "):
                tokens.append("h3")
            elif line.startswith("```"):
                tokens.append("code")
            elif line.startswith("| ") and line.endswith(" |"):
                tokens.append("table")
            elif line.startswith("- [ ") or line.startswith("- [x"):
                tokens.append("check")
            elif line.startswith("- "):
                tokens.append("ul")
            elif line and line[0].isdigit() and ". " in line:
                tokens.append("ol")
            elif line.startswith("> "):
                tokens.append("quote")
            elif line in {"---", "***"}:
                tokens.append("rule")
            else:
                tokens.append("p")
        return "|".join(tokens)

    def _novelty_score(self, signature: str) -> float:
        if not self._recent_signatures:
            return 0.0
        return max(
            SequenceMatcher(None, signature, existing).ratio()
            for existing in self._recent_signatures
        )

    def _select_template_spec(self) -> Tuple[TemplateSpec, float]:
        if not self.template_specs:
            self.template_specs = self.template_catalog.all_specs()

        total_generated = sum(self.family_counts.values())
        weights: List[float] = []
        for spec in self.template_specs:
            template_seen = self.template_counts.get(spec.template_id, 0)
            family_seen = self.family_counts.get(spec.family, 0)

            diversity_factor = 1.0 / (1.0 + template_seen * 0.45)
            family_balance_factor = 1.0 / (1.0 + family_seen * 0.2)
            coverage_factor = 1.0

            if self.coverage_targets:
                target_ratio = self.coverage_targets.get(spec.family)
                if target_ratio is not None:
                    if total_generated == 0:
                        coverage_factor = 1.5
                    else:
                        observed_ratio = family_seen / total_generated
                        deficit = target_ratio - observed_ratio
                        coverage_factor = max(0.25, 1.0 + deficit * 5.0)

            weights.append(max(0.01, spec.weight * diversity_factor * family_balance_factor * coverage_factor))

        selected_index = random.choices(range(len(self.template_specs)), weights=weights, k=1)[0]
        return self.template_specs[selected_index], weights[selected_index]

    def generate(
        self,
        num_images: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate markdown images."""
        self._configure_generation(**kwargs)
        self.template_counts = Counter()
        self.family_counts = Counter()
        self._recent_signatures = deque(maxlen=self.novelty_window)

        metadata = []
        for idx in tqdm(range(num_images), desc="Generating markdown images"):
            image, meta = self.generate_single(sample_index=idx)

            # Save image
            filename = f"markdown_{idx:05d}.png"
            self.save_image(image, filename)
            meta["file_name"] = str(self.output_dir / filename)

            metadata.append(meta)

        return metadata

    def generate_single(self, sample_index: int = 0, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        if kwargs:
            self._configure_generation(**kwargs)

        available_specs = self.template_specs or self.template_catalog.all_specs()
        if not available_specs:
            raise RuntimeError("Template catalog is empty. Add template specs under configs/generator/templates.")

        selected_template: TemplateSpec = available_specs[0]
        selected_weight = 0.0
        markdown_text = ""
        mutation_count = 0
        signature = ""
        novelty_score = 0.0
        sample_seed: Optional[int] = None
        selection_attempt = 1

        for attempt in range(self.novelty_max_attempts):
            sample_seed = self._derive_sample_seed(sample_index, attempt)
            self._seed_for_sample(sample_seed)

            selected_template, selected_weight = self._select_template_spec()

            original_markdown = self.data_generator.generate_markdown(
                template_id=selected_template.template_id,
                template_spec=selected_template,
            )
            markdown_text, mutation_count = self._mutate_similar_text(
                original_markdown,
                self.similar_char_ratio,
            )
            signature = self._structure_signature(markdown_text)
            novelty_score = self._novelty_score(signature)
            selection_attempt = attempt + 1

            if novelty_score < self.novelty_threshold or attempt == self.novelty_max_attempts - 1:
                break

        # Create style with random variations
        style = self._random_style()
        style.add_noise = random.random() < self.noise_ratio
        style.add_blur = random.random() < self.blur_ratio

        # Render markdown
        font_path = random.choice(self.font_paths)
        if self.markdown_renderer == "html2image":
            renderer = HtmlMarkdownRenderer(font_path, style)
        else:
            renderer = MarkdownRenderer(font_path, style)
        image = renderer.render(markdown_text)
        image, a4_scaled = self._fit_image_to_a4(image)

        self.template_counts[selected_template.template_id] += 1
        self.family_counts[selected_template.family] += 1
        self._recent_signatures.append(signature)

        generated_count = max(1, sum(self.family_counts.values()))
        family_ratio = self.family_counts[selected_template.family] / generated_count

        metadata = {
            "template": selected_template.template_id,
            "template_id": selected_template.template_id,
            "template_family": selected_template.family,
            "template_complexity": selected_template.complexity,
            "template_mode": selected_template.mode,
            "template_version": selected_template.version,
            "template_source": selected_template.source,
            "template_weight": round(selected_weight, 6),
            "GT_markdown": markdown_text,
            "GT_json": markdown_to_json_ast(markdown_text),
            "similar_char_mutations": mutation_count,
            "renderer": self.markdown_renderer,
            "style_profile": self.style_profile,
            "sample_index": sample_index,
            "sample_seed": sample_seed,
            "selection_attempt": selection_attempt,
            "structure_signature": signature,
            "novelty_score": round(novelty_score, 6),
            "family_ratio": round(family_ratio, 6),
            "a4_scaled": a4_scaled,
            "image_width": image.width,
            "image_height": image.height,
        }
        return image, metadata

    @staticmethod
    def _base_styles() -> List[MarkdownStyle]:
        return [
            MarkdownStyle(
                background_color=(255, 255, 255),
                h1_color=(0, 0, 0),
                add_noise=True,
                margin_left=34,
                margin_right=34,
                content_width=620,
            ),
            MarkdownStyle(
                background_color=(250, 250, 245),
                h1_color=(51, 51, 51),
                add_noise=True,
                add_blur=True,
                margin_left=48,
                margin_right=48,
                content_width=560,
                line_spacing=1.45,
            ),
            MarkdownStyle(
                background_color=(255, 253, 250),
                h1_color=(30, 30, 30),
                add_noise=True,
                add_contrast=True,
                margin_left=56,
                margin_right=56,
                content_width=540,
                h1_font_size=30,
            ),
            MarkdownStyle(
                background_color=(248, 249, 250),
                h1_color=(36, 41, 46),
                link_color=(3, 102, 214),
                add_noise=False,
                margin_left=40,
                margin_right=40,
                content_width=640,
                body_font_size=13,
                code_font_size=11,
            ),
            MarkdownStyle(
                background_color=(244, 240, 232),
                h1_color=(44, 38, 31),
                h2_color=(70, 64, 58),
                text_color=(42, 42, 42),
                add_noise=True,
                add_blur=False,
                add_contrast=True,
                margin_left=60,
                margin_right=52,
                content_width=520,
                line_spacing=1.58,
            ),
            MarkdownStyle(
                background_color=(236, 242, 246),
                h1_color=(12, 42, 68),
                h2_color=(29, 72, 102),
                link_color=(12, 96, 158),
                text_color=(25, 36, 46),
                code_bg_color=(222, 232, 240),
                add_noise=False,
                add_blur=True,
                margin_left=44,
                margin_right=44,
                content_width=600,
                line_spacing=1.4,
            ),
        ]

    @staticmethod
    def _clamp_color(value: int) -> int:
        return max(0, min(255, value))

    def _jitter_color(self, color: Tuple[int, int, int], span: int) -> Tuple[int, int, int]:
        return (
            self._clamp_color(color[0] + random.randint(-span, span)),
            self._clamp_color(color[1] + random.randint(-span, span)),
            self._clamp_color(color[2] + random.randint(-span, span)),
        )

    def _random_style(self) -> MarkdownStyle:
        """Generate random style variations."""
        selected = random.choice(self._base_styles())

        if self.style_profile == "legacy":
            selected.margin_top += random.randint(-8, 14)
            selected.margin_bottom += random.randint(-8, 14)
            selected.content_width += random.randint(-24, 24)
            selected.line_spacing = max(1.3, min(1.7, selected.line_spacing + random.uniform(-0.08, 0.1)))
            return selected

        if self.style_profile == "balanced":
            selected.margin_top += random.randint(-16, 24)
            selected.margin_bottom += random.randint(-16, 24)
            selected.margin_left += random.randint(-10, 16)
            selected.margin_right += random.randint(-10, 16)
            selected.content_width += random.randint(-64, 72)
            selected.line_spacing = max(1.2, min(1.9, selected.line_spacing + random.uniform(-0.2, 0.25)))
            selected.background_color = self._jitter_color(selected.background_color, 12)
            selected.h1_color = self._jitter_color(selected.h1_color, 16)
            selected.h2_color = self._jitter_color(selected.h2_color, 16)
            selected.text_color = self._jitter_color(selected.text_color, 12)
            selected.link_color = self._jitter_color(selected.link_color, 24)
        else:
            selected.margin_top += random.randint(-24, 36)
            selected.margin_bottom += random.randint(-24, 36)
            selected.margin_left += random.randint(-18, 24)
            selected.margin_right += random.randint(-18, 24)
            selected.content_width += random.randint(-96, 108)
            selected.line_spacing = max(1.15, min(2.0, selected.line_spacing + random.uniform(-0.28, 0.35)))
            selected.body_font_size = max(12, min(18, selected.body_font_size + random.randint(-2, 3)))
            selected.code_font_size = max(10, min(16, selected.code_font_size + random.randint(-1, 3)))
            selected.h1_font_size = max(
                selected.body_font_size + 6,
                min(40, selected.h1_font_size + random.randint(-4, 8)),
            )
            selected.h2_font_size = max(
                selected.body_font_size + 3,
                min(34, selected.h2_font_size + random.randint(-3, 6)),
            )
            selected.h3_font_size = max(
                selected.body_font_size + 1,
                min(28, selected.h3_font_size + random.randint(-2, 5)),
            )

            bg_base = random.randint(220, 255)
            selected.background_color = (
                bg_base,
                self._clamp_color(bg_base + random.randint(-12, 10)),
                self._clamp_color(bg_base + random.randint(-12, 10)),
            )
            selected.text_color = (
                random.randint(18, 70),
                random.randint(18, 70),
                random.randint(18, 70),
            )
            selected.h1_color = (
                random.randint(0, 60),
                random.randint(0, 60),
                random.randint(0, 80),
            )
            selected.h2_color = self._jitter_color(selected.h1_color, 20)
            selected.link_color = (
                random.randint(0, 40),
                random.randint(80, 150),
                random.randint(140, 220),
            )

        selected.margin_top = max(16, selected.margin_top)
        selected.margin_bottom = max(16, selected.margin_bottom)
        selected.margin_left = max(20, selected.margin_left)
        selected.margin_right = max(20, selected.margin_right)
        selected.content_width = max(460, min(720, selected.content_width))
        return selected
