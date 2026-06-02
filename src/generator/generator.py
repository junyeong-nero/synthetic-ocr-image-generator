"""Markdown generator module for synthetic OCR markdown image generation.

This module now supports composable document generation:
- `TextGenerator`: emits text-only markdown sections
- `TableGenerator`: emits markdown table sections
- `FormularGenerator`: emits markdown formula sections
- `MergeOrchestrator`: merges and shuffles sections into one markdown document
"""

import logging
import numpy as np
import random
from collections import Counter, deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image

from src.character_similarity import find_similar_chars
from src.generator.base import BaseGenerator
from src.generator.generation_config import (
    A4_MAX_HEIGHT_PX,
    A4_MAX_WIDTH_PX,
    DEFAULT_NOVELTY_MAX_ATTEMPTS,
    DEFAULT_NOVELTY_THRESHOLD,
    DEFAULT_NOVELTY_WINDOW,
    coerce_bool,
    coerce_optional_int,
    coerce_ratio,
    normalize_choice,
    resolve_effect_settings,
)
from src.generator.markdown_content import (
    DEFAULT_FORMULA_SOURCE_WEIGHTS,
    HARD_CODED_FORMULA_EXPRESSIONS,
    MarkdownDataGenerator,
)
from src.generator.markdown_renderers import HtmlMarkdownRenderer, MarkdownRenderer, PlaywrightMarkdownRenderer
from src.generator.markdown_render_utils import (
    MarkdownStyle,
    parse_markdown_formula_line,
    parse_markdown_image_line,
)
from src.generator.style_sampler import base_styles, clamp_color, jitter_color, random_style
from src.generator.template_catalog import TemplateCatalog, TemplateSpec, parse_coverage_targets
from src.generator.text_generator import TextGenerator
from src.generator.text_mutation import mutate_similar_text, mutate_text_generator_sections
from src.utils import markdown_to_json_ast, read_json

logger = logging.getLogger(__name__)

__all__ = [
    "A4_MAX_HEIGHT_PX",
    "A4_MAX_WIDTH_PX",
    "Generator",
    "HARD_CODED_FORMULA_EXPRESSIONS",
    "HtmlMarkdownRenderer",
    "MarkdownDataGenerator",
    "MarkdownRenderer",
    "PlaywrightMarkdownRenderer",
    "TemplateCatalog",
    "TemplateSpec",
    "TextGenerator",
    "normalize_chained_scripts",
    "parse_coverage_targets",
    "parse_markdown_formula_line",
    "parse_markdown_image_line",
]




def normalize_chained_scripts(expression: str) -> str:
    def parse_base(text: str, start: int) -> tuple[str, int]:
        if start >= len(text):
            return "", start
        if text[start] == "\\":
            end = start + 1
            while end < len(text) and text[end].isalpha():
                end += 1
            return text[start:end], end
        if text[start] == "{":
            depth = 1
            end = start + 1
            while end < len(text):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : end + 1], end + 1
                end += 1
            return text[start:], len(text)
        return text[start], start + 1

    def parse_script_arg(text: str, start: int) -> tuple[str, int]:
        token, end = parse_base(text, start)
        if token.startswith("{") and token.endswith("}"):
            return token[1:-1], end
        return token, end

    result: list[str] = []
    index = 0
    while index < len(expression):
        if expression[index] in {"^", "_"}:
            result.append(expression[index])
            index += 1
            continue

        base_token, cursor = parse_base(expression, index)
        scripts: list[tuple[str, str]] = []
        while cursor < len(expression) and expression[cursor] in {"^", "_"}:
            marker = expression[cursor]
            arg, cursor = parse_script_arg(expression, cursor + 1)
            scripts.append((marker, arg))

        if not scripts:
            result.append(base_token)
            index = cursor
            continue

        grouped: dict[str, list[str]] = {}
        marker_order: list[str] = []
        for marker, arg in scripts:
            if marker not in grouped:
                grouped[marker] = []
                marker_order.append(marker)
            grouped[marker].append(arg)

        rebuilt = [base_token]
        for marker in marker_order:
            values = grouped[marker]
            combined = values[0]
            for extra in values[1:]:
                combined = f"{combined}{marker}{{{extra}}}"
            rebuilt.append(f"{marker}{{{combined}}}")

        result.append("".join(rebuilt))
        index = cursor

    return "".join(result)


_normalize_chained_scripts = normalize_chained_scripts

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
        self.markdown_renderer = "playwright"
        self.style_profile = "balanced"
        self.novelty_window = DEFAULT_NOVELTY_WINDOW
        self.novelty_threshold = DEFAULT_NOVELTY_THRESHOLD
        self.novelty_max_attempts = DEFAULT_NOVELTY_MAX_ATTEMPTS
        self._recent_signatures: deque[str] = deque(maxlen=self.novelty_window)
        self.base_seed: Optional[int] = None
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
        return coerce_optional_int(value)

    @staticmethod
    def _coerce_ratio(value: Any, default: float) -> float:
        return coerce_ratio(value, default)

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        return coerce_bool(value, default)

    @staticmethod
    def _normalize_choice(
        value: Any,
        allowed: set[str],
        fallback: str,
        warning_label: str,
    ) -> str:
        return normalize_choice(value, allowed, fallback, warning_label, logger)

    def _resolve_effect_settings(
        self,
        *,
        enabled_key: str,
        ratio_key: str,
        enabled_default: bool,
        ratio_default: float,
        kwargs: dict[str, Any],
    ) -> tuple[bool, float]:
        return resolve_effect_settings(
            enabled_key=enabled_key,
            ratio_key=ratio_key,
            enabled_default=enabled_default,
            ratio_default=ratio_default,
            kwargs=kwargs,
        )

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
            {"pil", "html2image", "playwright"},
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

    def _configure_content_sources(self, **kwargs) -> None:
        self.data_generator.configure_content_sources(
            formula_source_mode=kwargs.get(
                "formula_source_mode",
                self.data_generator.formula_source_mode,
            ),
            formula_dataset_path=kwargs.get("formula_dataset_path", self.data_generator.formula_dataset_path),
            formula_dataset_weight=kwargs.get(
                "formula_dataset_weight",
                self.data_generator.formula_source_weights.get("dataset", DEFAULT_FORMULA_SOURCE_WEIGHTS["dataset"]),
            ),
            formula_random_weight=kwargs.get(
                "formula_random_weight",
                self.data_generator.formula_source_weights.get("random", DEFAULT_FORMULA_SOURCE_WEIGHTS["random"]),
            ),
            formula_synthetic_weight=kwargs.get(
                "formula_synthetic_weight",
                self.data_generator.formula_source_weights.get("synthetic", DEFAULT_FORMULA_SOURCE_WEIGHTS["synthetic"]),
            ),
        )

    def _configure_generation(self, **kwargs) -> None:
        if "seed" in kwargs:
            self.base_seed = self._coerce_optional_int(kwargs.get("seed"))

        self._configure_template_selection(**kwargs)
        self._configure_rendering(**kwargs)
        self._configure_novelty(**kwargs)
        self._configure_content_sources(**kwargs)

        self._load_similarity_db(kwargs.get("similarity_db_path"))

    def _mutate_similar_text(self, text: str, ratio: float) -> Tuple[str, int]:
        cached_candidates: Dict[str, List[Tuple[str, float]]] = {}

        def get_candidates(ch: str) -> List[Tuple[str, float]]:
            if ch not in cached_candidates:
                cached_candidates[ch] = find_similar_chars(ch, self.similarity_db, top_n=5)
            return cached_candidates[ch]
        return mutate_similar_text(
            text=text,
            ratio=ratio,
            similarity_db=self.similarity_db,
            protected_chars=self._protected_chars,
            candidate_lookup=get_candidates,
        )

    def _mutate_text_generator_sections(
        self,
        markdown_text: str,
        ratio: float,
        merge_order: List[str],
    ) -> Tuple[str, int]:
        return mutate_text_generator_sections(
            markdown_text=markdown_text,
            ratio=ratio,
            merge_order=merge_order,
            mutate_section=self._mutate_similar_text,
        )

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
    ) -> int:
        """Generate markdown images."""
        metadata_handle = kwargs.pop("metadata_handle", None)
        stats_accumulator = kwargs.pop("stats_accumulator", None)
        sample_start_index = int(kwargs.pop("sample_start_index", 0))
        if metadata_handle is None or stats_accumulator is None:
            raise RuntimeError("Streaming metadata writer is required for generation")

        self._configure_generation(**kwargs)
        self.template_counts = Counter()
        self.family_counts = Counter()
        self._recent_signatures = deque(maxlen=self.novelty_window)

        for idx in tqdm(range(num_images), desc="Generating markdown images"):
            sample_index = sample_start_index + idx
            image, meta = self.generate_single(sample_index=sample_index)

            filename = f"markdown_{sample_index:05d}.png"
            self.save_image(image, filename)
            meta["file_name"] = str(self.output_dir / filename)
            self.append_metadata(metadata_handle, stats_accumulator, meta)

        return num_images

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
        merge_order: List[str] = []
        composition_metadata: Dict[str, Any] = {
            "document_shape": selected_template.family,
            "block_types": list(merge_order),
            "block_type_counts": dict(Counter(merge_order)),
            "section_count": len(merge_order),
        }

        for attempt in range(self.novelty_max_attempts):
            sample_seed = self._derive_sample_seed(sample_index, attempt)
            self._seed_for_sample(sample_seed)

            selected_template, selected_weight = self._select_template_spec()

            original_markdown = self.data_generator.generate_markdown(
                template_id=selected_template.template_id,
                template_spec=selected_template,
            )
            merge_order = self.data_generator.pop_merge_order()
            if hasattr(self.data_generator, "pop_composition_metadata"):
                composition_metadata = self.data_generator.pop_composition_metadata()
            else:
                composition_metadata = {
                    "document_shape": selected_template.family,
                    "block_types": list(merge_order),
                    "block_type_counts": dict(Counter(merge_order)),
                    "section_count": len(merge_order),
                }
            markdown_text, mutation_count = self._mutate_text_generator_sections(
                original_markdown,
                self.similar_char_ratio,
                merge_order,
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
        elif self.markdown_renderer == "playwright":
            renderer = PlaywrightMarkdownRenderer(font_path, style)
        else:
            renderer = MarkdownRenderer(font_path, style)
        image = renderer.render(markdown_text)

        self.template_counts[selected_template.template_id] += 1
        self.family_counts[selected_template.family] += 1
        self._recent_signatures.append(signature)

        generated_count = max(1, sum(self.family_counts.values()))
        family_ratio = self.family_counts[selected_template.family] / generated_count

        metadata = {
            "template": selected_template.template_id,
            "template_id": selected_template.template_id,
            "template_family": selected_template.family,
            "document_family": selected_template.family,
            "document_shape": composition_metadata.get(
                "document_shape",
                selected_template.family,
            ),
            "block_types": list(composition_metadata.get("block_types", merge_order)),
            "block_type_counts": dict(
                composition_metadata.get(
                    "block_type_counts",
                    Counter(merge_order),
                )
            ),
            "section_count": int(
                composition_metadata.get(
                    "section_count",
                    len(merge_order),
                )
            ),
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
            "merge_order": merge_order,
            "image_width": image.width,
            "image_height": image.height,
        }
        return image, metadata

    @staticmethod
    def _base_styles() -> List[MarkdownStyle]:
        return base_styles()

    @staticmethod
    def _clamp_color(value: int) -> int:
        return clamp_color(value)

    def _jitter_color(self, color: Tuple[int, int, int], span: int) -> Tuple[int, int, int]:
        return jitter_color(color, span)

    def _random_style(self) -> MarkdownStyle:
        """Generate random style variations."""
        return random_style(self.style_profile)
