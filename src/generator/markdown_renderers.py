import importlib
import logging
import random
import tempfile
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from generator.markdown_render_utils import (
    MarkdownStyle,
    image_to_data_uri,
    parse_markdown_formula_line,
    parse_markdown_image_line,
    render_formula_image,
)

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _formula_font_size(style: MarkdownStyle) -> int:
        return int(style.body_font_size * 0.9)

    @staticmethod
    def _resolve_image_asset(
        image_src: str,
        image_assets: Optional[Dict[str, Image.Image]],
    ) -> Optional[Image.Image]:
        if image_assets:
            cached = image_assets.get(image_src)
            if cached is not None:
                return cached

        candidate_path: Optional[Path] = None
        if image_src.startswith("file://"):
            candidate_path = Path(image_src[len("file://") :])
        elif "://" not in image_src:
            candidate_path = Path(image_src)

        if candidate_path is None or not candidate_path.exists() or not candidate_path.is_file():
            return None

        try:
            loaded = Image.open(candidate_path).convert("RGB")
            loaded.load()
            return loaded
        except Exception:
            return None

    @staticmethod
    def _fit_media_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
        safe_max_width = max(1, int(max_width))
        safe_max_height = max(1, int(max_height))
        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        ratio = min(safe_max_width / width, safe_max_height / height)
        if ratio >= 1.0:
            return image.copy()

        new_width = max(1, int(width * ratio))
        new_height = max(1, int(height * ratio))
        if hasattr(Image, "Resampling"):
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image.resize((new_width, new_height), getattr(Image, "LANCZOS", 1))

    def _image_block_height(
        self,
        style: MarkdownStyle,
        image_src: str,
        image_assets: Optional[Dict[str, Image.Image]],
    ) -> int:
        fallback = self._image_placeholder_height(style)
        image_asset = self._resolve_image_asset(image_src, image_assets)
        if image_asset is None:
            return fallback

        available_width = max(30, style.content_width - 12)
        estimated_height = int(available_width * image_asset.height / max(1, image_asset.width))
        max_height = max(fallback, int(style.content_width * 0.85))
        return max(fallback, min(max_height, estimated_height))

    def render(
        self,
        markdown_text: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> Image.Image:
        lines = markdown_text.split("\n")
        style = self.style

        total_height = style.margin_top + style.margin_bottom
        line_heights = []

        for line in lines:
            height = self._get_line_height(line, image_assets=image_assets)
            line_heights.append(height)
            total_height += height

        width = style.margin_left + style.content_width + style.margin_right
        height = max(total_height, 200)

        img = Image.new("RGB", (width, int(height)), style.background_color)
        draw = ImageDraw.Draw(img)

        current_y = style.margin_top
        in_code_block = False
        code_block_start_y = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_block_start_y = current_y
                else:
                    draw.rectangle(
                        [
                            style.margin_left - 5,
                            code_block_start_y - 5,
                            style.margin_left + style.content_width + 5,
                            current_y + 5,
                        ],
                        fill=style.code_bg_color,
                    )
                    in_code_block = False

                current_y += line_heights[i]
                continue

            if in_code_block:
                current_y = self._draw_code_line(draw, line, current_y, style)
            elif (image_payload := self._parse_image_line(stripped)) is not None:
                current_y = self._draw_image_block(
                    img,
                    draw,
                    image_payload[0],
                    image_payload[1],
                    current_y,
                    style,
                    image_assets,
                )
            elif (formula_text := self._parse_formula_line(stripped)) is not None:
                current_y = self._draw_formula_line(img, draw, formula_text, current_y, style)
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

        return self._apply_effects(img, style)

    def _get_line_height(
        self,
        line: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> int:
        stripped = line.strip()
        base_spacing = int(self.style.line_spacing * self.style.body_font_size)

        if stripped.startswith("# "):
            return int(self.style.h1_font_size * self.style.line_spacing) + 10
        if stripped.startswith("## "):
            return int(self.style.h2_font_size * self.style.line_spacing) + 8
        if stripped.startswith("### "):
            return int(self.style.h3_font_size * self.style.line_spacing) + 6
        if (image_payload := self._parse_image_line(stripped)) is not None:
            return self._image_block_height(self.style, image_payload[1], image_assets) + 14
        if (formula_text := self._parse_formula_line(stripped)) is not None:
            formula_image = render_formula_image(
                formula_text,
                self._formula_font_size(self.style),
                self.style.code_text_color,
            )
            if formula_image is not None:
                return max(base_spacing + 18, formula_image.height + 20)
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
        draw.text((style.margin_left, y), text, font=self.h1_font, fill=style.h1_color)
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h1_font)
        line_y = int(bbox[3]) + 5
        draw.line(
            [(style.margin_left, line_y), (style.margin_left + style.content_width, line_y)],
            fill=style.h2_color,
            width=2,
        )
        return int(line_y + 15)

    def _draw_h2(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        draw.text((style.margin_left, y), text, font=self.h2_font, fill=style.h2_color)
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h2_font)
        return int(bbox[3] + 12)

    def _draw_h3(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        draw.text((style.margin_left, y), text, font=self.h3_font, fill=style.h3_color)
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h3_font)
        return int(bbox[3] + 10)

    def _draw_paragraph(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
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
            if "`" in line:
                y = self._draw_inline_code_line(draw, line, y, style)
            else:
                draw.text((style.margin_left, y), line, font=self.body_font, fill=style.text_color)
                y += int(style.body_font_size * style.line_spacing)

        return y + 5

    def _draw_inline_code_line(
        self,
        draw: ImageDraw.ImageDraw,
        line: str,
        y: int,
        style: MarkdownStyle,
    ) -> int:
        x = style.margin_left
        parts = line.split("`")

        for i, part in enumerate(parts):
            if i % 2 == 1:
                bbox = draw.textbbox((x, y), part, font=self.code_font)
                draw.rectangle([x - 2, y - 1, bbox[2] + 2, bbox[3] + 1], fill=style.code_bg_color)
                draw.text((x, y), part, font=self.code_font, fill=style.code_text_color)
                x = bbox[2] + 4
            else:
                draw.text((x, y), part, font=self.body_font, fill=style.text_color)
                bbox = draw.textbbox((x, y), part, font=self.body_font)
                x = bbox[2]

        return y + int(style.body_font_size * style.line_spacing)

    def _draw_code_line(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        draw.text((style.margin_left + 10, y), text, font=self.code_font, fill=style.code_text_color)
        return y + int(style.code_font_size * style.line_spacing)

    def _draw_blockquote(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        draw.line(
            [(style.margin_left, y), (style.margin_left, y + style.body_font_size + 10)],
            fill=style.blockquote_border_color,
            width=3,
        )
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
        marker = f"{number}." if ordered and number else "*"
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
        box_size = style.body_font_size - 2
        box_x = style.margin_left
        box_y = y + 2

        draw.rectangle([box_x, box_y, box_x + box_size, box_y + box_size], outline=style.text_color)
        if checked:
            draw.line(
                [(box_x + 2, box_y + box_size // 2), (box_x + box_size // 2, box_y + box_size - 2)],
                fill=style.text_color,
                width=2,
            )
            draw.line(
                [(box_x + box_size // 2, box_y + box_size - 2), (box_x + box_size - 2, box_y + 2)],
                fill=style.text_color,
                width=2,
            )

        draw.text((box_x + box_size + 8, y), text.strip(), font=self.body_font, fill=style.text_color)
        return y + int(style.body_font_size * style.line_spacing)

    def _draw_table_row(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        if text.replace("|", "").replace("-", "").strip() == "":
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
        draw.line(
            [(style.margin_left, y + 10), (style.margin_left + style.content_width, y + 10)],
            fill=(200, 200, 200),
            width=1,
        )
        return y + 20

    def _draw_formula_line(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        formula_text: str,
        y: int,
        style: MarkdownStyle,
    ) -> int:
        text = formula_text.strip()
        formula_image = render_formula_image(
            text,
            self._formula_font_size(style),
            style.code_text_color,
        )

        box_left = style.margin_left
        box_top = y + 1
        box_right = style.margin_left + style.content_width

        if formula_image is not None:
            available_width = max(20, style.content_width - 16)
            available_height = max(20, int(style.content_width * 0.45))
            rendered_formula = self._fit_media_image(formula_image, available_width, available_height)
            box_bottom = box_top + rendered_formula.height + 10

            draw.rectangle(
                [box_left, box_top, box_right, box_bottom],
                fill=style.code_bg_color,
                outline=style.blockquote_border_color,
                width=1,
            )

            formula_x = box_left + max(8, (style.content_width - rendered_formula.width) // 2)
            formula_y = box_top + 5
            canvas.paste(rendered_formula, (formula_x, formula_y), rendered_formula)
            return int(box_bottom + 8)

        text_y = y + 4
        text_bbox = draw.textbbox((0, 0), text, font=self.code_font)
        text_width = max(1, text_bbox[2] - text_bbox[0])
        x = style.margin_left + max(8, (style.content_width - text_width) // 2)
        bbox = draw.textbbox((x, text_y), text, font=self.code_font)
        box_bottom = bbox[3] + 5

        draw.rectangle(
            [box_left, box_top, box_right, box_bottom],
            fill=style.code_bg_color,
            outline=style.blockquote_border_color,
            width=1,
        )
        draw.text((x, text_y), text, font=self.code_font, fill=style.code_text_color)
        return int(box_bottom + 8)

    def _draw_image_block(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        alt_text: str,
        image_src: str,
        y: int,
        style: MarkdownStyle,
        image_assets: Optional[Dict[str, Image.Image]],
    ) -> int:
        block_height = self._image_block_height(style, image_src, image_assets)
        left = style.margin_left
        top = y + 4
        right = style.margin_left + style.content_width
        bottom = top + block_height

        image_asset = self._resolve_image_asset(image_src, image_assets)
        if image_asset is not None:
            draw.rectangle(
                [left, top, right, bottom],
                fill=(250, 250, 250),
                outline=(175, 175, 175),
                width=1,
            )

            max_width = max(24, style.content_width - 12)
            max_height = max(24, block_height - 12)
            rendered_image = self._fit_media_image(image_asset.convert("RGB"), max_width, max_height)
            paste_x = left + (style.content_width - rendered_image.width) // 2
            paste_y = top + (block_height - rendered_image.height) // 2
            canvas.paste(rendered_image, (paste_x, paste_y))
            return int(bottom + 10)

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
        label_y = top + max(8, (block_height - (label_bbox[3] - label_bbox[1])) // 2)
        draw.rectangle(
            [
                label_x - 6,
                label_y - 3,
                label_x + (label_bbox[2] - label_bbox[0]) + 6,
                label_y + (label_bbox[3] - label_bbox[1]) + 3,
            ],
            fill=(255, 255, 255),
        )
        draw.text((label_x, label_y), label, font=self.body_font, fill=(90, 90, 90))

        return int(bottom + 10)

    def _draw_italic(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        draw.text((style.margin_left, y), text, font=self.body_font, fill=(100, 100, 100))
        return y + int(style.body_font_size * style.line_spacing)

    def _apply_effects(self, img: Image.Image, style: MarkdownStyle) -> Image.Image:
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
            extensions=["extra", "tables", "fenced_code", "sane_lists", "nl2br"],
        )

    def _prepare_component_markdown(
        self,
        markdown_text: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> str:
        prepared_lines: List[str] = []
        for raw_line in markdown_text.splitlines():
            stripped = raw_line.strip()

            image_payload = parse_markdown_image_line(stripped)
            if image_payload is not None:
                alt_text = image_payload[0] or "Image"
                safe_alt = escape(alt_text)
                image_src = image_payload[1]
                image_asset = MarkdownRenderer._resolve_image_asset(image_src, image_assets)
                if image_asset is not None:
                    data_uri = image_to_data_uri(image_asset)
                    prepared_lines.extend(
                        [
                            '<figure class="md-image-placeholder">',
                            f'  <img class="md-image-rendered" src="{data_uri}" alt="{safe_alt}" />',
                            f"  <figcaption>{safe_alt}</figcaption>",
                            "</figure>",
                        ]
                    )
                else:
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
                formula_image = render_formula_image(
                    formula_text,
                    MarkdownRenderer._formula_font_size(self.style),
                    self.style.code_text_color,
                )
                if formula_image is not None:
                    formula_data_uri = image_to_data_uri(formula_image)
                    prepared_lines.append(
                        f'<div class="md-formula"><img class="md-formula-img" src="{formula_data_uri}" alt="formula" /></div>'
                    )
                else:
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
            formula_text = parse_markdown_formula_line(line)
            if formula_text:
                formula_image = render_formula_image(
                    formula_text,
                    MarkdownRenderer._formula_font_size(self.style),
                    self.style.code_text_color,
                )
                if formula_image is not None:
                    formula_bonus += formula_image.height + 12
                else:
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

    def _build_html_document(
        self,
        markdown_text: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> str:
        prepared_markdown = self._prepare_component_markdown(markdown_text, image_assets=image_assets)
        rendered_html = self._coerce_markdown_html(prepared_markdown)
        page_width = self.style.margin_left + self.style.content_width + self.style.margin_right
        css = f"""
@page {{
  margin: 12mm 10mm 14mm 10mm;
}}
@font-face {{
  font-family: 'RenderFont';
  src: url('file://{escape(self.font_path)}') format('truetype');
}}
html, body {{
  margin: 0;
  padding: 0;
  min-width: {page_width}px;
  width: auto;
  background: rgb{self.style.background_color};
  overflow: visible;
}}
*, *::before, *::after {{
  box-sizing: border-box;
}}
.markdown-body {{
  width: {page_width}px;
  padding: {self.style.margin_top}px {self.style.margin_right}px {self.style.margin_bottom}px {self.style.margin_left}px;
  color: rgb{self.style.text_color};
  font-family: 'RenderFont', sans-serif;
  font-size: {self.style.body_font_size}px;
  line-height: {self.style.line_spacing};
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}
.markdown-body h1 {{ font-size: {self.style.h1_font_size}px; color: rgb{self.style.h1_color}; margin: 0 0 16px 0; }}
.markdown-body h2 {{ font-size: {self.style.h2_font_size}px; color: rgb{self.style.h2_color}; margin: 18px 0 12px 0; }}
.markdown-body h3 {{ font-size: {self.style.h3_font_size}px; color: rgb{self.style.h3_color}; margin: 16px 0 8px 0; }}
.markdown-body a {{ color: rgb{self.style.link_color}; text-decoration: none; }}
.markdown-body p {{ margin: 0 0 10px 0; }}
.markdown-body p,
.markdown-body li,
.markdown-body td,
.markdown-body th {{
  line-height: {max(1.35, self.style.line_spacing)};
}}
.markdown-body br {{
  display: block;
  margin: 0;
  line-height: {self.style.line_spacing};
}}
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
  margin: 4px 0 16px 0;
  table-layout: auto;
  background: rgba(255, 255, 255, 0.92);
  break-inside: avoid;
  page-break-inside: avoid;
}}
.markdown-body th, .markdown-body td {{
  border: 1px solid rgba(0, 0, 0, 0.25);
  text-align: left;
  padding: 8px 12px;
  vertical-align: top;
  overflow-wrap: break-word;
  word-break: normal;
}}
.markdown-body th {{
  background: rgba(0, 0, 0, 0.06);
  font-weight: 600;
}}
.markdown-body tbody tr:nth-child(even) td {{
  background: rgba(0, 0, 0, 0.025);
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
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}}
.markdown-body .md-image-placeholder {{
  margin: 0 0 12px 0;
  break-inside: avoid;
  page-break-inside: avoid;
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
.markdown-body .md-image-rendered {{
  width: auto;
  max-width: 100%;
  display: block;
  border: 1px solid rgba(0, 0, 0, 0.28);
  object-fit: contain;
  max-height: 520px;
  margin: 0 auto;
}}
.markdown-body .md-image-placeholder figcaption {{
  margin-top: 6px;
  color: rgba(0, 0, 0, 0.65);
  font-size: {max(10, self.style.body_font_size - 1)}px;
}}
.markdown-body .md-formula .md-formula-img {{
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
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

    def render(
        self,
        markdown_text: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> Image.Image:
        try:
            Html2Image = importlib.import_module("html2image").Html2Image
        except ImportError as exc:
            raise RuntimeError(
                "html2image package is required for html->image rendering. "
                "Install with: uv sync --group generate"
            ) from exc

        width = self.style.margin_left + self.style.content_width + self.style.margin_right
        height = self._estimate_viewport_height(markdown_text)
        html_doc = self._build_html_document(markdown_text, image_assets=image_assets)

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

        return self._apply_effects(image)


class PlaywrightMarkdownRenderer(HtmlMarkdownRenderer):
    _CAPTURE_PADDING_PX = 8

    @staticmethod
    def _load_playwright_sync_api() -> Any:
        try:
            return importlib.import_module("playwright.sync_api")
        except ImportError as exc:
            raise RuntimeError(
                "playwright package is required for headless markdown rendering. "
                "Install with: uv sync --group generate && uv run playwright install chromium"
            ) from exc

    def render(
        self,
        markdown_text: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> Image.Image:
        playwright_sync_api = self._load_playwright_sync_api()
        sync_playwright = playwright_sync_api.sync_playwright

        width = self.style.margin_left + self.style.content_width + self.style.margin_right
        capture_padding = self._CAPTURE_PADDING_PX
        viewport_height = max(720, min(1600, self._estimate_viewport_height(markdown_text) + (capture_padding * 2)))
        html_doc = self._build_html_document(markdown_text, image_assets=image_assets)
        html_doc = html_doc.replace(
            '<body>\n  <div class="markdown-body">',
            (
                '<body>\n'
                f'  <div class="capture-shell" style="padding: {capture_padding}px; width: {width + (capture_padding * 2)}px; overflow: visible;">\n'
                '    <div class="markdown-body">'
            ),
            1,
        ).replace("</div>\n</body>", "</div>\n  </div>\n</body>", 1)

        with tempfile.TemporaryDirectory(prefix="markdown-playwright-") as temp_dir:
            html_path = Path(temp_dir) / "rendered.html"
            screenshot_path = Path(temp_dir) / "rendered.png"
            html_path.write_text(html_doc, encoding="utf-8")

            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        headless=True,
                        args=[
                            "--hide-scrollbars",
                            "--disable-gpu",
                            "--force-device-scale-factor=1",
                        ],
                    )
                    page = browser.new_page(
                        viewport={"width": width + (capture_padding * 2), "height": viewport_height},
                        device_scale_factor=1,
                    )
                    page.goto(html_path.as_uri(), wait_until="load")
                    page.wait_for_function("() => Array.from(document.images).every((img) => img.complete)")
                    page.evaluate(
                        "() => document.fonts ? document.fonts.ready.then(() => true) : true"
                    )
                    page.locator(".capture-shell").screenshot(
                        path=str(screenshot_path),
                        animations="disabled",
                    )
                    browser.close()
            except Exception as exc:
                raise RuntimeError(
                    "Headless Playwright markdown rendering failed. "
                    "Ensure Chromium is installed with: uv run playwright install chromium"
                ) from exc

            image = Image.open(screenshot_path).convert("RGB")
            image.load()

        return self._apply_effects(image)
