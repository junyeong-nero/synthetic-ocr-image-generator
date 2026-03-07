import base64
import importlib
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

_MARKDOWN_IMAGE_PATTERN = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)$")
_MARKDOWN_FORMULA_PATTERN = re.compile(r"^\$\$\s*(?P<formula>.+?)\s*\$\$$")
_SIMPLE_SUPERSCRIPT_PATTERN = re.compile(r"\^(?P<atom>[A-Za-z0-9])")
_SIMPLE_SUBSCRIPT_PATTERN = re.compile(r"_(?P<atom>[A-Za-z0-9])")
_CHAINED_SUPERSCRIPT_PATTERN = re.compile(r"\^\{(?P<first>[^{}]+)\}\^\{(?P<second>[^{}]+)\}")
_CHAINED_SUBSCRIPT_PATTERN = re.compile(r"_\{(?P<first>[^{}]+)\}_\{(?P<second>[^{}]+)\}")
_CHAINED_SUPERSCRIPT_SIMPLE_PATTERN = re.compile(r"\^\{(?P<first>[^{}]+)\}\^(?P<second>[A-Za-z0-9])")
_CHAINED_SUBSCRIPT_SIMPLE_PATTERN = re.compile(r"_\{(?P<first>[^{}]+)\}_(?P<second>[A-Za-z0-9])")
_SUPERSUBSUPER_PATTERN = re.compile(
    r"\^\{(?P<sup>[^{}]+)\}_\{(?P<sub>[^{}]+)\}\^\{(?P<extra_sup>[^{}]+)\}"
)
_SUBSUPSUB_PATTERN = re.compile(
    r"_\{(?P<sub>[^{}]+)\}\^\{(?P<sup>[^{}]+)\}_\{(?P<extra_sub>[^{}]+)\}"
)

_FORMULA_IMAGE_CACHE_MAX_ITEMS = 256
_FORMULA_IMAGE_CACHE: "OrderedDict[Tuple[str, int, Tuple[int, int, int]], Optional[Image.Image]]" = OrderedDict()
_LATEX_TO_IMAGE_RENDERER: Optional[Any] = None
_LATEX_TO_IMAGE_LOADED = False


def _get_cached_formula_image(
    cache_key: Tuple[str, int, Tuple[int, int, int]],
) -> Tuple[bool, Optional[Image.Image]]:
    if cache_key not in _FORMULA_IMAGE_CACHE:
        return False, None

    cached = _FORMULA_IMAGE_CACHE.pop(cache_key)
    _FORMULA_IMAGE_CACHE[cache_key] = cached
    return True, cached.copy() if cached is not None else None


def _store_cached_formula_image(
    cache_key: Tuple[str, int, Tuple[int, int, int]],
    image: Optional[Image.Image],
) -> None:
    if cache_key in _FORMULA_IMAGE_CACHE:
        _FORMULA_IMAGE_CACHE.pop(cache_key)
    _FORMULA_IMAGE_CACHE[cache_key] = image.copy() if image is not None else None

    while len(_FORMULA_IMAGE_CACHE) > _FORMULA_IMAGE_CACHE_MAX_ITEMS:
        _FORMULA_IMAGE_CACHE.popitem(last=False)


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


def _get_latex_to_image_renderer() -> Optional[Any]:
    global _LATEX_TO_IMAGE_RENDERER, _LATEX_TO_IMAGE_LOADED
    if _LATEX_TO_IMAGE_LOADED:
        return _LATEX_TO_IMAGE_RENDERER

    _LATEX_TO_IMAGE_LOADED = True
    try:
        latex_module = importlib.import_module("latex_to_image")
        renderer_cls = getattr(latex_module, "LaTeXToImg")
        _LATEX_TO_IMAGE_RENDERER = renderer_cls()
    except Exception:
        _LATEX_TO_IMAGE_RENDERER = None
    return _LATEX_TO_IMAGE_RENDERER


def _formula_array_to_rgba(
    formula_array: np.ndarray,
    text_color: Tuple[int, int, int],
) -> Optional[Image.Image]:
    if not isinstance(formula_array, np.ndarray) or formula_array.size == 0:
        return None

    image_data = np.asarray(formula_array)
    if image_data.ndim == 2:
        grayscale = image_data.astype(np.uint8)
    elif image_data.ndim == 3 and image_data.shape[2] >= 3:
        b_channel = image_data[..., 0].astype(np.float32)
        g_channel = image_data[..., 1].astype(np.float32)
        r_channel = image_data[..., 2].astype(np.float32)
        grayscale = np.clip(0.114 * b_channel + 0.587 * g_channel + 0.299 * r_channel, 0, 255).astype(np.uint8)
    else:
        return None

    alpha_black_on_white = (255 - grayscale).astype(np.uint8)
    alpha_white_on_black = grayscale.astype(np.uint8)
    alpha_black_on_white[alpha_black_on_white < 10] = 0
    alpha_white_on_black[alpha_white_on_black < 10] = 0

    def border_nonzero_ratio(alpha: np.ndarray) -> float:
        border = np.concatenate([alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]])
        if border.size == 0:
            return 1.0
        return float(np.count_nonzero(border)) / float(border.size)

    def nonzero_ratio(alpha: np.ndarray) -> float:
        total = alpha.size
        if total <= 0:
            return 1.0
        return float(np.count_nonzero(alpha)) / float(total)

    black_on_white_border = border_nonzero_ratio(alpha_black_on_white)
    white_on_black_border = border_nonzero_ratio(alpha_white_on_black)
    if white_on_black_border < black_on_white_border:
        alpha = alpha_white_on_black
    elif black_on_white_border < white_on_black_border:
        alpha = alpha_black_on_white
    else:
        alpha = (
            alpha_white_on_black
            if nonzero_ratio(alpha_white_on_black) < nonzero_ratio(alpha_black_on_white)
            else alpha_black_on_white
        )

    colored = np.zeros((grayscale.shape[0], grayscale.shape[1], 4), dtype=np.uint8)
    colored[..., 0] = int(text_color[0])
    colored[..., 1] = int(text_color[1])
    colored[..., 2] = int(text_color[2])
    colored[..., 3] = alpha

    image = Image.fromarray(colored, mode="RGBA")
    bbox = image.getbbox()
    if bbox is None:
        return None
    return image.crop(bbox)


def _normalize_chained_scripts(expression: str) -> str:
    normalized = _SIMPLE_SUPERSCRIPT_PATTERN.sub(r"^{\g<atom>}", expression)
    normalized = _SIMPLE_SUBSCRIPT_PATTERN.sub(r"_{\g<atom>}", normalized)

    for _ in range(6):
        updated = _CHAINED_SUPERSCRIPT_PATTERN.sub(r"^{\g<first>^{\g<second>}}", normalized)
        updated = _CHAINED_SUBSCRIPT_PATTERN.sub(r"_{\g<first>_{\g<second>}}", updated)
        updated = _CHAINED_SUPERSCRIPT_SIMPLE_PATTERN.sub(r"^{\g<first>^\g<second>}", updated)
        updated = _CHAINED_SUBSCRIPT_SIMPLE_PATTERN.sub(r"_{\g<first>_\g<second>}", updated)
        updated = _SUPERSUBSUPER_PATTERN.sub(r"^{\g<sup>^{\g<extra_sup>}}_{\g<sub>}", updated)
        updated = _SUBSUPSUB_PATTERN.sub(r"_{\g<sub>_{\g<extra_sub>}}^{\g<sup>}", updated)
        if updated == normalized:
            break
        normalized = updated
    return normalized


def _render_formula_array_with_latex_tools(renderer: Any, expression: str) -> Optional[np.ndarray]:
    latex_engine = getattr(renderer, "latex", None)
    if latex_engine is None:
        return None

    def compile_formula(math_expression: str) -> Optional[np.ndarray]:
        template = (
            "\\documentclass[12pt]{article}\n"
            "\\usepackage{amsmath,amssymb,amsfonts,mathtools,bm}\n"
            "\\pagestyle{empty}\n"
            "\\begin{document}\n"
            f"${math_expression}$\n"
            "\\end{document}\n"
        )

        work_dir = tempfile.gettempdir()
        tex_file_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".tex",
                prefix="eq-",
                dir=work_dir,
                delete=False,
            ) as handle:
                tex_file_path = handle.name
                handle.write(template)

            command = (
                "xelatex "
                "-interaction nonstopmode "
                "-halt-on-error "
                "-file-line-error "
                f"-output-directory {work_dir} "
                f"{tex_file_path}"
            )
            _, return_code = latex_engine.run_cmd(command)
            pdf_file = Path(tex_file_path).with_suffix(".pdf")
            if return_code != 0 or not pdf_file.exists() or pdf_file.stat().st_size <= 0:
                return None

            image_array = latex_engine.convert_pdf_to_png(pdf_file)
            if image_array is None:
                return None

            cropper = getattr(renderer, "cropper", None)
            if cropper is None:
                return image_array

            try:
                cropped = cropper(image_array)
            except Exception:
                return image_array
            return image_array if cropped is None else cropped
        except Exception:
            return None
        finally:
            if tex_file_path:
                try:
                    latex_engine.clear_files(tex_file_path)
                except Exception:
                    pass

    rendered = compile_formula(expression)
    if rendered is not None:
        return rendered

    normalized_expression = _normalize_chained_scripts(expression)
    if normalized_expression == expression:
        return None

    return compile_formula(normalized_expression)


def render_formula_image(
    formula_text: str,
    font_size: int,
    text_color: Tuple[int, int, int],
) -> Optional[Image.Image]:
    expression = formula_text.strip()
    if not expression:
        return None

    cache_key = (expression, int(font_size), text_color)
    cache_hit, cached_image = _get_cached_formula_image(cache_key)
    if cache_hit:
        return cached_image

    renderer = _get_latex_to_image_renderer()
    if renderer is None:
        _store_cached_formula_image(cache_key, None)
        return None

    rendered_array = _render_formula_array_with_latex_tools(renderer, expression)
    if rendered_array is None:
        _store_cached_formula_image(cache_key, None)
        return None

    formula_image = _formula_array_to_rgba(rendered_array, text_color)
    if formula_image is None:
        _store_cached_formula_image(cache_key, None)
        return None

    scale = max(0.35, min(3.0, float(max(6, int(font_size))) / 24.0))
    if abs(scale - 1.0) >= 0.05:
        target_size = (
            max(1, int(formula_image.width * scale)),
            max(1, int(formula_image.height * scale)),
        )
        formula_image = formula_image.resize(target_size, Image.Resampling.LANCZOS)

    _store_cached_formula_image(cache_key, formula_image)
    return formula_image.copy()


def image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    if image.mode in {"RGBA", "LA"}:
        image.save(buffer, format="PNG")
    else:
        image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@dataclass
class MarkdownStyle:
    margin_top: int = 40
    margin_bottom: int = 40
    margin_left: int = 40
    margin_right: int = 40
    content_width: int = 600
    line_spacing: float = 1.5
    h1_font_size: int = 28
    h2_font_size: int = 22
    h3_font_size: int = 18
    body_font_size: int = 14
    code_font_size: int = 12
    text_color: Tuple[int, int, int] = (33, 33, 33)
    h1_color: Tuple[int, int, int] = (0, 0, 0)
    h2_color: Tuple[int, int, int] = (50, 50, 50)
    h3_color: Tuple[int, int, int] = (70, 70, 70)
    link_color: Tuple[int, int, int] = (0, 102, 204)
    code_bg_color: Tuple[int, int, int] = (245, 245, 245)
    code_text_color: Tuple[int, int, int] = (0, 0, 0)
    blockquote_color: Tuple[int, int, int] = (100, 100, 100)
    blockquote_border_color: Tuple[int, int, int] = (200, 200, 200)
    background_color: Tuple[int, int, int] = (255, 255, 255)
    add_noise: bool = True
    add_blur: bool = False
    add_contrast: bool = False
