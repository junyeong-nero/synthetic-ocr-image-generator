"""Apply a DistributionProfile to a single sample's style and rendered image."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from src.generator.degradation import apply_capture_degradation
from src.generator.distribution_profile import (
    CaptureSample,
    DistributionProfile,
    points_to_css_px,
    render_scale_for_dpi,
)
from src.generator.markdown_render_utils import MarkdownStyle


@dataclass
class ProfileRenderPlan:
    profile_id: str
    capture: CaptureSample
    render_scale: float
    rng: random.Random
    body_font_pt: Optional[float] = None
    typography: Dict[str, Any] = field(default_factory=dict)
    page: Dict[str, Any] = field(default_factory=dict)
    page_trimmed: bool = False

    def metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "distribution_profile": self.profile_id,
            "capture_channel": self.capture.channel,
            "target_dpi": int(self.capture.dpi),
            "render_scale": float(round(self.render_scale, 4)),
            "colored_background": bool(self.typography.get("colored_background", True)),
            "visual_difficulty": self.capture.difficulty(),
            "degradation_params": dict(self.capture.params),
        }
        if self.page.get("aspect_ratio"):
            metadata["page_aspect_ratio"] = float(self.page["aspect_ratio"])
            metadata["page_trimmed"] = bool(self.page_trimmed)
        # Hub schemas are inferred from the first row, so never emit None-typed values.
        if self.body_font_pt is not None:
            metadata["body_font_pt"] = float(self.body_font_pt)
        return metadata


def page_width_css(style: MarkdownStyle) -> int:
    return int(style.margin_left + style.content_width + style.margin_right)


def plan_profile_render(
    profile: DistributionProfile,
    style: MarkdownStyle,
    rng: random.Random,
) -> ProfileRenderPlan:
    """Mutate ``style`` in place according to the profile and return the plan."""
    typography = profile.sample_typography(rng)
    width_css = page_width_css(style)

    body_pt = typography.get("body_font_pt")
    if body_pt:
        new_body = points_to_css_px(float(body_pt), width_css)
        ratio = new_body / max(1, style.body_font_size)
        style.body_font_size = new_body
        style.code_font_size = max(8, int(round(style.code_font_size * ratio)))
        style.h1_font_size = max(new_body + 4, int(round(style.h1_font_size * ratio)))
        style.h2_font_size = max(new_body + 2, int(round(style.h2_font_size * ratio)))
        style.h3_font_size = max(new_body + 1, int(round(style.h3_font_size * ratio)))

    line_spacing = typography.get("line_spacing")
    if line_spacing:
        style.line_spacing = float(line_spacing)

    spacing_scale = typography.get("spacing_scale")
    if spacing_scale:
        style.spacing_scale = float(spacing_scale)

    # Real pages are overwhelmingly printed on white paper; the base style
    # sampler's pastel backgrounds are kept only at the profile's rate
    # (OmniDocBench tracks this as the `colorful_background` page attribute).
    colored = typography.get("colored_background")
    if colored is not None and not colored:
        style.background_color = (255, 255, 255)
        style.code_bg_color = (246, 246, 246)

    # Printed text is near-black; the base sampler's grey/blue text is rare.
    ink_gray = typography.get("ink_gray")
    if ink_gray is not None:
        gray = max(0, min(120, int(ink_gray)))
        style.text_color = (gray, gray, gray)
        style.code_text_color = (gray, gray, gray)
        # Headings keep the base sampler's accent colour only at the
        # profile's `colored_headings` rate (default: always, as before).
        keep_heading_color = typography.get("colored_headings", True)
        if not keep_heading_color:
            style.h1_color = style.h2_color = style.h3_color = (gray, gray, gray)

    capture = profile.sample_capture(rng)
    # Profile degradations replace the renderer's legacy noise/blur/contrast.
    style.add_noise = False
    style.add_blur = False
    style.add_contrast = False
    style.render_scale = render_scale_for_dpi(capture.dpi, width_css)

    return ProfileRenderPlan(
        profile_id=profile.profile_id,
        capture=capture,
        render_scale=style.render_scale,
        rng=rng,
        body_font_pt=float(body_pt) if body_pt else None,
        typography=typography,
        page=profile.sample_page(rng),
    )


def finalize_profile_image(
    image: Image.Image,
    plan: ProfileRenderPlan,
    *,
    renderer_applied_scale: bool,
) -> Image.Image:
    if not renderer_applied_scale and abs(plan.render_scale - 1.0) > 1e-3:
        new_size = (
            max(1, int(round(image.width * plan.render_scale))),
            max(1, int(round(image.height * plan.render_scale))),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    aspect = plan.page.get("aspect_ratio")
    if aspect:
        image = pad_to_aspect(image, float(aspect))
    return apply_capture_degradation(
        image,
        plan.capture.params,
        plan.rng,
        channel=plan.capture.channel,
    )


def pad_to_aspect(image: Image.Image, aspect_ratio: float) -> Image.Image:
    """Place short content on a full sheet (height / width = ``aspect_ratio``).

    Real pages keep their paper size even when text ends early, so the page
    is extended downwards with the paper colour. Content taller than one sheet
    is left untouched so the ground truth stays complete.
    """
    target_height = int(round(image.width * max(0.5, aspect_ratio)))
    if target_height <= image.height:
        return image
    rgb = image.convert("RGB")
    paper = rgb.getpixel((rgb.width - 1, rgb.height - 1))
    sheet = Image.new("RGB", (rgb.width, target_height), paper)
    sheet.paste(rgb, (0, 0))
    return sheet


def sheet_overflow_ratio(image: Image.Image, plan: ProfileRenderPlan) -> float:
    """How many sheets tall the rendered content is (1.0 = exactly one page)."""
    aspect = plan.page.get("aspect_ratio")
    if not aspect or image.width <= 0:
        return 1.0
    return image.height / (image.width * float(aspect))


def fit_markdown_to_sheet(markdown_text: str, overflow_ratio: float) -> Tuple[str, int]:
    """Drop trailing blocks so the document fits on one sheet.

    Real pages end at the paper boundary, so a long document is cut like the
    first page of a multi-page file. Blocks are the blank-line separated
    chunks the composer emits; trailing headings without content are dropped
    too. Returns the trimmed markdown and the number of non-heading blocks kept.
    """
    chunks = [chunk for chunk in markdown_text.strip().split("\n\n") if chunk.strip()]
    if len(chunks) <= 2 or overflow_ratio <= 1.0:
        return markdown_text, sum(1 for chunk in chunks if not _is_heading(chunk))
    first_content = next((i for i, chunk in enumerate(chunks) if not _is_heading(chunk)), None)
    if first_content is None:
        return markdown_text, 0
    keep = max(2, min(len(chunks) - 1, int(len(chunks) * 0.97 / overflow_ratio)))
    # Never trim away every content block: a single block taller than the
    # sheet stays as a tall page rather than leaving only headings.
    keep = max(keep, first_content + 1)
    kept = chunks[:keep]
    while len(kept) > first_content + 1 and _is_heading(kept[-1]):
        kept.pop()
    return "\n\n".join(kept) + "\n", sum(1 for chunk in kept if not _is_heading(chunk))


def _is_heading(chunk: str) -> bool:
    return chunk.lstrip().startswith("#") and "\n" not in chunk.strip()
