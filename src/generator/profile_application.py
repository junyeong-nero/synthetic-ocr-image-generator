"""Apply a DistributionProfile to a single sample's style and rendered image."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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

    # Real pages are overwhelmingly printed on white paper; the base style
    # sampler's pastel backgrounds are kept only at the profile's rate
    # (OmniDocBench tracks this as the `colorful_background` page attribute).
    colored = typography.get("colored_background")
    if colored is not None and not colored:
        style.background_color = (255, 255, 255)
        style.code_bg_color = (246, 246, 246)

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
    return apply_capture_degradation(
        image,
        plan.capture.params,
        plan.rng,
        channel=plan.capture.channel,
    )
