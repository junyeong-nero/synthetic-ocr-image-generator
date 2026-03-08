import random

from generator.markdown_render_utils import MarkdownStyle


def base_styles() -> list[MarkdownStyle]:
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


def clamp_color(value: int) -> int:
    return max(0, min(255, value))


def jitter_color(color: tuple[int, int, int], span: int) -> tuple[int, int, int]:
    return (
        clamp_color(color[0] + random.randint(-span, span)),
        clamp_color(color[1] + random.randint(-span, span)),
        clamp_color(color[2] + random.randint(-span, span)),
    )


def random_style(style_profile: str) -> MarkdownStyle:
    selected = random.choice(base_styles())

    if style_profile == "legacy":
        selected.margin_top += random.randint(-8, 14)
        selected.margin_bottom += random.randint(-8, 14)
        selected.content_width += random.randint(-24, 24)
        selected.line_spacing = max(1.3, min(1.7, selected.line_spacing + random.uniform(-0.08, 0.1)))
        return selected

    if style_profile == "balanced":
        selected.margin_top += random.randint(-16, 24)
        selected.margin_bottom += random.randint(-16, 24)
        selected.margin_left += random.randint(-4, 18)
        selected.margin_right += random.randint(-4, 18)
        selected.content_width += random.randint(-36, 48)
        selected.line_spacing = max(1.2, min(1.9, selected.line_spacing + random.uniform(-0.2, 0.25)))
        selected.background_color = jitter_color(selected.background_color, 12)
        selected.h1_color = jitter_color(selected.h1_color, 16)
        selected.h2_color = jitter_color(selected.h2_color, 16)
        selected.text_color = jitter_color(selected.text_color, 12)
        selected.link_color = jitter_color(selected.link_color, 24)
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
            clamp_color(bg_base + random.randint(-12, 10)),
            clamp_color(bg_base + random.randint(-12, 10)),
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
        selected.h2_color = jitter_color(selected.h1_color, 20)
        selected.link_color = (
            random.randint(0, 40),
            random.randint(80, 150),
            random.randint(140, 220),
        )

    selected.margin_top = max(16, selected.margin_top)
    selected.margin_bottom = max(16, selected.margin_bottom)
    selected.margin_left = max(28, selected.margin_left)
    selected.margin_right = max(28, selected.margin_right)
    selected.content_width = max(500, min(720, selected.content_width))
    return selected
