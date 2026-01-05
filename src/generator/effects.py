import random
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

logger = logging.getLogger(__name__)


def _find_perspective_coeffs(
    source_coords: List[Tuple[int, int]], target_coords: List[Tuple[int, int]]
) -> np.ndarray:
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.extend([
            [s[0], s[1], 1, 0, 0, 0, -t[0] * s[0], -t[0] * s[1]],
            [0, 0, 0, s[0], s[1], 1, -t[1] * s[0], -t[1] * s[1]],
        ])
    A = np.array(matrix, dtype=float)
    B = np.array(target_coords, dtype=float).flatten()
    return np.linalg.solve(A, B)


def render_text_with_effects(
    text: str,
    font_path: str,
    background_color: Tuple[int, int, int],
    font_size: int,
    bold: bool = False,
    tilt: int = 0,
    shadow: bool = False,
    distortion: bool = False,
    blur: bool = False,
    contrast: bool = False,
) -> Image.Image:
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        logger.warning(f"Font '{font_path}' not found. Using default font.")
        font = ImageFont.load_default()

    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    _, _, text_width, text_height = dummy_draw.textbbox((0, 0), text, font=font)

    padding = font_size // 4
    img_width = text_width + padding * 2
    img_height = text_height + padding * 2
    if tilt != 0:
        img_width, img_height = int(img_width * 1.1), int(img_height * 1.1)

    img = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    x, y = (img_width - text_width) // 2, (img_height - text_height) // 2

    text_color = (0, 0, 0) if sum(background_color) > 384 else (255, 255, 255)

    if shadow:
        shadow_offset = max(1, font_size // 20)
        draw.text(
            (x + shadow_offset, y + shadow_offset),
            text,
            font=font,
            fill=(50, 50, 50),
        )

    draw.text((x, y), text, font=font, fill=text_color)

    if bold:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            draw.text((x + dx, y + dy), text, font=font, fill=text_color)

    if distortion:
        width, height = img.size
        source_coords = [(0, 0), (width, 0), (width, height), (0, height)]
        max_distort = font_size // 10
        target_coords = [
            (
                s_x + random.randint(-max_distort, max_distort),
                s_y + random.randint(-max_distort, max_distort),
            )
            for s_x, s_y in source_coords
        ]
        coeffs = _find_perspective_coeffs(source_coords, target_coords)
        img = img.transform(
            (width, height),
            Image.PERSPECTIVE,
            coeffs,
            Image.BICUBIC,
            fillcolor=background_color,
        )

    if tilt != 0:
        img = img.rotate(tilt, expand=True, fillcolor=background_color)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

    if blur:
        blur_radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if contrast:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.7, 1.5)
        img = enhancer.enhance(factor)

    return img
