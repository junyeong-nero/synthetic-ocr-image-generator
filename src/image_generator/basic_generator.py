import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

logger = logging.getLogger(__name__)


def find_coeffs(
    source_coords: List[Tuple[int, int]], target_coords: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Calculates the 8 coefficients for a perspective distortion.

    This function solves a system of linear equations (Ax = B) to find the
    coefficients that map the source coordinates to the target coordinates.
    """
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.extend(
            [
                [s[0], s[1], 1, 0, 0, 0, -t[0] * s[0], -t[0] * s[1]],
                [0, 0, 0, s[0], s[1], 1, -t[1] * s[0], -t[1] * s[1]],
            ]
        )
    A = np.array(matrix, dtype=float)
    B = np.array(target_coords, dtype=float).flatten()
    return np.linalg.solve(A, B)


def _generate_text_image(
    text: str,
    font_path: str,
    background_color: Tuple[int, int, int],
    font_size: int,
    bold: bool,
    tilt: int,
    shadow: bool,
    distortion: bool,
    blur: bool,
    contrast: bool,
) -> Image.Image:
    """
    Generates an image of a single sentence with various stylistic effects.

    Args:
        text: The text to render.
        font_path: Path to the .ttf font file.
        background_color: RGB tuple for the image background.
        font_size: The font size.
        bold: If True, applies a faux bold effect.
        tilt: The angle to rotate the image.
        shadow: If True, adds a drop shadow to the text.
        distortion: If True, applies a perspective distortion.
        blur: If True, applies a Gaussian blur.
        contrast: If True, adjusts the image contrast.

    Returns:
        A PIL Image object with the rendered text and effects.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        logger.warning(f"Font '{font_path}' not found. Using default font.")
        font = ImageFont.load_default()

    # Determine text size to create an appropriately sized image
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    _, _, text_width, text_height = dummy_draw.textbbox((0, 0), text, font=font)

    padding = font_size // 2
    img_width = text_width + padding * 2
    img_height = text_height + padding * 2
    if tilt != 0:
        # Increase canvas size to prevent cropping during rotation
        img_width, img_height = int(img_width * 1.5), int(img_height * 1.5)

    img = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    x, y = (img_width - text_width) // 2, (img_height - text_height) // 2

    # Choose text color based on background brightness
    text_color = (0, 0, 0) if sum(background_color) > 384 else (255, 255, 255)

    # Apply effects
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
        # Faux bold by drawing text with a 1px offset
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
        coeffs = find_coeffs(source_coords, target_coords)
        img = img.transform(
            (width, height),
            Image.PERSPECTIVE,
            coeffs,
            Image.BICUBIC,
            fillcolor=background_color,
        )

    if tilt != 0:
        img = img.rotate(tilt, expand=True, fillcolor=background_color)
        bbox = img.getbbox()  # Crop to the bounding box of the rotated content
        if bbox:
            img = img.crop(bbox)

    if blur:
        blur_radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if contrast:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.7, 1.5)  # Decrease or increase contrast
        img = enhancer.enhance(factor)

    return img
