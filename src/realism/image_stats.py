"""Per-image visual statistics for document images.

All metrics are computed the same way for real reference images and for
generated samples so their distributions can be compared directly.
"""

from __future__ import annotations

import math
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from src.generator.distribution_profile import A4_WIDTH_INCH

# Images are analysed at a fixed width so resolution-dependent metrics
# (blur, noise) are comparable across sources with different DPI.
ANALYSIS_WIDTH = 1000

_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def compute_image_stats(image: Image.Image) -> Dict[str, float]:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]

    scale = ANALYSIS_WIDTH / max(1, width)
    if scale < 1.0:
        rgb_small = cv2.resize(rgb, (ANALYSIS_WIDTH, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    else:
        rgb_small = rgb
    gray = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)

    background = float(np.percentile(gray, 90))
    ink = float(np.percentile(gray, 3))
    binary_share = float(np.mean((gray < 20) | (gray > 235)))
    channel_spread = float(np.mean(rgb_small.max(axis=2).astype(np.int16) - rgb_small.min(axis=2)))
    skew = estimate_skew(gray)

    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(height / max(1, width)),
        "est_dpi_a4": float(width / A4_WIDTH_INCH),
        "skew_deg": skew,
        "abs_skew_deg": abs(skew),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise_sigma": estimate_noise_sigma(gray),
        "background_luma": background,
        "ink_luma": ink,
        "contrast": background - ink,
        "ink_ratio": _ink_ratio(gray),
        "colorfulness": colorfulness(rgb_small),
        "is_grayscale": 1.0 if channel_spread < 3.0 else 0.0,
        "is_binary": 1.0 if binary_share > 0.97 else 0.0,
        "is_colored_background": 1.0 if _background_is_colored(rgb_small) else 0.0,
    }


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """Dominant text-line angle in degrees via projection-profile variance."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary.mean() < 0.5:
        return 0.0
    height, width = binary.shape
    target = 600
    if width > target:
        binary = cv2.resize(binary, (target, max(1, int(height * target / width))), interpolation=cv2.INTER_AREA)
    center = (binary.shape[1] / 2.0, binary.shape[0] / 2.0)

    def score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, matrix, (binary.shape[1], binary.shape[0]), flags=cv2.INTER_NEAREST)
        profile = rotated.sum(axis=1, dtype=np.float64)
        return float(np.var(profile))

    coarse = np.arange(-max_angle, max_angle + 1e-6, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.05)
    best = max(fine, key=score)
    # Rotating by `best` straightens the page, so the page itself is skewed by -best.
    return round(float(-best), 2)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer (1996) fast additive-noise standard deviation estimate."""
    height, width = gray.shape
    if height < 3 or width < 3:
        return 0.0
    conv = cv2.filter2D(gray.astype(np.float64), -1, _IMMERKAER_KERNEL)
    total = np.sum(np.abs(conv[1:-1, 1:-1]))
    return float(total * math.sqrt(0.5 * math.pi) / (6.0 * (width - 2) * (height - 2)))


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(
        math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )


def _ink_ratio(gray: np.ndarray) -> float:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(binary.mean() / 255.0)


def _background_is_colored(rgb: np.ndarray) -> bool:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray >= np.percentile(gray, 60)
    if not mask.any():
        return False
    paper = rgb[mask].astype(np.float64).mean(axis=0)
    return bool(paper.max() - paper.min() > 12 or paper.mean() < 200)
