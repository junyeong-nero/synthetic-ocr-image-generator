"""Real-scan / real-photo artifacts layered onto a rendered page.

Each function here takes a ``float32`` RGB ``np.ndarray`` (and a seeded
``random.Random``) and returns a modified array of the same dtype. They are
pure, self-contained transforms so a later "scenario" step can compose several
of them behind one correlated knob (see ``docs/distribution.md``). Dispatch
and ordering live in ``src/generator/degradation.py::apply_capture_degradation``
so this module stays a plain function library with no pipeline logic.

Effects:

- ``apply_stamp``: a red organisation seal (circle or rounded square, double
  ring, short Korean text), slightly rotated with uneven ink, multiply-blended
  onto the page.
- ``apply_highlighter``: a translucent yellow bar over 1-3 text lines found by
  a horizontal ink projection profile.
- ``apply_fold_lines``: a paper crease (dark line + light edge + a brightness
  step on one side).
- ``apply_punch_holes``: 2-3 dark round holes confined to the left margin,
  left of the page's ink bounding box.
- ``apply_scanner_border``: a dark shadow band along one or two page edges.
- ``apply_toner_streaks``: faint vertical streaks (roller/toner artifacts).
- ``apply_page_curl``: a smooth vertical warp (``cv2.remap``) simulating book
  curvature in a photographed page.
- ``apply_edge_crop``: crops part of the page border, never past the ink
  bounding box, so ground truth stays complete.
- ``build_mirrored_reverse_side``: the horizontally-mirrored, band-shuffled
  reverse-side image used by ``degradation.py``'s ``bleed_through``.
- ``binarize_image``: Otsu / adaptive / Sauvola thresholding for
  ``binarize_method``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts" / "ko"
_STAMP_FONT_CANDIDATES: Tuple[str, ...] = (
    "NanumGothicExtraBold.ttf",
    "NanumGothicBold.ttf",
    "NanumGothic.ttf",
    "NanumBarunGothicBold.ttf",
)
_STAMP_ORG_NAMES: Tuple[str, ...] = ("대한상사", "서울지점", "행정관리부", "한국청", "품질보증부")
_STAMP_SEALS: Tuple[str, ...] = ("직인", "(인)", "인증")


def ink_bounding_box(gray: np.ndarray, threshold: float = 200.0) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box ``(x0, y0, x1, y1)`` of pixels darker than ``threshold``.

    Returns ``None`` when the image has no such pixels (blank page).
    """
    mask = gray < threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _load_stamp_font(size: int) -> ImageFont.ImageFont:
    for name in _STAMP_FONT_CANDIDATES:
        path = _FONT_DIR / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def apply_stamp(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Draw a red organisation seal onto the page (call before grayscale)."""
    height, width = arr.shape[:2]
    size = max(28, int(min(width, height) * rng.uniform(0.10, 0.16)))

    stamp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stamp)
    color = (rng.randint(150, 205), rng.randint(15, 45), rng.randint(15, 45), 255)
    ring_w = max(1, size // 22)
    pad = ring_w * 2
    shape = rng.choice(("circle", "rounded_square"))
    if shape == "circle":
        draw.ellipse((pad, pad, size - pad, size - pad), outline=color, width=ring_w)
        draw.ellipse(
            (pad + ring_w * 3, pad + ring_w * 3, size - pad - ring_w * 3, size - pad - ring_w * 3),
            outline=color,
            width=max(1, ring_w - 1),
        )
    else:
        radius = size // 8
        draw.rounded_rectangle((pad, pad, size - pad, size - pad), radius=radius, outline=color, width=ring_w)
        draw.rounded_rectangle(
            (pad + ring_w * 3, pad + ring_w * 3, size - pad - ring_w * 3, size - pad - ring_w * 3),
            radius=max(1, radius - ring_w),
            outline=color,
            width=max(1, ring_w - 1),
        )

    font = _load_stamp_font(max(8, size // 6))
    org = rng.choice(_STAMP_ORG_NAMES)
    seal = rng.choice(_STAMP_SEALS)
    for i, line in enumerate((org, seal)):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        y = size * (0.34 if i == 0 else 0.58)
        draw.text(((size - text_w) / 2, y), line, font=font, fill=color)

    # Uneven ink: knock out random speckles from the alpha channel.
    np_rng = np.random.default_rng(rng.getrandbits(63))
    stamp_arr = np.asarray(stamp, dtype=np.float32).copy()
    speckle = np_rng.random(stamp_arr.shape[:2]) < 0.12
    stamp_arr[speckle, 3] = 0.0

    angle = rng.uniform(-18.0, 18.0)
    stamp_img = Image.fromarray(stamp_arr.astype(np.uint8), mode="RGBA").rotate(
        angle, expand=True, resample=Image.BICUBIC
    )
    stamp_arr = np.asarray(stamp_img, dtype=np.float32)
    sh, sw = stamp_arr.shape[:2]

    if rng.random() < 0.7:
        # Lower-right corner of the page.
        x0 = int(width - sw - rng.uniform(0.02, 0.08) * width)
        y0 = int(height - sh - rng.uniform(0.03, 0.10) * height)
    else:
        # Over the last text lines, found via the ink bounding box.
        gray = arr.mean(axis=2)
        box = ink_bounding_box(gray)
        if box is not None:
            _, _, ink_x1, ink_y1 = box
            x0 = int(min(width - sw, max(0, ink_x1 - sw * rng.uniform(0.4, 0.9))))
            y0 = int(min(height - sh, max(0, ink_y1 - sh * rng.uniform(0.6, 1.0))))
        else:
            x0 = int(width - sw - rng.uniform(0.02, 0.08) * width)
            y0 = int(height - sh - rng.uniform(0.03, 0.10) * height)

    x0 = max(0, min(width - sw, x0))
    y0 = max(0, min(height - sh, y0))
    x1, y1 = x0 + sw, y0 + sh

    result = arr.copy()
    alpha = (stamp_arr[..., 3:4] / 255.0) * rng.uniform(0.55, 0.85)
    ink_color = stamp_arr[..., :3]
    region = result[y0:y1, x0:x1]
    blended = region * (ink_color / 255.0)
    result[y0:y1, x0:x1] = region * (1.0 - alpha) + blended * alpha
    return result


def _group_consecutive(indices: np.ndarray) -> List[Tuple[int, int]]:
    if indices.size == 0:
        return []
    groups: List[Tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value - prev > 1:
            groups.append((start, prev + 1))
            start = value
        prev = value
    groups.append((start, prev + 1))
    return groups


def apply_highlighter(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Translucent yellow bar over 1-3 text lines (horizontal projection profile)."""
    height, width = arr.shape[:2]
    gray = arr.mean(axis=2)
    ink_per_row = (gray < 200.0).sum(axis=1)
    text_rows = np.where(ink_per_row > max(1.0, ink_per_row.max() * 0.05))[0]
    bands = _group_consecutive(text_rows)
    if not bands:
        return arr

    n_lines = rng.randint(1, min(3, len(bands)))
    start_idx = rng.randrange(0, len(bands) - n_lines + 1)
    chosen = bands[start_idx: start_idx + n_lines]
    y0, y1 = chosen[0][0], chosen[-1][1]
    pad = max(2, int((y1 - y0) * 0.2))
    y0 = max(0, y0 - pad)
    y1 = min(height, y1 + pad)

    result = arr.copy()
    color = np.array([255.0, 232.0, 60.0], dtype=np.float32)
    alpha = rng.uniform(0.35, 0.55)
    region = result[y0:y1, :]
    result[y0:y1, :] = region * (1.0 - alpha) + color * alpha
    return result


def apply_fold_lines(arr: np.ndarray, count: int, rng: random.Random) -> np.ndarray:
    """A paper crease: dark line, light edge, slight brightness step on one side."""
    count = max(0, min(2, int(count)))
    if count <= 0:
        return arr
    result = arr.copy()
    height, width = result.shape[:2]
    for _ in range(count):
        vertical = rng.random() < 0.5
        thickness = max(1, int(round(min(width, height) * 0.003)))
        step = rng.uniform(0.03, 0.08) * (1.0 if rng.random() < 0.5 else -1.0)
        dark = 1.0 - rng.uniform(0.1, 0.25)
        light = 1.0 + rng.uniform(0.08, 0.18)
        if vertical:
            x = int(rng.uniform(0.15, 0.85) * width)
            dark_lo, dark_hi = max(0, x - thickness), x
            result[:, dark_lo:dark_hi] *= dark
            light_hi = min(width, x + 2)
            result[:, x:light_hi] = np.clip(result[:, x:light_hi] * light, 0, 255)
            result[:, x:] *= (1.0 + step)
        else:
            y = int(rng.uniform(0.15, 0.85) * height)
            dark_lo, dark_hi = max(0, y - thickness), y
            result[dark_lo:dark_hi, :] *= dark
            light_hi = min(height, y + 2)
            result[y:light_hi, :] = np.clip(result[y:light_hi, :] * light, 0, 255)
            result[y:, :] *= (1.0 + step)
    return np.clip(result, 0, 255)


def apply_punch_holes(arr: np.ndarray, rng: random.Random, ink_threshold: float = 200.0) -> np.ndarray:
    """2-3 dark round holes confined to the left margin, left of any ink."""
    height, width = arr.shape[:2]
    gray = arr.mean(axis=2)
    box = ink_bounding_box(gray, ink_threshold)
    ink_left = box[0] if box is not None else width

    radius = max(3, int(min(width, height) * 0.012))
    margin_limit = min(ink_left - 2, int(width * 0.08))
    if margin_limit < radius * 2 + 4:
        return arr  # no safe margin to punch into

    result = arr.copy()
    n_holes = rng.choice((2, 3))
    centers_y = np.linspace(height * 0.15, height * 0.85, n_holes)
    for base_y in centers_y:
        cx = rng.uniform(radius + 2, margin_limit - radius - 1)
        cy = float(base_y) + rng.uniform(-6.0, 6.0)
        cv2.circle(
            result,
            (int(round(cx)), int(round(cy))),
            radius,
            (25.0, 25.0, 25.0),
            -1,
            lineType=cv2.LINE_AA,
        )
    return result


def apply_scanner_border(arr: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    """A dark shadow band along one or two page edges."""
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0:
        return arr
    height, width = arr.shape[:2]
    band = max(4, int(min(width, height) * 0.05))
    n_edges = 1 if rng.random() < 0.6 else 2
    edges = rng.sample(("left", "right", "top", "bottom"), k=n_edges)

    result = arr.copy()
    gradient = np.linspace(1.0 - strength, 1.0, band, dtype=np.float32)
    for edge in edges:
        if edge == "left":
            result[:, :band] *= gradient[np.newaxis, :, np.newaxis]
        elif edge == "right":
            result[:, width - band:] *= gradient[::-1][np.newaxis, :, np.newaxis]
        elif edge == "top":
            result[:band, :] *= gradient[:, np.newaxis, np.newaxis]
        else:
            result[height - band:, :] *= gradient[::-1][:, np.newaxis, np.newaxis]
    return np.clip(result, 0, 255)


def apply_toner_streaks(arr: np.ndarray, count: int, rng: random.Random) -> np.ndarray:
    """Faint vertical streaks (roller / toner artifacts)."""
    count = max(0, int(count))
    if count <= 0:
        return arr
    result = arr.copy()
    height, width = result.shape[:2]
    for _ in range(count):
        x = rng.randrange(0, width)
        streak_width = rng.randint(1, 3)
        strength = rng.uniform(0.03, 0.12)
        x0, x1 = max(0, x - streak_width // 2), min(width, x + streak_width // 2 + 1)
        factor = (1.0 + strength) if rng.random() < 0.5 else (1.0 - strength)
        result[:, x0:x1] = np.clip(result[:, x0:x1] * factor, 0, 255)
    return result


def apply_page_curl(
    arr: np.ndarray,
    amount: float,
    rng: random.Random,
    fill: Tuple[float, float, float],
) -> np.ndarray:
    """Smooth vertical warp simulating book curvature (photographed pages)."""
    amount = max(0.0, float(amount))
    if amount <= 0:
        return arr
    height, width = arr.shape[:2]
    max_shift = amount * height
    side = rng.choice((-1.0, 1.0))
    xs = np.arange(width, dtype=np.float32)
    t = xs / max(1, width - 1)
    if side < 0:
        t = 1.0 - t
    curve = (t ** 2) * max_shift

    map_x, map_y = np.meshgrid(xs, np.arange(height, dtype=np.float32))
    map_y = map_y - curve[np.newaxis, :]
    return cv2.remap(
        arr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(float(v) for v in fill),
    )


def apply_edge_crop(arr: np.ndarray, amount: float, rng: random.Random, ink_threshold: float = 200.0) -> np.ndarray:
    """Crop the page border, never past the ink bounding box."""
    amount = max(0.0, float(amount))
    if amount <= 0:
        return arr
    height, width = arr.shape[:2]
    gray = arr.mean(axis=2)
    box = ink_bounding_box(gray, ink_threshold)
    if box is None:
        return arr
    ink_x0, ink_y0, ink_x1, ink_y1 = box

    max_left = ink_x0
    max_top = ink_y0
    max_right = width - ink_x1
    max_bottom = height - ink_y1

    crop_left = min(max_left, int(round(rng.uniform(0, amount) * width)))
    crop_top = min(max_top, int(round(rng.uniform(0, amount) * height)))
    crop_right = min(max_right, int(round(rng.uniform(0, amount) * width)))
    crop_bottom = min(max_bottom, int(round(rng.uniform(0, amount) * height)))

    if crop_left + crop_right >= width or crop_top + crop_bottom >= height:
        return arr
    if crop_left == crop_top == crop_right == crop_bottom == 0:
        return arr
    return arr[crop_top: height - crop_bottom, crop_left: width - crop_right]


def _shuffle_mirrored_bands(mirrored: np.ndarray, rng: random.Random, n_bands: int = 9) -> np.ndarray:
    """Cut ``mirrored`` into horizontal bands, shuffle and horizontally shift each.

    Kept separate from blurring so callers/tests can check the shuffle alone.
    """
    height, width = mirrored.shape[:2]
    n_bands = max(3, min(n_bands, height))
    edges = np.linspace(0, height, n_bands + 1).astype(int)
    bands = [mirrored[edges[i]:edges[i + 1]] for i in range(n_bands)]

    order = list(range(n_bands))
    rng.shuffle(order)
    max_shift = max(1, width // 15)
    shuffled_bands = []
    for idx in order:
        band = bands[idx]
        shift = rng.randint(-max_shift, max_shift)
        shuffled_bands.append(np.roll(band, shift, axis=1))
    return np.concatenate(shuffled_bands, axis=0)


def build_mirrored_reverse_side(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Reverse-side page for ``bleed_through``: mirrored, band-shuffled, blurred.

    Unlike a plain mirror-and-roll, this cuts the mirrored page into bands
    that are reordered and independently shifted, so the resulting ghost text
    does not line up with the page's own (periodic) lines.
    """
    mirrored = arr[:, ::-1, :]
    reverse = _shuffle_mirrored_bands(mirrored, rng)
    return cv2.GaussianBlur(reverse, (0, 0), sigmaX=3.0)


def binarize_image(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Bitonal threshold: ``otsu`` (global), ``adaptive`` (local mean) or ``sauvola``."""
    method = (method or "otsu").lower()
    gray_u8 = gray.astype(np.uint8)
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15,
        )
    if method == "sauvola":
        from skimage.filters import threshold_sauvola

        local_thresh = threshold_sauvola(gray_u8, window_size=25)
        return np.where(gray_u8 > local_thresh, 255, 0).astype(np.uint8)
    _, result = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result
