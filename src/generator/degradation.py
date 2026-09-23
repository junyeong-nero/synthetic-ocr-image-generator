"""Capture-channel degradations applied after markdown rendering.

Parameters are sampled by ``DistributionProfile.sample_capture`` and applied
here deterministically from a seeded ``random.Random``. Unknown or missing keys
are ignored, so a profile only needs to list the effects it wants.

Supported parameter keys (all optional):

- ``paper_tint`` (bool): warm/grey paper background tint
- ``ink_fade`` (0..1): pull dark ink towards the paper colour
- ``bleed_through`` (0..1): faint mirrored copy of the page (reverse side)
- ``skew_deg`` (float): in-plane rotation
- ``perspective`` (0..~0.1): corner jitter as a fraction of page size
- ``illumination`` (0..1): linear lighting gradient strength
- ``shadow_strength`` (0..1): soft cast shadow over one side of the page
- ``blur_sigma`` (px): gaussian blur at output resolution
- ``motion_blur_px`` (int): horizontal-ish motion blur kernel length
- ``noise_sigma`` (0..255 scale): additive gaussian sensor noise
- ``speckle_density`` (fraction of pixels): salt-and-pepper dust
- ``grayscale`` (bool): convert to single channel (kept as RGB)
- ``binarize`` (bool): fax/bitonal style thresholding
- ``jpeg_quality`` (int | None): JPEG re-encode quality
"""

from __future__ import annotations

import io
import math
import random
from typing import Any, Mapping, Tuple

import cv2
import numpy as np
from PIL import Image

_PHOTO_BACKGROUNDS: Tuple[Tuple[int, int, int], ...] = (
    (58, 52, 46),
    (92, 78, 60),
    (40, 42, 48),
    (120, 118, 112),
    (164, 140, 108),
)


def apply_capture_degradation(
    image: Image.Image,
    params: Mapping[str, Any],
    rng: random.Random,
    *,
    channel: str = "",
) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    np_rng = np.random.default_rng(rng.getrandbits(63))
    photographed = channel == "photographed"
    paper = _paper_color(arr)

    if params.get("paper_tint"):
        arr = _apply_paper_tint(arr, rng)
        paper = _paper_color(arr)

    ink_fade = float(params.get("ink_fade") or 0.0)
    if ink_fade > 0:
        arr = arr + (paper - arr) * min(ink_fade, 0.9)

    bleed = float(params.get("bleed_through") or 0.0)
    if bleed > 0:
        arr = _apply_bleed_through(arr, paper, bleed, rng)

    fill = rng.choice(_PHOTO_BACKGROUNDS) if photographed else tuple(int(v) for v in paper)

    perspective = float(params.get("perspective") or 0.0)
    if perspective > 0:
        arr = _apply_perspective(arr, perspective, fill, rng, pad=photographed)

    skew = float(params.get("skew_deg") or 0.0)
    if abs(skew) > 1e-3:
        arr = _apply_rotation(arr, skew, fill)

    illumination = float(params.get("illumination") or 0.0)
    if illumination > 0:
        arr = _apply_illumination(arr, illumination, rng)

    shadow = float(params.get("shadow_strength") or 0.0)
    if shadow > 0:
        arr = _apply_shadow(arr, shadow, rng)

    blur_sigma = float(params.get("blur_sigma") or 0.0)
    if blur_sigma > 0.05:
        arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=blur_sigma)

    motion = int(params.get("motion_blur_px") or 0)
    if motion >= 2:
        arr = _apply_motion_blur(arr, motion, rng)

    noise_sigma = float(params.get("noise_sigma") or 0.0)
    if noise_sigma > 0:
        arr = arr + np_rng.normal(0.0, noise_sigma, size=arr.shape[:2])[..., None]

    speckle = float(params.get("speckle_density") or 0.0)
    if speckle > 0:
        arr = _apply_speckle(arr, speckle, np_rng)

    arr = np.clip(arr, 0, 255)

    if params.get("grayscale") or params.get("binarize"):
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if params.get("binarize"):
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)

    result = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    quality = params.get("jpeg_quality")
    if quality:
        result = _jpeg_roundtrip(result, int(quality))
    return result


def _paper_color(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, 3)
    luminance = flat.mean(axis=1)
    threshold = np.percentile(luminance, 75)
    bright = flat[luminance >= threshold]
    if bright.size == 0:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return bright.mean(axis=0).astype(np.float32)


def _apply_paper_tint(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    # Multiply so that ink stays dark while white paper takes the tint.
    base = rng.uniform(0.88, 0.99)
    tint = np.array(
        [base + rng.uniform(0.0, 0.03), base + rng.uniform(-0.01, 0.02), base - rng.uniform(0.0, 0.06)],
        dtype=np.float32,
    )
    return arr * np.clip(tint, 0.75, 1.0)


def _apply_bleed_through(arr: np.ndarray, paper: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    mirrored = arr[:, ::-1, :]
    shift_y = rng.randint(-arr.shape[0] // 20, arr.shape[0] // 20)
    mirrored = np.roll(mirrored, shift_y, axis=0)
    mirrored = cv2.GaussianBlur(mirrored, (0, 0), sigmaX=1.5)
    ink = np.clip((paper - mirrored) / 255.0, 0.0, 1.0)
    return arr - ink * 255.0 * strength


def _apply_rotation(arr: np.ndarray, angle: float, fill: Tuple[int, ...]) -> np.ndarray:
    height, width = arr.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(math.ceil(height * sin + width * cos))
    new_h = int(math.ceil(height * cos + width * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(
        arr,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(float(v) for v in fill),
    )


def _apply_perspective(
    arr: np.ndarray,
    amount: float,
    fill: Tuple[int, ...],
    rng: random.Random,
    *,
    pad: bool,
) -> np.ndarray:
    height, width = arr.shape[:2]
    margin = int(round(max(width, height) * amount)) if pad else 0
    if margin:
        arr = cv2.copyMakeBorder(
            arr, margin, margin, margin, margin,
            cv2.BORDER_CONSTANT, value=tuple(float(v) for v in fill),
        )
    h2, w2 = arr.shape[:2]
    src = np.float32([
        [margin, margin],
        [w2 - margin, margin],
        [w2 - margin, h2 - margin],
        [margin, h2 - margin],
    ])
    jitter_x, jitter_y = width * amount, height * amount
    dst = src + np.float32([
        [rng.uniform(0, jitter_x), rng.uniform(0, jitter_y)],
        [-rng.uniform(0, jitter_x), rng.uniform(0, jitter_y)],
        [-rng.uniform(0, jitter_x), -rng.uniform(0, jitter_y)],
        [rng.uniform(0, jitter_x), -rng.uniform(0, jitter_y)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        arr,
        matrix,
        (w2, h2),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(float(v) for v in fill),
    )


def _apply_illumination(arr: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    height, width = arr.shape[:2]
    theta = rng.uniform(0, 2 * math.pi)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    proj = (xs / max(width, 1)) * math.cos(theta) + (ys / max(height, 1)) * math.sin(theta)
    proj = (proj - proj.min()) / max(float(proj.max() - proj.min()), 1e-6)
    gain = 1.0 - strength * proj
    return arr * gain[..., None]


def _apply_shadow(arr: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    height, width = arr.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    side = rng.choice(["left", "right", "top", "bottom"])
    extent = rng.uniform(0.15, 0.45)
    if side == "left":
        mask[:, : int(width * extent)] = 1.0
    elif side == "right":
        mask[:, int(width * (1 - extent)):] = 1.0
    elif side == "top":
        mask[: int(height * extent), :] = 1.0
    else:
        mask[int(height * (1 - extent)):, :] = 1.0
    soften = max(3.0, min(width, height) * 0.08)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=soften)
    return arr * (1.0 - strength * mask)[..., None]


def _apply_motion_blur(arr: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    angle = rng.uniform(-30, 30)
    rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(arr, -1, kernel)


def _apply_speckle(arr: np.ndarray, density: float, np_rng: np.random.Generator) -> np.ndarray:
    height, width = arr.shape[:2]
    count = int(height * width * min(density, 0.05))
    if count <= 0:
        return arr
    ys = np_rng.integers(0, height, size=count)
    xs = np_rng.integers(0, width, size=count)
    values = np_rng.choice([20.0, 235.0], size=count, p=[0.7, 0.3])
    arr = arr.copy()
    arr[ys, xs] = values[:, None]
    return arr


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=max(5, min(100, quality)))
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    decoded.load()
    return decoded
