"""Capture-channel degradations applied after markdown rendering.

Parameters are sampled by ``DistributionProfile.sample_capture`` and applied
here deterministically from a seeded ``random.Random``. Unknown or missing keys
are ignored, so a profile only needs to list the effects it wants. This module
is a thin pipeline: the individual physical-artifact effects (stamp,
highlighter, fold lines, punch holes, scanner border, toner streaks, page
curl, edge crop, the bleed-through reverse side, binarize methods) live in
``src/generator/capture_artifacts.py`` as self-contained functions so a later
"scenario" step can group several of them behind one correlated knob.

Supported parameter keys (all optional):

- ``paper_tint`` (bool): warm/grey paper background tint
- ``ink_fade`` (0..1): pull dark ink towards the paper colour
- ``bleed_through`` (0..1): mirrored, band-shuffled copy of the page (reverse
  side), so ghosts do not line up with the page's own lines
- ``highlighter`` (bool): translucent yellow bar over 1-3 text lines found by
  a horizontal ink projection profile
- ``stamp`` (bool): red organisation seal, multiply-blended onto the page
- ``fold_lines`` (int 0-2): paper crease (dark line + light edge)
- ``punch_holes`` (bool): 2-3 dark holes confined to the left margin
- ``edge_crop`` (0..~0.1): crop part of the border, never past the ink
  bounding box (GT stays complete)
- ``page_curl`` (0..~0.05): smooth vertical warp (book curvature), for
  photographed pages
- ``perspective`` (0..~0.1): corner jitter as a fraction of page size
- ``skew_deg`` (float): in-plane rotation
- ``scanner_border`` (0..1): dark shadow band along one or two page edges
- ``illumination`` (0..1): linear lighting gradient strength
- ``shadow_strength`` (0..1): soft cast shadow over one side of the page
- ``blur_sigma`` (px): gaussian blur at output resolution
- ``motion_blur_px`` (int): horizontal-ish motion blur kernel length
- ``toner_streaks`` (int): faint vertical streaks
- ``noise_sigma`` (0..255 scale): additive gaussian sensor noise
- ``speckle_density`` (fraction of pixels): salt-and-pepper dust
- ``grayscale`` (bool): convert to single channel (kept as RGB)
- ``binarize`` (bool): fax/bitonal style thresholding
- ``binarize_method`` (``otsu`` default | ``adaptive`` | ``sauvola``): used
  when ``binarize`` is true
- ``jpeg_quality`` (int | None): JPEG re-encode quality

Application order (see ``apply_capture_degradation``): paper colour effects
(tint, ink fade, bleed-through) are applied first; then, on the still-clean,
axis-aligned page, decorative marks that are not GT content (highlighter,
stamp, fold lines, punch holes) — ``edge_crop``'s real-text ink bounding box
is also snapshotted here, as a mask, *before* those marks are drawn, so a
stamp or punch hole is never mistaken for protected ink; then page-shape/
geometry effects (page curl, perspective, skew), which carry that ink mask
through the same warps; then ``edge_crop`` itself crops the final,
post-geometry image against the warped mask (never past it); then scan/photo
lighting (scanner border, illumination, shadow); then optics/sensor effects
(blur, motion blur, toner streaks, noise, speckle); then the final
grayscale/binarize + JPEG pass.
"""

from __future__ import annotations

import io
import math
import random
from typing import Any, Mapping, Tuple

import cv2
import numpy as np
from PIL import Image

from src.generator import capture_artifacts

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

    fill = rng.choice(_PHOTO_BACKGROUNDS) if photographed else tuple(int(v) for v in paper)

    ink_fade = float(params.get("ink_fade") or 0.0)
    if ink_fade > 0:
        arr = arr + (paper - arr) * min(ink_fade, 0.9)

    bleed = float(params.get("bleed_through") or 0.0)
    if bleed > 0:
        arr = _apply_bleed_through(arr, paper, bleed, rng)

    # Snapshot the real text ink bounding box now, before any decorative
    # marks (highlighter/stamp/fold_lines/punch_holes, none of which are GT
    # content) or geometry are applied, as a 0/255 mask. edge_crop carries
    # this mask through page_curl/perspective/skew below so it can crop the
    # *final* geometry (see apply_edge_crop's docstring) without ever
    # cropping into real text ink.
    edge_crop = float(params.get("edge_crop") or 0.0)
    ink_mask: np.ndarray | None = None
    if edge_crop > 0:
        ink_box = capture_artifacts.ink_bounding_box(arr.mean(axis=2))
        if ink_box is not None:
            x0, y0, x1, y1 = ink_box
            ink_mask = np.zeros(arr.shape[:2], dtype=np.float32)
            ink_mask[y0:y1, x0:x1] = 255.0

    # Marks that need the page in its clean, axis-aligned state: text-line
    # detection (highlighter) and the left margin (punch holes) would be
    # thrown off by a later skew, perspective warp or photographed-background
    # padding.
    if params.get("highlighter"):
        arr = capture_artifacts.apply_highlighter(arr, rng)

    if params.get("stamp"):
        arr = capture_artifacts.apply_stamp(arr, rng)

    fold_lines = int(params.get("fold_lines") or 0)
    if fold_lines > 0:
        arr = capture_artifacts.apply_fold_lines(arr, fold_lines, rng)

    if params.get("punch_holes"):
        arr = capture_artifacts.apply_punch_holes(arr, rng)

    page_curl = float(params.get("page_curl") or 0.0)
    if page_curl > 0:
        height, width = arr.shape[:2]
        map_x, map_y, _ = capture_artifacts.page_curl_maps(height, width, page_curl, rng)
        arr = cv2.remap(
            arr, map_x, map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(float(v) for v in fill),
        )
        if ink_mask is not None:
            ink_mask = cv2.remap(
                ink_mask, map_x, map_y,
                interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

    perspective = float(params.get("perspective") or 0.0)
    if perspective > 0:
        matrix, margin, w2, h2 = _perspective_matrix(arr.shape, perspective, rng, pad=photographed)
        if margin:
            arr = cv2.copyMakeBorder(
                arr, margin, margin, margin, margin,
                cv2.BORDER_CONSTANT, value=tuple(float(v) for v in fill),
            )
            if ink_mask is not None:
                ink_mask = cv2.copyMakeBorder(
                    ink_mask, margin, margin, margin, margin,
                    cv2.BORDER_CONSTANT, value=0.0,
                )
        arr = _warp_perspective(arr, matrix, (w2, h2), tuple(float(v) for v in fill))
        if ink_mask is not None:
            ink_mask = _warp_perspective(
                ink_mask, matrix, (w2, h2), 0.0, interpolation=cv2.INTER_NEAREST
            )

    skew = float(params.get("skew_deg") or 0.0)
    if abs(skew) > 1e-3:
        matrix, new_w, new_h = _rotation_matrix(arr.shape, skew)
        arr = _warp_affine(arr, matrix, (new_w, new_h), tuple(float(v) for v in fill))
        if ink_mask is not None:
            ink_mask = _warp_affine(
                ink_mask, matrix, (new_w, new_h), 0.0, interpolation=cv2.INTER_NEAREST
            )

    if edge_crop > 0:
        ink_box = capture_artifacts.mask_bounding_box(ink_mask) if ink_mask is not None else None
        arr = capture_artifacts.apply_edge_crop(arr, edge_crop, rng, ink_box=ink_box)

    scanner_border = float(params.get("scanner_border") or 0.0)
    if scanner_border > 0:
        arr = capture_artifacts.apply_scanner_border(arr, scanner_border, rng)

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

    toner_streaks = int(params.get("toner_streaks") or 0)
    if toner_streaks > 0:
        arr = capture_artifacts.apply_toner_streaks(arr, toner_streaks, rng)

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
            gray = capture_artifacts.binarize_image(gray, params.get("binarize_method", "otsu"))
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
    reverse = capture_artifacts.build_mirrored_reverse_side(arr, rng)
    ink = np.clip((paper - reverse) / 255.0, 0.0, 1.0)
    return arr - ink * 255.0 * strength


def _rotation_matrix(shape: Tuple[int, ...], angle: float) -> Tuple[np.ndarray, int, int]:
    """Build the ``warpAffine`` matrix + expanded output size for an in-plane rotation.

    Split from applying the matrix so a caller (``apply_capture_degradation``)
    can warp an ink mask through the *same* matrix as the colour image (kept
    symmetric with ``_perspective_matrix``, whose split matters more since
    that one also draws random jitter).
    """
    height, width = shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(math.ceil(height * sin + width * cos))
    new_h = int(math.ceil(height * cos + width * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return matrix, new_w, new_h


def _warp_affine(
    arr: np.ndarray,
    matrix: np.ndarray,
    size: Tuple[int, int],
    border_value: Any,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    return cv2.warpAffine(
        arr, matrix, size,
        flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def _perspective_matrix(
    shape: Tuple[int, ...],
    amount: float,
    rng: random.Random,
    *,
    pad: bool,
) -> Tuple[np.ndarray, int, int, int]:
    """Build the ``warpPerspective`` matrix + padded output size for corner jitter.

    Split from applying the matrix (and from ``copyMakeBorder``, which the
    caller still does separately for the colour image vs. the 0-filled ink
    mask) so the *same* random jitter and matrix can be reused to warp an ink
    mask alongside the colour image — computing it twice would consume the
    rng differently each time and warp the mask with different jitter than
    the image.
    """
    height, width = shape[:2]
    margin = int(round(max(width, height) * amount)) if pad else 0
    w2, h2 = width + 2 * margin, height + 2 * margin
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
    return matrix, margin, w2, h2


def _warp_perspective(
    arr: np.ndarray,
    matrix: np.ndarray,
    size: Tuple[int, int],
    border_value: Any,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    return cv2.warpPerspective(
        arr, matrix, size,
        flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
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
