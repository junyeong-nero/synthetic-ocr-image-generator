"""Estimate profile capture parameters from planar print/capture pairs.

Optical estimates are in captured pixels. A forward-warped clean image avoids
resampling sensor noise. Estimates assume Gaussian blur and additive noise;
lighting, printer halftones and nonplanar pages can confound them.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.realism.capture_align import align_capture, decompose_homography, warp_clean
from src.realism.capture_noise import estimate_noise


def estimate_jpeg_quality(image: Image.Image) -> int | None:
    """Nearest standard Pillow/IJG quality by quantization tables, not pixels."""
    tables = getattr(image, 'quantization', None)
    if not tables:
        return None
    best = (float('inf'), None)
    for quality in range(1, 101):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8)).save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as reference:
            distance = sum(np.mean((np.array(table) - reference.quantization.get(key, reference.quantization[0])) ** 2)
                           for key, table in tables.items())
        if distance < best[0]:
            best = (distance, quality)
    return best[1]


def _optical_fit(clean: np.ndarray, captured: np.ndarray, valid: np.ndarray,
                 quality: int | None, seed: int) -> dict:
    gray = cv2.cvtColor(captured, cv2.COLOR_RGB2GRAY).astype(np.float32)
    reference = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper_mask = valid & (cv2.erode(reference, np.ones((15, 15), np.uint8)) > 250)
    if paper_mask.sum() < 100:
        raise ValueError('Cannot fit: insufficient blank paper')
    paper = np.median(captured[paper_mask], axis=0)
    # JPEG bitonal captures have ringing near edges; use both dark and bright modes.
    dark = gray[valid & (gray < 128)]
    bitonal = bool(len(dark) and np.mean(dark < 25) > .9 and np.mean(gray[paper_mask] > 240) > .98)
    grayscale = bool(np.percentile(np.ptp(captured[valid].astype(float), axis=1), 95) < 3)
    result = {'paper_rgb': paper.tolist(), 'grayscale': grayscale, 'binarize': bitonal,
              'paper_tint': bool(np.mean(paper) < 247 or np.ptp(paper) > 5)}
    if bitonal:
        result.update(blur_sigma=None, noise_sigma=None, ink_fade=None, ink_level=None)
        return result
    # Fit gain/offset jointly with blur on content, including nearby white pixels.
    region = valid & (cv2.erode(reference, np.ones((9, 9), np.uint8)) < 240)
    if region.sum() < 100:
        raise ValueError('Cannot fit: insufficient ink')
    target = gray[region]
    best = (float('inf'), 0., 1., 0., reference)
    for sigma in np.arange(0, 3.01, .1):
        blurred = cv2.GaussianBlur(reference, (0, 0), float(sigma)) if sigma > 0 else reference
        if quality is not None:
            # JPEG softens edges on the captured side; forward-simulate it on
            # the candidate so the minimum sits at the optical blur, not at
            # optical + compression blur. RGB roundtrip matches files saved
            # from RGB images with Pillow defaults.
            rgb = np.stack([np.clip(blurred, 0, 255).astype(np.uint8)] * 3, axis=-1)
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format='JPEG', quality=int(quality))
            buffer.seek(0)
            with Image.open(buffer) as image:
                packet = np.asarray(image).astype(np.float32)
            blurred = packet.mean(axis=-1) if packet.ndim == 3 else packet.astype(np.float32)
        x = blurred[region]
        slope = float(np.mean((x - x.mean()) * (target - target.mean())) / max(float(x.var()), 1e-6))
        offset = float(target.mean() - slope * x.mean())
        error = float(np.mean((target - (slope * x + offset)) ** 2))
        if error < best[0]:
            best = (error, float(sigma), slope, offset, blurred)
    _, sigma, slope, offset, blurred = best
    noise = estimate_noise(gray, paper_mask, quality, seed=seed)
    result.update(blur_sigma=sigma, noise_sigma=noise,
                  ink_fade=float(np.clip(offset / max(float(paper.mean()), 1), 0, .9)),
                  ink_level=float(offset))
    return result


def fit_capture_pair(clean: Image.Image, captured: Image.Image, *, clean_dpi: float | None = None,
                     seed: int = 0) -> dict:
    if clean_dpi is not None and (not np.isfinite(clean_dpi) or clean_dpi <= 0):
        raise ValueError('clean_dpi must be positive and finite')
    quality = estimate_jpeg_quality(captured)
    a, b = np.asarray(clean.convert('RGB')), np.asarray(captured.convert('RGB'))
    alignment = align_capture(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), seed=seed)
    geometry = decompose_homography(alignment.homography, clean.size)
    warped = warp_clean(a, alignment.homography, captured.size)
    valid = cv2.warpPerspective(np.ones(a.shape[:2], np.uint8), alignment.homography, captured.size)
    valid = cv2.erode(valid, np.ones((21, 21), np.uint8)) > 0
    optics = _optical_fit(warped, b, valid, quality, seed)
    params = {key: value for key, value in optics.items() if key not in ('paper_rgb', 'ink_level')}
    params.update(dpi=float((clean_dpi or clean.width / (210 / 25.4)) * geometry['scale']),
                  skew_deg=geometry['skew_deg'], perspective=geometry['perspective'], jpeg_quality=quality)
    return {'params': params, 'alignment': {'homography': alignment.homography.tolist(),
            'correlation': alignment.correlation, 'inliers': alignment.inliers,
            'ecc_refined': alignment.ecc_refined, 'scale': geometry['scale']},
            'levels': {'paper_rgb': optics['paper_rgb'], 'ink_level': optics['ink_level']},
            'clean_dpi': float(clean_dpi or clean.width / (210 / 25.4)),
            'dpi_assumption': 'explicit' if clean_dpi is not None else 'A4 width (210 mm)'}


def fit_capture_files(clean: str | Path, captured: str | Path, **kwargs) -> dict:
    with Image.open(clean) as a, Image.open(captured) as b:
        result = fit_capture_pair(a, b, **kwargs)
    return {'clean': str(clean), 'captured': str(captured), **result}
