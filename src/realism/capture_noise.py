"""Blank-paper residual noise calibration including clipping and JPEG loss."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


def estimate_noise(gray: np.ndarray, paper_mask: np.ndarray, quality: int | None, *, seed: int) -> float:
    """Match high-pass residual spread to a seeded Gaussian sensor model.

    Flat paper excludes registration/text-edge errors. Forward simulation
    accounts for white clipping and JPEG suppression; this estimates pre-JPEG
    noise only under that model, not arbitrary scanner denoising.
    """
    highpass = gray - cv2.GaussianBlur(gray, (0, 0), 2.)
    observed = float(np.std(highpass[paper_mask]))
    if observed < .05:
        return 0.
    level = float(np.median(gray[paper_mask]))
    if level > 250:
        level = 255.
    noise = np.random.default_rng(seed).normal(size=(256, 256))
    best = (float('inf'), 0.)
    for sigma in np.arange(0, 25.01, .25):
        sample = np.clip(level + noise * sigma, 0, 255).astype(np.uint8)
        if quality is not None:
            buffer = io.BytesIO()
            Image.fromarray(sample).save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as image:
                sample = np.array(image)
        sample = sample.astype(np.float32)
        residual = sample - cv2.GaussianBlur(sample, (0, 0), 2.)
        error = abs(float(np.std(residual[8:-8, 8:-8])) - observed)
        if error < best[0]:
            best = (error, float(sigma))
    return best[1]
