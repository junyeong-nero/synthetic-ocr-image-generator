"""Deterministic planar registration; homographies map clean pixels to capture pixels."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Alignment:
    homography: np.ndarray
    correlation: float
    inliers: int
    ecc_refined: bool


def align_capture(clean: np.ndarray, captured: np.ndarray, *, seed: int = 0) -> Alignment:
    """ORB/RANSAC initialization followed by ECC, rejecting unrelated/blank pages."""
    cv2.setRNGSeed(seed)
    orb = cv2.ORB_create(nfeatures=7000, fastThreshold=10)
    k1, d1 = orb.detectAndCompute(clean, None)
    k2, d2 = orb.detectAndCompute(captured, None)
    if d1 is None or d2 is None:
        raise ValueError('Cannot align: insufficient page features')
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < .7 * n.distance]
    if len(good) < 12:
        raise ValueError('Cannot align: insufficient matching features')
    a = np.float32([k1[m.queryIdx].pt for m in good])
    b = np.float32([k2[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    if matrix is None or mask.sum() < 10 or mask.mean() < .2:
        raise ValueError('Cannot align: unreliable homography')
    refined = False
    # Refine in capture coordinates so both images have the same pixel scale.
    # This avoids ECC bias when a sharp high-resolution template is compared
    # directly with a blurred, downsampled scan.
    template = warp_clean(clean, matrix, captured.shape[::-1])
    try:
        _, correction = cv2.findTransformECC(
            template.astype(np.float32) / 255,
            captured.astype(np.float32) / 255,
            np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), None, 5,
        )
        candidate = correction @ matrix
        if np.isfinite(candidate).all():
            matrix = candidate
            refined = True
    except cv2.error:
        pass  # RANSAC remains usable; report that refinement did not converge.
    aligned = warp_clean(clean, matrix, captured.shape[::-1])
    valid = cv2.warpPerspective(np.ones_like(clean), matrix, captured.shape[::-1]) > 0
    if valid.sum() < 100:
        raise ValueError('Cannot align: insufficient overlap')
    correlation = float(np.corrcoef(aligned[valid], captured[valid])[0, 1])
    if not np.isfinite(correlation) or correlation < .65:
        raise ValueError('Cannot align: low image correlation')
    return Alignment(matrix / matrix[2, 2], correlation, int(mask.sum()), refined)


def warp_clean(clean: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Forward warp with area prefiltering when the capture is smaller."""
    height, width = clean.shape[:2]
    scale = decompose_homography(matrix, (width, height))['scale']
    if scale < .95:
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        clean = cv2.resize(clean, new_size, interpolation=cv2.INTER_AREA)
        sx, sy = new_size[0] / width, new_size[1] / height
        resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
        matrix = matrix @ np.linalg.inv(resize)
    return cv2.warpPerspective(clean, matrix, size, borderValue=(255, 255, 255))


def decompose_homography(matrix: np.ndarray, size: tuple[int, int]) -> dict[str, float]:
    """Similarity fit gives scale/rotation; corner non-parallelism estimates jitter.

    Perspective is moment-matched to independent inward U(0, amount) corner
    jitter: E|U1-U2-U3+U4| = 7*amount/15. It is not uniquely identifiable
    from one page. Scale also includes any perspective foreshortening.
    """
    width, height = size
    corners = np.float64([[0, 0], [width, 0], [width, height], [0, height]])
    projected = cv2.perspectiveTransform(corners[:, None], matrix.astype(np.float64))[:, 0]
    x, y = corners - corners.mean(axis=0), projected - projected.mean(axis=0)
    u, singular, vt = np.linalg.svd(x.T @ y)
    rotation = u @ vt
    scale = float(singular.sum() / (x * x).sum())
    unrotated = y @ rotation.T / scale
    delta = unrotated[0] - unrotated[1] + unrotated[2] - unrotated[3]
    perspective = float(np.mean(np.abs(delta) / [width, height]) * 15 / 7)
    # Similarity scale includes the expected (1-amount) inward contraction.
    perspective /= 1 + perspective
    return {'skew_deg': float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
            'scale': scale, 'perspective': perspective}
