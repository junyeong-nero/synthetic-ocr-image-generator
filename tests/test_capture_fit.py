"""Tests for src/realism/capture_fit.py: fit capture-degradation parameters
from a (clean render, captured page) pair.

Ground truth comes from the generator's own degradation pipeline
(`apply_capture_degradation`), so a recovered value is in the same units a
profile uses.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.generator import degradation
from src.generator.degradation import apply_capture_degradation
from src.generator.distribution_profile import A4_WIDTH_INCH
from src.realism.capture_align import align_capture, decompose_homography
from src.realism.capture_fit import estimate_jpeg_quality, fit_capture_files, fit_capture_pair

# 150 dpi A4 page.
PAGE_W, PAGE_H = 1240, 1754
CLEAN_DPI = PAGE_W / A4_WIDTH_INCH

_WORDS = "capture scan photo page table total amount region notice article section budget report".split()


def _document_page(seed: int = 0, width: int = PAGE_W, height: int = PAGE_H) -> Image.Image:
    """A text-heavy page: title, paragraphs and a ruled table.

    Uses Pillow's bundled default font (no font file on disk), with seeded
    random words so ORB sees non-repeating local structure.
    """
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    body = ImageFont.load_default(size=22)
    title = ImageFont.load_default(size=40)
    margin = 110
    draw.text((margin, 90), f"Report {seed:03d} - Quarterly Notice", font=title, fill=(10, 10, 10))
    y = 170
    while y < height * 0.55:
        x = margin
        while x < width - margin - 120:
            word = rng.choice(_WORDS) + (str(rng.randint(0, 99)) if rng.random() < 0.2 else "")
            draw.text((x, y), word, font=body, fill=(15, 15, 15))
            x += int(draw.textlength(word + " ", font=body))
        y += 34
    table_top, row_h, cols = int(height * 0.6), 44, [margin, 420, 700, 950, width - margin]
    for row in range(9):
        top = table_top + row * row_h
        draw.line((margin, top, width - margin, top), fill=(0, 0, 0), width=2)
        for col in range(len(cols) - 1):
            text = rng.choice(_WORDS) if col == 0 else f"{rng.randint(0, 99999):,}"
            draw.text((cols[col] + 10, top + 10), text, font=body, fill=(15, 15, 15))
    bottom = table_top + 9 * row_h
    draw.line((margin, bottom, width - margin, bottom), fill=(0, 0, 0), width=2)
    for x in cols:
        draw.line((x, table_top, x, bottom), fill=(0, 0, 0), width=2)
    return image


def _save_jpeg(image: Image.Image, path: Path, quality: int) -> Path:
    image.save(path, format="JPEG", quality=quality)
    return path


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), matrix).reshape(-1, 2)


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def test_align_recovers_rotated_and_perspective_warped_copy() -> None:
    clean = _document_page(seed=1)
    corners = np.float32([[0, 0], [PAGE_W, 0], [PAGE_W, PAGE_H], [0, PAGE_H]])
    # Rotate ~3 degrees, shrink to 80%, keystone the top edge and add a desk border.
    target = np.float32([[180, 140], [1090, 190], [1120, 1540], [120, 1500]])
    truth = cv2.getPerspectiveTransform(corners, target)
    warped = cv2.warpPerspective(
        np.asarray(clean), truth, (1250, 1700), flags=cv2.INTER_LINEAR, borderValue=(70, 60, 50)
    )
    alignment = align_capture(np.asarray(clean.convert("L")), cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY))
    error = np.abs(_project(alignment.homography, corners) - target).max()
    assert error < 1.5, f"corner reprojection error {error:.2f} px"
    assert alignment.correlation > 0.9


def test_decompose_homography_reads_rotation_scale_and_no_perspective() -> None:
    matrix, new_w, new_h = degradation._rotation_matrix((PAGE_H, PAGE_W), 2.0)
    homography = np.vstack([matrix, [0.0, 0.0, 1.0]])
    homography[:2] *= 0.5  # also halve the resolution
    result = decompose_homography(homography, (PAGE_W, PAGE_H))
    assert result["skew_deg"] == pytest.approx(2.0, abs=1e-3)
    assert result["scale"] == pytest.approx(0.5, abs=1e-4)
    assert result["perspective"] == pytest.approx(0.0, abs=1e-4)


def test_perspective_estimate_matches_generator_parameter_in_expectation() -> None:
    # The generator jitters each corner inward by U(0, amount) of the page
    # size; the estimate is moment-matched to that model, so its mean over
    # many draws recovers `amount` (a single draw can land anywhere below it).
    for amount in (0.02, 0.05, 0.09):
        estimates = []
        for seed in range(400):
            matrix, _, _, _ = degradation._perspective_matrix(
                (PAGE_H, PAGE_W), amount, random.Random(seed), pad=False
            )
            estimates.append(decompose_homography(matrix, (PAGE_W, PAGE_H))["perspective"])
        assert np.mean(estimates) == pytest.approx(amount, rel=0.06)


# --------------------------------------------------------------------------
# JPEG quality
# --------------------------------------------------------------------------


@pytest.mark.parametrize("quality", [35, 60, 85, 95])
def test_jpeg_quality_is_read_from_quantization_tables(tmp_path, quality: int) -> None:
    path = _save_jpeg(_document_page(seed=2, width=320, height=200), tmp_path / "page.jpg", quality)
    with Image.open(path) as handle:
        assert estimate_jpeg_quality(handle) == quality


def test_jpeg_quality_is_none_for_lossless_files(tmp_path) -> None:
    path = tmp_path / "page.png"
    _document_page(seed=2, width=320, height=200).save(path)
    with Image.open(path) as handle:
        assert estimate_jpeg_quality(handle) is None


# --------------------------------------------------------------------------
# full per-pair fit (generator degradations with known parameters)
# --------------------------------------------------------------------------


def _capture(clean: Image.Image, params: dict, *, scale: float = 1.0, seed: int = 0, channel: str = "") -> Image.Image:
    """Render-at-lower-dpi (area resize) then degrade, like a capture at `scale`."""
    if scale != 1.0:
        size = (int(round(clean.width * scale)), int(round(clean.height * scale)))
        clean = clean.resize(size, Image.Resampling.BOX)
    return apply_capture_degradation(clean, params, random.Random(seed), channel=channel)


def test_fit_recovers_known_scan_parameters(tmp_path) -> None:
    clean = _document_page(seed=3)
    params = {"skew_deg": 1.2, "blur_sigma": 1.0, "noise_sigma": 4.0, "ink_fade": 0.2, "grayscale": True}
    captured = _capture(clean, params, scale=2 / 3, seed=3)
    clean_path = tmp_path / "clean.png"
    clean.save(clean_path)
    captured_path = _save_jpeg(captured, tmp_path / "captured.jpg", 85)

    fit = fit_capture_files(clean_path, captured_path)
    fitted = fit["params"]
    assert fitted["dpi"] == pytest.approx(CLEAN_DPI * 2 / 3, rel=0.02)
    assert fitted["skew_deg"] == pytest.approx(1.2, abs=0.1)
    assert fitted["perspective"] < 0.005
    assert fitted["blur_sigma"] == pytest.approx(1.0, abs=0.25)
    assert fitted["noise_sigma"] == pytest.approx(4.0, abs=1.0)
    assert fitted["ink_fade"] == pytest.approx(0.2, abs=0.05)
    assert fitted["jpeg_quality"] == 85
    assert fitted["grayscale"] is True
    assert fitted["binarize"] is False
    assert fitted["paper_tint"] is False


def test_fit_recovers_photo_like_capture_with_perspective_and_tint() -> None:
    clean = _document_page(seed=4)
    params = {
        "paper_tint": True,
        "perspective": 0.05,
        "skew_deg": -2.5,
        "blur_sigma": 1.5,
        "noise_sigma": 6.0,
    }
    captured = _capture(clean, params, seed=11, channel="photographed")
    fit = fit_capture_pair(clean, captured)
    fitted = fit["params"]
    assert fitted["skew_deg"] == pytest.approx(-2.5, abs=0.6)  # perspective jitter adds some rotation
    assert 0.0 < fitted["perspective"] < 0.1
    assert fitted["blur_sigma"] == pytest.approx(1.5, abs=0.3)
    assert fitted["noise_sigma"] == pytest.approx(6.0, abs=1.2)
    assert fitted["ink_fade"] < 0.05
    assert fitted["paper_tint"] is True
    assert fitted["grayscale"] is False
    assert fitted["jpeg_quality"] is None


def test_fit_detects_binarized_capture_and_skips_unidentifiable_keys(tmp_path) -> None:
    clean = _document_page(seed=5)
    captured = _capture(clean, {"skew_deg": 0.6, "binarize": True}, scale=0.8, seed=5)
    captured_path = _save_jpeg(captured, tmp_path / "fax.jpg", 60)
    clean_path = tmp_path / "clean.png"
    clean.save(clean_path)

    fitted = fit_capture_files(clean_path, captured_path)["params"]
    assert fitted["binarize"] is True
    assert fitted["grayscale"] is True
    assert fitted["skew_deg"] == pytest.approx(0.6, abs=0.1)
    assert fitted["jpeg_quality"] == 60
    # Blur, noise and fade are thresholded away by binarization.
    assert fitted["blur_sigma"] is None
    assert fitted["noise_sigma"] is None
    assert fitted["ink_fade"] is None


def test_fit_on_identical_images_reports_a_clean_capture() -> None:
    clean = _document_page(seed=6)
    fitted = fit_capture_pair(clean, clean.copy())["params"]
    assert fitted["dpi"] == pytest.approx(CLEAN_DPI, rel=0.005)
    assert abs(fitted["skew_deg"]) < 0.05
    assert fitted["perspective"] < 0.002
    assert fitted["blur_sigma"] < 0.2
    assert fitted["noise_sigma"] < 0.5
    assert fitted["ink_fade"] < 0.02
    assert fitted["binarize"] is False


def test_fit_uses_explicit_clean_dpi() -> None:
    clean = _document_page(seed=7)
    fitted = fit_capture_pair(clean, clean.copy(), clean_dpi=300)["params"]
    assert fitted["dpi"] == pytest.approx(300, abs=1.0)


def test_fit_raises_when_capture_cannot_be_aligned() -> None:
    clean = _document_page(seed=8)
    blank = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    with pytest.raises(ValueError, match="align"):
        fit_capture_pair(clean, blank)


def test_fit_is_repeatable_with_seed() -> None:
    clean = _document_page(seed=9, width=700, height=900)
    captured = _capture(clean, {'blur_sigma': 0.7, 'noise_sigma': 3}, seed=9)
    assert fit_capture_pair(clean, captured, seed=19) == fit_capture_pair(clean, captured, seed=19)


@pytest.mark.parametrize('dpi', [0, -10, float('nan'), float('inf')])
def test_fit_rejects_invalid_dpi(dpi) -> None:
    image = Image.new('RGB', (20, 20), 'white')
    with pytest.raises(ValueError, match='dpi'):
        fit_capture_pair(image, image, clean_dpi=dpi)
