"""Tests for src/generator/capture_artifacts.py and its dispatch from
src/generator/degradation.py::apply_capture_degradation.
"""

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.generator import capture_artifacts
from src.generator.degradation import apply_capture_degradation
from src.generator.distribution_profile import CaptureSample

MARGIN = 40
ROW_PERIOD = 24
ROW_THICKNESS = 8


def _striped_page(width: int = 400, height: int = 520) -> Image.Image:
    """A page with periodic horizontal ink rows, like the fixture in
    test_distribution_profile.py, used here to test row/column-local effects.
    """
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for row in range(MARGIN, height - MARGIN, ROW_PERIOD):
        draw.rectangle((MARGIN, row, width - MARGIN, row + ROW_THICKNESS), fill=(20, 20, 20))
    return image


def _ink_rows(height: int = 520) -> set:
    rows = set()
    for row in range(MARGIN, height - MARGIN, ROW_PERIOD):
        rows.update(range(row, row + ROW_THICKNESS))
    return rows


# --------------------------------------------------------------------------
# stamp
# --------------------------------------------------------------------------


def test_apply_stamp_adds_a_localized_red_dominant_mark():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_stamp(arr.copy(), random.Random(2))

    diff_mask = np.any(np.abs(result - arr) > 1.0, axis=2)
    assert diff_mask.any(), "stamp should change some pixels"
    # localized: not a full-page effect
    assert diff_mask.mean() < 0.35

    red_dominant = (result[..., 0] > result[..., 1] + 15) & (result[..., 0] > result[..., 2] + 15)
    assert (red_dominant & diff_mask).any(), "stamp region should be red-dominant"


def test_apply_stamp_is_deterministic():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    a = capture_artifacts.apply_stamp(arr.copy(), random.Random(42))
    b = capture_artifacts.apply_stamp(arr.copy(), random.Random(42))
    assert np.array_equal(a, b)


def test_apply_capture_degradation_dispatches_stamp_before_grayscale():
    page = _striped_page()
    result = apply_capture_degradation(page, {"stamp": True}, random.Random(5), channel="scanned")
    arr = np.asarray(result, dtype=np.float32)
    red_dominant = (arr[..., 0] > arr[..., 1] + 15) & (arr[..., 0] > arr[..., 2] + 15)
    assert red_dominant.any()


def test_load_stamp_font_returns_none_when_no_candidate_font_exists(monkeypatch):
    monkeypatch.setattr(capture_artifacts, "_FONT_DIR", Path("/nonexistent/path/does-not-exist"))
    assert capture_artifacts._load_stamp_font(20) is None


def test_draw_stamp_text_skips_drawing_when_font_is_none():
    stamp = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stamp)
    before = np.asarray(stamp).copy()
    capture_artifacts._draw_stamp_text(draw, 120, None, (200, 20, 20, 255), random.Random(1))
    after = np.asarray(stamp)
    assert np.array_equal(before, after), "no glyphs (not even tofu/blank ones) should be drawn without a font"


def test_draw_stamp_text_draws_when_font_is_available():
    stamp = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stamp)
    before = np.asarray(stamp).copy()
    font = ImageFont.load_default()
    capture_artifacts._draw_stamp_text(draw, 120, font, (200, 20, 20, 255), random.Random(1))
    after = np.asarray(stamp)
    assert not np.array_equal(before, after)


def test_apply_stamp_does_not_crash_and_still_draws_the_ring_without_a_font(monkeypatch):
    monkeypatch.setattr(capture_artifacts, "_load_stamp_font", lambda size: None)
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_stamp(arr.copy(), random.Random(2))
    diff_mask = np.any(np.abs(result - arr) > 1.0, axis=2)
    assert diff_mask.any()
    red_dominant = (result[..., 0] > result[..., 1] + 15) & (result[..., 0] > result[..., 2] + 15)
    assert (red_dominant & diff_mask).any()


# --------------------------------------------------------------------------
# highlighter
# --------------------------------------------------------------------------


def test_apply_highlighter_tints_a_contiguous_band_over_a_text_line():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_highlighter(arr.copy(), random.Random(5))

    diff_rows = np.where(np.any(np.abs(result - arr) > 1.0, axis=(1, 2)))[0]
    assert diff_rows.size > 0
    assert diff_rows.max() - diff_rows.min() + 1 == diff_rows.size, "changed rows must be contiguous"
    assert _ink_rows() & set(diff_rows.tolist()), "band should cover a real text line"

    band = result[diff_rows.min(): diff_rows.max() + 1]
    orig_band = arr[diff_rows.min(): diff_rows.max() + 1]
    # yellow tint: blue channel does not increase
    assert band[..., 2].mean() <= orig_band[..., 2].mean() + 1e-3


def test_apply_highlighter_no_text_rows_is_identity():
    blank = np.full((200, 200, 3), 255.0, dtype=np.float32)
    result = capture_artifacts.apply_highlighter(blank.copy(), random.Random(1))
    assert np.array_equal(result, blank)


def test_apply_highlighter_does_not_paint_the_margins_outside_ink_extent():
    # Real highlighter strokes cover the text, not the full page width: the
    # fixture's ink rows only span [MARGIN, width-MARGIN], so a strip well
    # inside each margin must stay untouched by the highlighter.
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_highlighter(arr.copy(), random.Random(5))
    diff_rows = np.where(np.any(np.abs(result - arr) > 1.0, axis=(1, 2)))[0]
    assert diff_rows.size > 0
    band = result[diff_rows.min(): diff_rows.max() + 1]
    orig_band = arr[diff_rows.min(): diff_rows.max() + 1]
    assert np.array_equal(band[:, :20], orig_band[:, :20]), "left margin painted"
    assert np.array_equal(band[:, -20:], orig_band[:, -20:]), "right margin painted"
    # but the band does change some pixels within the ink extent
    assert not np.array_equal(band[:, MARGIN:-MARGIN], orig_band[:, MARGIN:-MARGIN])


# --------------------------------------------------------------------------
# fold_lines
# --------------------------------------------------------------------------


def test_apply_fold_lines_changes_pixels_and_is_deterministic():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    a = capture_artifacts.apply_fold_lines(arr.copy(), 1, random.Random(2))
    b = capture_artifacts.apply_fold_lines(arr.copy(), 1, random.Random(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, arr)


def test_apply_fold_lines_zero_count_is_identity():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_fold_lines(arr.copy(), 0, random.Random(2))
    assert np.array_equal(result, arr)


# --------------------------------------------------------------------------
# punch_holes
# --------------------------------------------------------------------------


def test_apply_punch_holes_stay_in_left_margin_and_never_touch_ink():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_punch_holes(arr.copy(), random.Random(7))

    diff_mask = np.any(np.abs(result - arr) > 1.0, axis=2)
    assert diff_mask.any(), "punch holes should change some pixels"
    ys, xs = np.where(diff_mask)
    assert xs.max() < MARGIN, "holes must stay left of the ink bounding box"
    assert result[ys, xs].mean() < arr[ys, xs].mean(), "holes are dark"


def test_apply_punch_holes_is_deterministic():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    a = capture_artifacts.apply_punch_holes(arr.copy(), random.Random(3))
    b = capture_artifacts.apply_punch_holes(arr.copy(), random.Random(3))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# scanner_border
# --------------------------------------------------------------------------


def test_apply_scanner_border_darkens_an_outer_band_only():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_scanner_border(arr.copy(), 0.8, random.Random(1))

    assert result.mean() <= arr.mean() + 1e-6
    diff_mask = np.any(np.abs(result - arr) > 1.0, axis=2)
    assert diff_mask.any()
    ys, xs = np.where(diff_mask)
    touches_edge = ys.min() == 0 or xs.min() == 0 or ys.max() == arr.shape[0] - 1 or xs.max() == arr.shape[1] - 1
    assert touches_edge


def test_apply_scanner_border_zero_strength_is_identity():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_scanner_border(arr.copy(), 0.0, random.Random(1))
    assert np.array_equal(result, arr)


# --------------------------------------------------------------------------
# toner_streaks
# --------------------------------------------------------------------------


def test_apply_toner_streaks_changes_only_narrow_vertical_bands():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_toner_streaks(arr.copy(), 3, random.Random(4))

    diff = np.abs(result - arr) > 0.5
    changed_cols = np.where(diff.any(axis=(0, 2)))[0]
    assert 0 < changed_cols.size < arr.shape[1] * 0.3


def test_apply_toner_streaks_zero_count_is_identity():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_toner_streaks(arr.copy(), 0, random.Random(4))
    assert np.array_equal(result, arr)


# --------------------------------------------------------------------------
# page_curl
# --------------------------------------------------------------------------


def test_apply_page_curl_warps_geometry_and_is_deterministic():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    fill = (255.0, 255.0, 255.0)
    a = capture_artifacts.apply_page_curl(arr.copy(), 0.03, random.Random(9), fill)
    b = capture_artifacts.apply_page_curl(arr.copy(), 0.03, random.Random(9), fill)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, arr)
    # the canvas grows taller (by up to amount * height) to make room for
    # content shifted down by the curl, so it never loses bottom-edge ink
    assert a.shape[1] == arr.shape[1]
    assert a.shape[0] >= arr.shape[0]
    assert a.shape[0] <= arr.shape[0] + int(np.ceil(0.03 * arr.shape[0])) + 1


def test_apply_page_curl_zero_amount_is_identity():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_page_curl(arr.copy(), 0.0, random.Random(9), (255.0, 255.0, 255.0))
    assert np.array_equal(result, arr)


def test_apply_page_curl_does_not_lose_ink_reaching_the_bottom_edge():
    # A page whose text runs all the way to the last rows (fit-to-sheet
    # trimming fills pages up to the paper edge). page_curl shifts content
    # down by up to `amount * height`; without extra canvas, the bottom rows
    # get pushed past the image bounds and are lost even though GT_markdown
    # still describes them (Global Constraint 5). Use a several-pixel-thick
    # band (like a real text line, not a 1px line) so ordinary bilinear
    # resampling of the warp doesn't itself dilute the darkest pixel.
    width, height = 300, 200
    arr = np.full((height, width, 3), 255.0, dtype=np.float32)
    arr[height - 6:, :, :] = 0.0  # ink band reaching the very bottom edge
    fill = (255.0, 255.0, 255.0)
    result = capture_artifacts.apply_page_curl(arr.copy(), 0.05, random.Random(3), fill)
    gray = result.mean(axis=2)
    darkest_per_column = gray.min(axis=0)
    assert np.all(darkest_per_column < 100.0), "some columns lost their bottom-edge ink"


# --------------------------------------------------------------------------
# edge_crop
# --------------------------------------------------------------------------


def test_apply_edge_crop_never_removes_ink():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_edge_crop(arr.copy(), 0.05, random.Random(3))

    assert result.shape[0] <= arr.shape[0] and result.shape[1] <= arr.shape[1]
    assert not np.array_equal(result.shape, (0, 0, 3))
    orig_dark = int((arr.mean(axis=2) < 200).sum())
    new_dark = int((result.mean(axis=2) < 200).sum())
    assert new_dark == orig_dark, "cropping must not remove any ink pixels"


def test_apply_edge_crop_zero_amount_is_identity():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    result = capture_artifacts.apply_edge_crop(arr.copy(), 0.0, random.Random(3))
    assert np.array_equal(result, arr)


def _black_ink_page(width: int = 400, height: int = 520) -> Image.Image:
    # Pure-black ink (rather than the shared fixture's (20, 20, 20)) so a
    # strict darkness threshold isolates real text ink from the `photographed`
    # channel's desk-background fill colours, which are all measurably
    # lighter (luminance ~43-137, see degradation._PHOTO_BACKGROUNDS) than
    # true ink but well under a loose 200 threshold -- so a loose threshold
    # would wrongly count "cropped off some background" as "lost ink".
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for row in range(MARGIN, height - MARGIN, ROW_PERIOD):
        draw.rectangle((MARGIN, row, width - MARGIN, row + ROW_THICKNESS), fill=(0, 0, 0))
    return image


def test_edge_crop_after_perspective_and_skew_can_reach_the_page_edge():
    # edge_crop must run against the FINAL (post-perspective/skew) geometry,
    # not the pre-geometry paper margin, or it can never produce the "photo
    # frame cuts off the page border" look: the desk background would always
    # stay fully visible around the page.
    page = _black_ink_page()
    geometry_params = {"perspective": 0.06, "skew_deg": 4.0}
    crop_params = {**geometry_params, "edge_crop": 0.2}
    seed = 4

    baseline = apply_capture_degradation(page, geometry_params, random.Random(seed), channel="photographed")
    cropped = apply_capture_degradation(page, crop_params, random.Random(seed), channel="photographed")

    ink_threshold = 30.0
    base_ink = int((np.asarray(baseline, dtype=np.float32).mean(axis=2) < ink_threshold).sum())
    crop_arr = np.asarray(cropped, dtype=np.float32)
    crop_ink = int((crop_arr.mean(axis=2) < ink_threshold).sum())
    assert crop_ink >= base_ink * 0.97, (
        f"cropping must not remove the page's real ink: {crop_ink} < {base_ink}"
    )

    gray = crop_arr.mean(axis=2)
    edge_means = [gray[0, :].mean(), gray[-1, :].mean(), gray[:, 0].mean(), gray[:, -1].mean()]
    assert any(mean > 200.0 for mean in edge_means), (
        "expected at least one image border to be paper-coloured (page cut off "
        f"by the crop), got edge means {edge_means}"
    )


# --------------------------------------------------------------------------
# bleed_through rebuild: mirrored + shuffled + shifted bands
# --------------------------------------------------------------------------


def test_bleed_through_bands_are_shuffled_not_a_uniform_roll():
    arr = np.asarray(_striped_page(), dtype=np.float32)
    mirrored = arr[:, ::-1, :]
    shuffled = capture_artifacts._shuffle_mirrored_bands(mirrored, random.Random(4), n_bands=8)
    assert not any(
        np.array_equal(shuffled, np.roll(mirrored, shift, axis=0))
        for shift in range(0, arr.shape[0], 4)
    )


def test_bleed_through_ghost_bleeds_into_gaps_between_lines():
    page = _striped_page()
    result = apply_capture_degradation(page, {"bleed_through": 0.9}, random.Random(11), channel="scanned")
    arr = np.asarray(result, dtype=np.float32)
    gray = arr.mean(axis=2)
    row_profile = gray.mean(axis=1)
    # A gap row (between two ink rows) that was pure white before should now
    # show some ghost ink, proving the ghost does not sit exactly on the
    # page's own (periodic) line positions.
    gap_row = MARGIN + ROW_THICKNESS + (ROW_PERIOD - ROW_THICKNESS) // 2
    assert row_profile[gap_row] < 250.0


def test_bleed_through_is_deterministic():
    page = _striped_page()
    a = apply_capture_degradation(page, {"bleed_through": 0.5}, random.Random(11), channel="scanned")
    b = apply_capture_degradation(page, {"bleed_through": 0.5}, random.Random(11), channel="scanned")
    assert np.array_equal(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------
# binarize_method
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["otsu", "adaptive", "sauvola"])
def test_binarize_image_methods_produce_bitonal_output(method):
    gray = np.asarray(_striped_page().convert("L"))
    result = capture_artifacts.binarize_image(gray, method)
    assert set(np.unique(result).tolist()) <= {0, 255}
    assert result.shape == gray.shape


def test_apply_capture_degradation_respects_binarize_method():
    page = _striped_page()
    otsu = apply_capture_degradation(page, {"binarize": True, "binarize_method": "otsu"}, random.Random(0))
    sauvola = apply_capture_degradation(page, {"binarize": True, "binarize_method": "sauvola"}, random.Random(0))
    assert set(np.unique(np.asarray(otsu.convert("L"))).tolist()) <= {0, 255}
    assert set(np.unique(np.asarray(sauvola.convert("L"))).tolist()) <= {0, 255}


def test_apply_capture_degradation_binarize_method_defaults_to_otsu():
    page = _striped_page()
    default = apply_capture_degradation(page, {"binarize": True}, random.Random(0))
    explicit = apply_capture_degradation(page, {"binarize": True, "binarize_method": "otsu"}, random.Random(0))
    assert np.array_equal(np.asarray(default), np.asarray(explicit))


# --------------------------------------------------------------------------
# CaptureSample.difficulty()
# --------------------------------------------------------------------------


def test_capture_sample_difficulty_increases_with_page_curl():
    base = CaptureSample(channel="photographed", dpi=200, params={})
    curled = CaptureSample(channel="photographed", dpi=200, params={"page_curl": 0.05})
    assert base.difficulty() == "easy"
    assert curled.difficulty() in {"medium", "hard"}


def test_capture_sample_difficulty_increases_with_edge_crop():
    base = CaptureSample(channel="photographed", dpi=200, params={})
    cropped = CaptureSample(channel="photographed", dpi=200, params={"edge_crop": 0.08})
    assert base.difficulty() == "easy"
    assert cropped.difficulty() in {"medium", "hard"}


# --------------------------------------------------------------------------
# Dispatch wiring: every new key is off by default and wired when present
# --------------------------------------------------------------------------


def test_new_capture_artifact_keys_default_off_stay_identity():
    page = _striped_page()
    params = {
        "stamp": False,
        "highlighter": False,
        "fold_lines": 0,
        "punch_holes": False,
        "scanner_border": 0.0,
        "toner_streaks": 0,
        "page_curl": 0.0,
        "edge_crop": 0.0,
        "binarize_method": "sauvola",  # irrelevant since binarize is off
    }
    result = apply_capture_degradation(page, params, random.Random(0))
    assert np.array_equal(np.asarray(result), np.asarray(page))


@pytest.mark.parametrize(
    "params",
    [
        {"stamp": True},
        {"highlighter": True},
        {"fold_lines": 2},
        {"punch_holes": True},
        {"scanner_border": 0.5},
        {"toner_streaks": 4},
        {"page_curl": 0.03},
        {"edge_crop": 0.05},
    ],
)
def test_apply_capture_degradation_dispatches_each_new_key_deterministically(params):
    page = _striped_page()
    a = apply_capture_degradation(page, params, random.Random(6), channel="photographed")
    b = apply_capture_degradation(page, params, random.Random(6), channel="photographed")
    assert np.array_equal(np.asarray(a), np.asarray(b))

    empty = apply_capture_degradation(page, {}, random.Random(6), channel="photographed")
    assert np.asarray(a).shape != np.asarray(empty).shape or not np.array_equal(
        np.asarray(a), np.asarray(empty)
    )
