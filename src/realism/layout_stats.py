"""Image-only layout statistics for document images.

These complement `image_stats.py`'s pixel-texture metrics with *structure*:
how much text there is, how tall a line is, how wide the margins are, how
many columns the page has, and whether it has long horizontal rules (table
borders, `<hr>`s). Computed the same way for real reference images and
generated samples, at the same fixed analysis width, so distributions are
comparable across sources with different native resolution.

Every metric here is derived from a simple ink/no-ink binarisation of the
page (Otsu threshold) and row/column projection profiles. That works well
for born-digital and scanned pages. **On photographed pages, perspective
distortion, uneven illumination and desk background make these metrics
unreliable** (line bands blur together, margins are contaminated by the
photographed border) — the same caveat `image_stats.estimate_skew` documents
for the skew estimator. **A table with solid vertical borders (a ruled
column separator running the table's full height) can also put ink in every
row of the table**, merging the whole table into one `text_line_count` band
and widening its measured height — `text_line_count` / `text_line_height_*`
are therefore an undercount / overcount on table-heavy pages.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from src.generator.distribution_profile import A4_WIDTH_INCH
from src.realism.image_stats import resize_to_analysis_width

# A row counts as part of a text line when at least this share of its pixels
# are ink. Low, because a single line of small text can be sparse.
_ROW_INK_THRESHOLD = 0.01

# Column-gutter detection works per text-line band, not per pixel row: for
# each band, an "internal gap" is a white run bounded by ink on both sides
# (so trailing whitespace after a short line -- unbounded on the right --
# never counts) and wider than this fraction of page width (well above an
# inter-word gap in justified text, or a bullet-to-text indent). A gutter is
# an x-range covered by such an internal gap in a large share of bands
# (`_COLUMN_GAP_BAND_COVERAGE`), centred in the middle part of the text area
# (`_COLUMN_GUTTER_CENTER_*_FRAC`, as a fraction of the ink bbox width) --
# that excludes left-hugging bullet/list indents, which are wide and
# consistent but never near the middle of the page.
_COLUMN_GAP_MIN_WIDTH_FRAC = 0.025
_COLUMN_GAP_BAND_COVERAGE = 0.6
_COLUMN_GUTTER_CENTER_MIN_FRAC = 0.25
_COLUMN_GUTTER_CENTER_MAX_FRAC = 0.75

# A long horizontal rule survives a morphological opening with a kernel this
# wide (fraction of page width) and is thinner than this many rows; ordinary
# text lines are taller and gap-fragmented, so they do not survive the open.
_RULE_MIN_WIDTH_FRAC = 0.35
_RULE_MAX_HEIGHT_PX = 5


def compute_layout_stats(image: Image.Image) -> Dict[str, float]:
    rgb = np.asarray(image.convert("RGB"))
    rgb_small = resize_to_analysis_width(rgb)
    gray = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bands = _text_line_bands(binary)
    line_count = len(bands)
    if bands:
        heights = [end - start + 1 for start, end in bands]
        median_height_px = float(np.median(heights))
    else:
        median_height_px = 0.0

    px_per_inch = width / A4_WIDTH_INCH
    line_height_frac = median_height_px / width
    line_height_pt = median_height_px / px_per_inch * 72.0

    top, bottom, left, right = _ink_bbox(binary)
    if top is None:
        # Blank page: no ink anywhere.
        margins = {"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0}
        column_count = 1
        text_area_frac = 0.0
    else:
        margins = {
            "top": top / height,
            "bottom": (height - 1 - bottom) / height,
            "left": left / width,
            "right": (width - 1 - right) / width,
        }
        column_count = _estimate_column_count(binary, bands, left, right, width)
        text_area_frac = ((bottom - top + 1) * (right - left + 1)) / (height * width)

    rule_count = _count_horizontal_rules(binary, width)

    return {
        "text_line_count": float(line_count),
        "text_line_height_frac": float(line_height_frac),
        "text_line_height_pt": float(line_height_pt),
        "margin_top_frac": float(margins["top"]),
        "margin_bottom_frac": float(margins["bottom"]),
        "margin_left_frac": float(margins["left"]),
        "margin_right_frac": float(margins["right"]),
        "column_count": float(column_count),
        "text_area_frac": float(text_area_frac),
        "rule_count": float(rule_count),
    }


def _text_line_bands(binary: np.ndarray, threshold: float = _ROW_INK_THRESHOLD) -> List[Tuple[int, int]]:
    """Contiguous row ranges whose ink share exceeds `threshold`."""
    row_frac = binary.mean(axis=1) / 255.0
    is_text_row = row_frac > threshold

    bands: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(is_text_row):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, len(is_text_row) - 1))
    return bands


def _ink_bbox(binary: np.ndarray) -> Tuple[int, int, int, int] | Tuple[None, None, None, None]:
    row_ink = binary.any(axis=1)
    col_ink = binary.any(axis=0)
    if not row_ink.any():
        return None, None, None, None
    row_idx = np.flatnonzero(row_ink)
    col_idx = np.flatnonzero(col_ink)
    return int(row_idx[0]), int(row_idx[-1]), int(col_idx[0]), int(col_idx[-1])


def _false_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous (start, end) [end exclusive] index ranges where `mask` is False."""
    inverted = ~mask
    padded = np.concatenate(([False], inverted, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)  # exclusive
    return list(zip(starts.tolist(), ends.tolist()))


def _estimate_column_count(
    binary: np.ndarray, bands: List[Tuple[int, int]], left: int, right: int, width: int
) -> int:
    """Count persistent column gutters from per-band internal white runs.

    For each text-line band, finds "internal" white runs: bounded by ink on
    both sides *within that band*, so trailing whitespace after a short
    line (unbounded on the right) is never one, and wider than
    `_COLUMN_GAP_MIN_WIDTH_FRAC` of the page -- much wider than an
    inter-word gap in justified text, so ordinary prose does not vote. A
    gutter is an x-range covered by such a run in a large share of bands,
    centred in the middle part of the text area -- a wide, consistent
    bullet-to-text indent still does not qualify, since it hugs the left
    margin rather than sitting mid-page.

    A table with vertically bordered/padded cells produces the same kind of
    wide, bounded, band-internal gap between columns. On a page dominated by
    such a table (most bands come from table rows), this can still be
    mistaken for a document column gutter -- not corrected for here.
    """
    if right <= left or not bands:
        return 1

    region_width = right - left + 1
    min_run_px = max(1, int(round(_COLUMN_GAP_MIN_WIDTH_FRAC * width)))

    votes = np.zeros(region_width, dtype=np.int32)
    counted_bands = 0
    for row_start, row_end in bands:
        band_ink = binary[row_start : row_end + 1, left : right + 1].any(axis=0)
        if not band_ink.any():
            continue
        counted_bands += 1
        for start, end in _false_runs(band_ink):
            if start == 0 or end == region_width:
                continue  # touches this band's own edge: not bounded on both sides
            if end - start >= min_run_px:
                votes[start:end] += 1

    if counted_bands == 0:
        return 1

    gap_mask = votes >= (_COLUMN_GAP_BAND_COVERAGE * counted_bands)
    center_lo = _COLUMN_GUTTER_CENTER_MIN_FRAC * region_width
    center_hi = _COLUMN_GUTTER_CENTER_MAX_FRAC * region_width

    gutters = 0
    for start, end in _false_runs(~gap_mask):
        run_width = end - start
        center = (start + end) / 2.0
        if run_width >= min_run_px and center_lo <= center <= center_hi:
            gutters += 1
    return gutters + 1


def _count_horizontal_rules(binary: np.ndarray, width: int) -> int:
    kernel_width = max(1, int(round(_RULE_MIN_WIDTH_FRAC * width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if not opened.any():
        return 0

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    count = 0
    for i in range(1, num_labels):  # label 0 is the background
        comp_width = stats[i, cv2.CC_STAT_WIDTH]
        comp_height = stats[i, cv2.CC_STAT_HEIGHT]
        if comp_width >= kernel_width and comp_height <= _RULE_MAX_HEIGHT_PX:
            count += 1
    return count
