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

# A column counts towards a gutter when it has ink in at most this share of
# the page's *text* rows (not all of them): a full-width running title or a
# centred page number crossing the gutter must not disqualify it, since real
# two-column pages almost always have exactly that kind of full-width
# furniture. As long as the two-column body dominates the text-row count,
# tolerating a systematic contribution from a few furniture rows still
# leaves a wide margin against a genuine single-column page (which has no
# empty column at all). A gutter run must also be at least this wide (as a
# fraction of page width) and must not touch the ink bounding box's edges.
_COLUMN_GAP_ROW_COVERAGE = 0.65
_COLUMN_GAP_MIN_WIDTH_FRAC = 0.015
_COLUMN_GAP_EDGE_MARGIN_FRAC = 0.02

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
        column_count = _estimate_column_count(binary, top, bottom, left, right, width)
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


def _estimate_column_count(
    binary: np.ndarray, top: int, bottom: int, left: int, right: int, width: int
) -> int:
    """Count persistent vertical white gaps across the page's text rows.

    Evaluated over *text* rows only (rows that are part of a `_text_line_bands`
    band): blank inter-line rows are white in every column regardless of
    layout and would only dilute the signal. A column needs to be white in
    just `_COLUMN_GAP_ROW_COVERAGE` of the text rows, not all of them, so
    full-width furniture that crosses the gutter (running title, centred
    page number) does not defeat detection as long as the two-column body
    dominates the text-row count.
    """
    if bottom <= top or right <= left:
        return 1

    row_frac = binary.mean(axis=1) / 255.0
    text_row_mask = row_frac[top : bottom + 1] > _ROW_INK_THRESHOLD
    if not text_row_mask.any():
        return 1

    region = binary[top : bottom + 1, left : right + 1][text_row_mask]
    ink_share = (region > 0).mean(axis=0)
    gap_mask = ink_share <= (1.0 - _COLUMN_GAP_ROW_COVERAGE)

    region_width = right - left + 1
    min_gap_px = max(1, int(round(_COLUMN_GAP_MIN_WIDTH_FRAC * width)))
    edge_margin_px = max(1, int(round(_COLUMN_GAP_EDGE_MARGIN_FRAC * width)))

    padded = np.concatenate(([False], gap_mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)  # exclusive

    gutters = 0
    for start, end in zip(starts, ends):
        run_width = end - start
        center = (start + end) / 2.0
        if run_width >= min_gap_px and edge_margin_px <= center <= (region_width - edge_margin_px):
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
