"""Aggregate per-image stats, compare distributions, and suggest profile specs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
from PIL import Image

from src.realism.image_stats import compute_image_stats

logger = logging.getLogger(__name__)

QUANTILE_POINTS = np.linspace(0.0, 1.0, 101)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Metrics that best reflect "how hard does this look" for the compare report.
KEY_METRICS = (
    "est_dpi_a4",
    "abs_skew_deg",
    "laplacian_var",
    "noise_sigma",
    "contrast",
    "background_luma",
    "colorfulness",
    "ink_ratio",
    "aspect_ratio",
    "is_grayscale",
    "is_binary",
    "is_colored_background",
)


def iter_image_paths(root: Path, limit: Optional[int] = None) -> Iterator[Path]:
    count = 0
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path
            count += 1
            if limit is not None and count >= limit:
                return


def iter_metadata_image_paths(
    metadata_path: Path,
    limit: Optional[int] = None,
    where: Optional[Dict[str, str]] = None,
) -> Iterator[Path]:
    count = 0
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if where and any(str(row.get(key)) != value for key, value in where.items()):
                continue
            file_name = row.get("file_name")
            if not file_name:
                continue
            path = Path(file_name)
            if not path.is_absolute() and not path.exists():
                path = metadata_path.parent / path
            yield path
            count += 1
            if limit is not None and count >= limit:
                return


def iter_hf_images(
    dataset: str,
    *,
    split: str = "train",
    image_column: str = "image",
    config: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[Image.Image]:
    from datasets import load_dataset

    stream = load_dataset(dataset, config, split=split, streaming=True)
    for index, row in enumerate(stream):
        if limit is not None and index >= limit:
            return
        image = row.get(image_column)
        if isinstance(image, Image.Image):
            yield image


def measure_images(images: Iterable[Image.Image | Path]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for item in images:
        try:
            if isinstance(item, Path):
                with Image.open(item) as handle:
                    rows.append(compute_image_stats(handle))
            else:
                rows.append(compute_image_stats(item))
        except Exception as exc:  # pragma: no cover - defensive for bad files
            logger.warning("Skipping unreadable image %s: %s", item, exc)
    return rows


def summarize_stats(rows: List[Dict[str, float]], *, source: str = "") -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if rows:
        for key in rows[0].keys():
            values = np.array([row[key] for row in rows if key in row], dtype=np.float64)
            if values.size == 0:
                continue
            metrics[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
                "quantiles": [round(float(v), 6) for v in np.quantile(values, QUANTILE_POINTS)],
            }
    return {
        "source": source,
        "count": len(rows),
        "metrics": metrics,
        "suggested_profile_specs": suggest_profile_specs(rows),
    }


def _histogram_spec(values: np.ndarray, bins: int, clip: Optional[tuple[float, float]] = None) -> Dict[str, Any]:
    if clip is not None:
        values = np.clip(values, *clip)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return {"value": round(lo, 3)}
    weights, edges = np.histogram(values, bins=bins, range=(lo, hi))
    return {
        "histogram": {
            "bins": [round(float(edge), 3) for edge in edges],
            "weights": [int(weight) for weight in weights],
        }
    }


def suggest_profile_specs(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    """Profile specs that can be measured directly from images.

    Degradation strengths such as blur_sigma or noise_sigma are *not* read off
    one-to-one (rendering and content also affect them); calibrate those by
    generating a pilot set and running ``distribution compare``.
    """
    if not rows:
        return {}

    def column(key: str) -> np.ndarray:
        return np.array([row[key] for row in rows], dtype=np.float64)

    skew = column("skew_deg")
    return {
        "dpi": {**_histogram_spec(column("est_dpi_a4"), bins=8, clip=(60, 400)), "round": 0},
        "skew_deg": {**_histogram_spec(skew, bins=12, clip=(-10, 10)), "round": 2},
        "grayscale": {"p": round(float(column("is_grayscale").mean()), 3)},
        "binarize": {"p": round(float(column("is_binary").mean()), 3)},
        "colored_background": {"p": round(float(column("is_colored_background").mean()), 3)},
    }


def compare_summaries(reference: Dict[str, Any], candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-metric gap between two stat summaries.

    ``w1_norm`` is the 1-D Wasserstein distance (mean absolute quantile gap)
    divided by the reference inter-quantile range (p95 - p05), so values are
    comparable across metrics. Rough reading: < 0.1 close, 0.1-0.3 noticeable,
    > 0.3 clearly different.
    """
    results: List[Dict[str, Any]] = []
    ref_metrics = reference.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})
    for key in KEY_METRICS:
        ref = ref_metrics.get(key)
        cand = cand_metrics.get(key)
        if not ref or not cand:
            continue
        ref_q = np.array(ref["quantiles"], dtype=np.float64)
        cand_q = np.array(cand["quantiles"], dtype=np.float64)
        w1 = float(np.mean(np.abs(ref_q - cand_q)))
        # Floor the spread so near-constant reference metrics (e.g. white
        # background luma) do not blow up the normalised distance.
        spread = max(float(ref["p95"] - ref["p05"]), 0.05 * abs(float(ref["median"])), 1e-6)
        if key.startswith("is_"):
            spread = 1.0
        w1_norm = w1 / spread
        direction = "higher" if cand["median"] > ref["median"] else "lower"
        if key.startswith("is_"):
            direction = "higher" if cand["mean"] > ref["mean"] else "lower"
        results.append(
            {
                "metric": key,
                "reference_median": ref["median"],
                "candidate_median": cand["median"],
                "reference_mean": ref["mean"],
                "candidate_mean": cand["mean"],
                "w1": w1,
                "w1_norm": w1_norm,
                "candidate_is": direction,
                "verdict": "close" if w1_norm < 0.1 else ("noticeable" if w1_norm < 0.3 else "different"),
            }
        )
    return results


def format_comparison_markdown(rows: List[Dict[str, Any]], reference: str, candidate: str) -> str:
    lines = [
        f"# Distribution gap: `{candidate}` vs `{reference}`",
        "",
        "| Metric | Reference median | Candidate median | W1 (norm) | Candidate is | Verdict |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: -item["w1_norm"]):
        ref_value = row["reference_mean"] if row["metric"].startswith("is_") else row["reference_median"]
        cand_value = row["candidate_mean"] if row["metric"].startswith("is_") else row["candidate_median"]
        lines.append(
            f"| {row['metric']} | {ref_value:.3f} | {cand_value:.3f} | {row['w1_norm']:.3f} "
            f"| {row['candidate_is']} | {row['verdict']} |"
        )
    lines.append("")
    lines.append("For `is_*` metrics the columns show the share of images (mean), not the median.")
    return "\n".join(lines) + "\n"
