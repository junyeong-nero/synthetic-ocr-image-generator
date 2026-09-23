"""Real-world distribution profiles for synthetic OCR generation.

A distribution profile describes *what the real target data looks like* and is
loaded from YAML under ``configs/generator/distributions``.  It controls:

- document family mix (fed into coverage targets)
- block type weights used when filling non-required block slots
- physical page typography (body font size in points, line spacing)
- capture channels (born-digital / scanned / photographed) with per-channel
  resolution (DPI) and degradation parameter distributions

Every numeric field accepts a *distribution spec* (see ``sample_value``) so the
profile can encode measured histograms instead of hand-tuned uniform ranges.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_DISTRIBUTION_DIR = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "generator" / "distributions"
)

# A4 paper width in inches; used to translate target DPI into render scale.
A4_WIDTH_INCH = 8.27


def sample_value(spec: Any, rng: random.Random) -> Any:
    """Sample a value from a distribution spec.

    Supported spec forms:

    - scalar (``3``, ``0.5``, ``"text"``, ``true``): returned as-is
    - ``[lo, hi]`` list of two numbers: uniform in range
    - ``{uniform: [lo, hi]}``
    - ``{normal: [mean, std]}``
    - ``{lognormal: [mu, sigma]}`` (parameters of the underlying normal)
    - ``{beta: [a, b], range: [lo, hi]}`` (range defaults to ``[0, 1]``)
    - ``{choices: [...], weights: [...]}`` (weights optional)
    - ``{histogram: {bins: [e0, e1, ..., en], weights: [w1, ..., wn]}}``:
      piecewise-uniform density, e.g. produced by ``distribution measure``
    - ``{p: 0.3}``: Bernoulli, returns a bool

    Any dict spec may add ``clip: [lo, hi]`` and ``round: <ndigits>``;
    ``round: 0`` returns an int.
    """
    if isinstance(spec, bool) or spec is None or isinstance(spec, str):
        return spec
    if isinstance(spec, (int, float)):
        return spec
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            lo, hi = sorted((float(spec[0]), float(spec[1])))
            return rng.uniform(lo, hi)
        raise ValueError(f"Unsupported list distribution spec: {spec!r}")
    if not isinstance(spec, Mapping):
        raise ValueError(f"Unsupported distribution spec: {spec!r}")

    if "p" in spec:
        return rng.random() < float(spec["p"])

    if "choices" in spec:
        choices = list(spec["choices"])
        weights = spec.get("weights")
        if not choices:
            raise ValueError("choices spec must not be empty")
        if weights is not None and len(weights) != len(choices):
            raise ValueError("choices and weights must have the same length")
        return rng.choices(choices, weights=weights, k=1)[0]

    if "uniform" in spec:
        lo, hi = sorted(float(v) for v in spec["uniform"])
        value = rng.uniform(lo, hi)
    elif "normal" in spec:
        mean, std = (float(v) for v in spec["normal"])
        value = rng.gauss(mean, std)
    elif "lognormal" in spec:
        mu, sigma = (float(v) for v in spec["lognormal"])
        value = rng.lognormvariate(mu, sigma)
    elif "beta" in spec:
        a, b = (float(v) for v in spec["beta"])
        lo, hi = (float(v) for v in spec.get("range", (0.0, 1.0)))
        value = lo + (hi - lo) * rng.betavariate(a, b)
    elif "histogram" in spec:
        value = _sample_histogram(spec["histogram"], rng)
    elif "value" in spec:
        value = spec["value"]
    else:
        raise ValueError(f"Unknown distribution spec keys: {sorted(spec)}")

    clip = spec.get("clip")
    if clip is not None:
        lo, hi = (float(v) for v in clip)
        value = min(hi, max(lo, value))
    if "round" in spec:
        digits = int(spec["round"])
        value = int(round(value)) if digits == 0 else round(value, digits)
    return value


def _sample_histogram(hist: Mapping[str, Any], rng: random.Random) -> float:
    bins = [float(v) for v in hist.get("bins", [])]
    weights = [max(0.0, float(v)) for v in hist.get("weights", [])]
    if len(bins) < 2 or len(weights) != len(bins) - 1:
        raise ValueError("histogram spec needs len(bins) == len(weights) + 1")
    if sum(weights) <= 0:
        raise ValueError("histogram weights must sum to a positive value")
    index = rng.choices(range(len(weights)), weights=weights, k=1)[0]
    return rng.uniform(bins[index], bins[index + 1])


def _normalize_weights(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    cleaned = {str(k): max(0.0, float(v)) for k, v in raw.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in cleaned.items()}


@dataclass(frozen=True)
class CaptureChannel:
    name: str
    weight: float
    dpi: Any
    degradations: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureSample:
    channel: str
    dpi: int
    params: Dict[str, Any]

    def difficulty(self) -> str:
        """Coarse visual difficulty bucket derived from sampled parameters."""
        score = 0.0
        score += min(abs(float(self.params.get("skew_deg", 0.0))) / 2.0, 1.5)
        score += float(self.params.get("perspective", 0.0)) * 20.0
        score += float(self.params.get("blur_sigma", 0.0)) * 1.2
        score += float(self.params.get("noise_sigma", 0.0)) / 6.0
        jpeg_quality = self.params.get("jpeg_quality")
        if jpeg_quality:
            score += max(0.0, (85 - float(jpeg_quality)) / 25.0)
        score += max(0.0, (150 - self.dpi) / 75.0)
        score += 1.0 if self.params.get("binarize") else 0.0
        score += float(self.params.get("shadow_strength", 0.0)) * 2.0
        if score < 1.0:
            return "easy"
        if score < 2.5:
            return "medium"
        return "hard"


@dataclass(frozen=True)
class DistributionProfile:
    profile_id: str
    version: str
    description: str
    sources: List[Dict[str, Any]]
    family_mix: Dict[str, float]
    block_weights: Dict[str, float]
    typography: Dict[str, Any]
    capture_channels: List[CaptureChannel]
    source_path: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source_path: str = "") -> "DistributionProfile":
        channels_raw = data.get("capture_channels") or {}
        if not isinstance(channels_raw, Mapping) or not channels_raw:
            raise ValueError("distribution profile requires non-empty 'capture_channels'")
        channels: List[CaptureChannel] = []
        for name, cfg in channels_raw.items():
            cfg = cfg or {}
            channels.append(
                CaptureChannel(
                    name=str(name),
                    weight=max(0.0, float(cfg.get("weight", 1.0))),
                    dpi=cfg.get("dpi", 150),
                    degradations=dict(cfg.get("degradations") or {}),
                )
            )
        if sum(channel.weight for channel in channels) <= 0:
            raise ValueError("capture channel weights must sum to a positive value")

        return cls(
            profile_id=str(data.get("id") or Path(source_path).stem or "custom"),
            version=str(data.get("version", "1")),
            description=str(data.get("description", "")).strip(),
            sources=list(data.get("sources") or []),
            family_mix=_normalize_weights(data.get("family_mix")),
            block_weights={
                str(k): max(0.0, float(v))
                for k, v in (data.get("block_weights") or {}).items()
            },
            typography=dict(data.get("typography") or {}),
            capture_channels=channels,
            source_path=source_path,
        )

    def coverage_targets(self) -> Dict[str, float]:
        return dict(self.family_mix)

    def sample_capture(self, rng: random.Random) -> CaptureSample:
        weights = [channel.weight for channel in self.capture_channels]
        channel = rng.choices(self.capture_channels, weights=weights, k=1)[0]
        dpi = int(round(float(sample_value(channel.dpi, rng))))
        params: Dict[str, Any] = {}
        for key, spec in channel.degradations.items():
            params[key] = sample_value(spec, rng)
        return CaptureSample(channel=channel.name, dpi=max(50, dpi), params=params)

    def sample_typography(self, rng: random.Random) -> Dict[str, Any]:
        return {key: sample_value(spec, rng) for key, spec in self.typography.items()}


def available_profiles(config_dir: Optional[Path] = None) -> List[str]:
    root = Path(config_dir) if config_dir else DEFAULT_DISTRIBUTION_DIR
    if not root.exists():
        return []
    return sorted(path.stem for path in root.glob("*.yaml"))


def load_distribution_profile(
    name_or_path: str,
    config_dir: Optional[Path] = None,
) -> DistributionProfile:
    """Load a profile by name (``real_world_v1``) or by explicit YAML path."""
    candidate = Path(name_or_path)
    if not candidate.suffix:
        root = Path(config_dir) if config_dir else DEFAULT_DISTRIBUTION_DIR
        candidate = root / f"{name_or_path}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Distribution profile '{name_or_path}' not found. "
            f"Available: {', '.join(available_profiles(config_dir)) or 'none'}"
        )
    with open(candidate, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return DistributionProfile.from_dict(data, source_path=str(candidate))


def render_scale_for_dpi(target_dpi: int, page_width_css_px: int) -> float:
    """Device scale factor that maps a CSS page width onto A4 at ``target_dpi``."""
    if page_width_css_px <= 0:
        return 1.0
    nominal_dpi = page_width_css_px / A4_WIDTH_INCH
    scale = float(target_dpi) / nominal_dpi
    if not math.isfinite(scale):
        return 1.0
    return max(0.5, min(4.0, scale))


def points_to_css_px(points: float, page_width_css_px: int) -> int:
    """Convert a physical font size (pt) to CSS px on the virtual A4 page."""
    nominal_dpi = page_width_css_px / A4_WIDTH_INCH
    return max(8, int(round(points / 72.0 * nominal_dpi)))
