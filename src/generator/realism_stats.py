import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class _RunningStats:
    count: int = 0
    total: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def update(self, value: float | int) -> None:
        numeric = float(value)
        self.count += 1
        self.total += numeric
        if self.min_value is None or numeric < self.min_value:
            self.min_value = numeric
        if self.max_value is None or numeric > self.max_value:
            self.max_value = numeric

    def to_dict(self) -> Dict[str, float]:
        if self.count <= 0 or self.min_value is None or self.max_value is None:
            return {}
        return {
            "count": float(self.count),
            "min": float(self.min_value),
            "max": float(self.max_value),
            "mean": float(self.total / self.count),
        }


def _length(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, str)):
        return len(value)
    return len(str(value))


class RealismStatsAccumulator:
    def __init__(self, format_name: Optional[str] = None):
        self.format_name = format_name
        self.total_samples = 0
        self.field_presence: Dict[str, int] = {}
        self.text_fields = [
            "typo_text",
            "original_text",
            "text",
            "GT_markdown",
            "markdown",
            "html",
            "ground_truth",
        ]
        self.list_fields = ["line_items", "elements"]
        self.dict_fields = ["entities"]
        self.text_lengths: Dict[str, _RunningStats] = {key: _RunningStats() for key in self.text_fields}
        self.list_lengths: Dict[str, _RunningStats] = {key: _RunningStats() for key in self.list_fields}
        self.dict_lengths: Dict[str, _RunningStats] = {key: _RunningStats() for key in self.dict_fields}
        self.numeric_values: Dict[str, _RunningStats] = {}
        self.format_counts: Dict[str, int] = {}

    def update(self, item: Dict[str, Any]) -> None:
        self.total_samples += 1

        for key in item.keys():
            self.field_presence[key] = self.field_presence.get(key, 0) + 1

        fmt = item.get("format")
        if isinstance(fmt, str):
            self.format_counts[fmt] = self.format_counts.get(fmt, 0) + 1

        for key in self.text_fields:
            if key in item:
                length = _length(item.get(key))
                if length is not None:
                    self.text_lengths[key].update(length)

        for key in self.list_fields:
            value = item.get(key)
            if isinstance(value, (list, tuple)):
                self.list_lengths[key].update(len(value))

        for key in self.dict_fields:
            value = item.get(key)
            if isinstance(value, dict):
                self.dict_lengths[key].update(len(value))

        for key, value in item.items():
            if key.endswith("_id") or key in {"seed", "index"}:
                continue
            if isinstance(value, (int, float)):
                self.numeric_values.setdefault(key, _RunningStats()).update(value)

    def finalize(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"total_samples": self.total_samples}
        if self.format_name:
            stats["format"] = self.format_name

        stats["field_presence"] = dict(self.field_presence)
        if self.format_counts:
            stats["format_counts"] = dict(self.format_counts)

        stats["text_length_stats"] = {
            key: running.to_dict()
            for key, running in self.text_lengths.items()
            if running.count > 0
        }
        stats["list_length_stats"] = {
            key: running.to_dict()
            for key, running in self.list_lengths.items()
            if running.count > 0
        }
        stats["dict_length_stats"] = {
            key: running.to_dict()
            for key, running in self.dict_lengths.items()
            if running.count > 0
        }
        stats["numeric_field_stats"] = {
            key: running.to_dict()
            for key, running in self.numeric_values.items()
            if running.count > 0
        }
        return stats


def compute_realism_stats(
    metadata: Iterable[Dict[str, Any]],
    format_name: Optional[str] = None,
) -> Dict[str, Any]:
    accumulator = RealismStatsAccumulator(format_name=format_name)
    for item in metadata:
        accumulator.update(item)
    return accumulator.finalize()


def write_realism_stats(
    output_dir: str | Path,
    metadata: Iterable[Dict[str, Any]] | RealismStatsAccumulator,
    format_name: Optional[str] = None,
) -> str:
    path = Path(output_dir) / "realism_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(metadata, RealismStatsAccumulator):
        stats = metadata.finalize()
    else:
        stats = compute_realism_stats(metadata, format_name=format_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return str(path)
