from typing import Any, Dict, List, Optional, Sequence


def _numeric_stats(values: Sequence[float | int]) -> Dict[str, float]:
    if not values:
        return {}
    total = sum(values)
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(total / len(values)),
    }


def _length(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, str)):
        return len(value)
    return len(str(value))


def compute_realism_stats(
    metadata: List[Dict[str, Any]],
    format_name: Optional[str] = None,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"total_samples": len(metadata)}
    if format_name:
        stats["format"] = format_name

    field_presence: Dict[str, int] = {}
    text_fields = [
        "typo_text",
        "original_text",
        "text",
        "GT_markdown",
        "markdown",
        "html",
        "ground_truth",
    ]
    list_fields = ["line_items", "elements"]
    dict_fields = ["entities"]

    text_lengths: Dict[str, List[int]] = {key: [] for key in text_fields}
    list_lengths: Dict[str, List[int]] = {key: [] for key in list_fields}
    dict_lengths: Dict[str, List[int]] = {key: [] for key in dict_fields}
    numeric_values: Dict[str, List[float]] = {}

    format_counts: Dict[str, int] = {}

    for item in metadata:
        for key in item.keys():
            field_presence[key] = field_presence.get(key, 0) + 1

        fmt = item.get("format")
        if isinstance(fmt, str):
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        for key in text_fields:
            if key in item:
                length = _length(item.get(key))
                if length is not None:
                    text_lengths[key].append(length)

        for key in list_fields:
            value = item.get(key)
            if isinstance(value, (list, tuple)):
                list_lengths[key].append(len(value))

        for key in dict_fields:
            value = item.get(key)
            if isinstance(value, dict):
                dict_lengths[key].append(len(value))

        for key, value in item.items():
            if key.endswith("_id") or key in {"seed", "index"}:
                continue
            if isinstance(value, (int, float)):
                numeric_values.setdefault(key, []).append(float(value))

    stats["field_presence"] = field_presence
    if format_counts:
        stats["format_counts"] = format_counts

    stats["text_length_stats"] = {
        key: _numeric_stats(values) for key, values in text_lengths.items() if values
    }
    stats["list_length_stats"] = {
        key: _numeric_stats(values) for key, values in list_lengths.items() if values
    }
    stats["dict_length_stats"] = {
        key: _numeric_stats(values) for key, values in dict_lengths.items() if values
    }
    stats["numeric_field_stats"] = {
        key: _numeric_stats(values) for key, values in numeric_values.items() if values
    }

    return stats


def write_realism_stats(
    output_dir,
    metadata: List[Dict[str, Any]],
    format_name: Optional[str] = None,
) -> str:
    from pathlib import Path
    import json

    path = Path(output_dir) / "realism_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_realism_stats(metadata, format_name=format_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return str(path)
