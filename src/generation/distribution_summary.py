"""Summaries of generated sample distributions for dataset cards."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

SUMMARY_FIELDS = (
    "document_family",
    "capture_channel",
    "visual_difficulty",
    "target_dpi",
    "distribution_profile",
)


def summarize_metadata_distribution(metadata_path: Path) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = {field: Counter() for field in SUMMARY_FIELDS}
    block_counter: Counter[str] = Counter()
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            for field in SUMMARY_FIELDS:
                value = row.get(field)
                if value not in (None, ""):
                    counters[field][str(value)] += 1
            block_counts = row.get("block_type_counts")
            if isinstance(block_counts, dict):
                for block_type, count in block_counts.items():
                    try:
                        block_counter[str(block_type)] += int(count)
                    except (TypeError, ValueError):
                        continue

    summary = {field: dict(counter.most_common()) for field, counter in counters.items() if counter}
    if block_counter:
        summary["block_type"] = dict(block_counter.most_common())
    return summary


def format_distribution_tables(summary: Dict[str, Dict[str, int]]) -> list[str]:
    lines: list[str] = []
    for field, counts in summary.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        lines.extend([f"### `{field}`", "", "| Value | Count | Share |", "|---|---:|---:|"])
        for value, count in counts.items():
            lines.append(f"| {value} | {count:,} | {count / total:.1%} |")
        lines.append("")
    return lines
