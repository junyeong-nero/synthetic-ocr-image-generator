import json
from typing import Any

from utils import markdown_to_json_ast


def json_to_markdown(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def build_unified_ground_truth(fmt: str, metadata: dict[str, Any]) -> tuple[str, Any]:
    if fmt == "markdown":
        markdown_text = str(metadata.get("GT_markdown", metadata.get("markdown", "")))
        return markdown_text, markdown_to_json_ast(markdown_text)

    fallback_json = {"ground_truth": metadata.get("ground_truth", ""), "format": fmt}
    return json_to_markdown(fallback_json), fallback_json


def attach_unified_ground_truth(fmt: str, metadata: dict[str, Any]) -> dict[str, Any]:
    markdown_gt, json_gt = build_unified_ground_truth(fmt, metadata)
    updated = dict(metadata)
    updated["GT_markdown"] = markdown_gt
    updated["GT_json"] = json_gt
    updated.pop("ground_truth", None)
    updated.pop("markdown", None)
    updated.pop("json", None)
    return updated
