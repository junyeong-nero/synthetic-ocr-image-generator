import ast
import json
import re

from typing import Any, Dict, Optional


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TABLE_ROW_PATTERN = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TABLE_CELL_PATTERN = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def _strip_code_fence_language(text: str) -> str:
    cleaned = text.strip()
    if "\n" not in cleaned:
        return cleaned
    first_line, rest = cleaned.split("\n", 1)
    if first_line.strip().lower() in {"json", "javascript", "js", "python", "py", "html"}:
        return rest.strip()
    return cleaned


def _repair_json_candidate(candidate: str) -> str:
    repaired = candidate.strip().replace("\ufeff", "")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _extract_balanced_json_candidate(text: str) -> Optional[str]:
    starts = []
    for idx, ch in enumerate(text):
        if ch in "[{":
            starts.append((idx, ch))
    for start_idx, start_ch in starts:
        end_ch = "}" if start_ch == "{" else "]"
        depth = 0
        in_str = False
        escape = False
        for end_idx in range(start_idx, len(text)):
            ch = text[end_idx]
            if in_str:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    return text[start_idx : end_idx + 1]
    return None


def _json_candidates(output: str) -> list[str]:
    candidates: list[str] = []
    raw = output.strip()
    if raw:
        candidates.append(raw)
    for match in _JSON_FENCE_PATTERN.finditer(output):
        fenced = _strip_code_fence_language(match.group(1))
        if fenced:
            candidates.append(fenced)
    balanced = _extract_balanced_json_candidate(output)
    if balanced:
        candidates.append(balanced)
    return candidates


def _parse_dict_candidate(candidate: str) -> Optional[Dict[str, Any]]:
    repaired = _repair_json_candidate(candidate)
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(repaired)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        return None
    return None


def _extract_markdown_table(output: str) -> Optional[str]:
    lines = [line.rstrip() for line in output.splitlines()]
    table_lines: list[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.count("|") >= 2:
            table_lines.append(stripped)
            in_table = True
        elif in_table:
            break
    if len(table_lines) < 2:
        return None
    if not re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", table_lines[1]):
        return None
    return "\n".join(table_lines)


def _extract_html_tag_block(output: str, tag: str) -> Optional[str]:
    pattern = re.compile(rf"<{tag}\b[^>]*>.*?</{tag}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(output)
    if not match:
        return None
    return match.group(0)


def _markdown_table_to_html(markdown_table: str) -> str:
    rows = []
    for line in markdown_table.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return ""

    html_lines = ["<table>"]
    header = rows[0]
    html_lines.append("<tr>" + "".join(f"<th>{cell}</th>" for cell in header) + "</tr>")
    for row in rows[1:]:
        html_lines.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    html_lines.append("</table>")
    return "\n".join(html_lines)


def table_html_to_json(table_html: str) -> Dict[str, Any]:
    if not isinstance(table_html, str) or "<table" not in table_html.lower():
        return {"cells": [], "num_rows": 0, "num_cols": 0}

    rows = _TABLE_ROW_PATTERN.findall(table_html)
    cells: list[dict[str, Any]] = []
    max_cols = 0
    for row_idx, row in enumerate(rows):
        row_cells = _TABLE_CELL_PATTERN.findall(row)
        max_cols = max(max_cols, len(row_cells))
        for col_idx, cell_html in enumerate(row_cells):
            text = _HTML_TAG_PATTERN.sub("", cell_html).strip()
            cells.append({"row": row_idx, "col": col_idx, "text": text})

    return {
        "cells": cells,
        "num_rows": len(rows),
        "num_cols": max_cols,
    }

def parse_model_output_as_json(output: str) -> Optional[Dict[str, Any]]:
    """Parse model output as JSON, handling various formats."""
    if isinstance(output, dict):
        return output
    if not isinstance(output, str):
        return None

    for candidate in _json_candidates(output):
        parsed = _parse_dict_candidate(candidate)
        if parsed is not None:
            return parsed

    return None


def extract_html_table(output: str) -> str:
    """Extract HTML table from model output."""
    if not isinstance(output, str):
        return str(output)

    output = output.strip()

    table_block = _extract_html_tag_block(output, "table")
    if table_block:
        return table_block

    parsed_json = parse_model_output_as_json(output)
    if isinstance(parsed_json, dict):
        html_candidate = parsed_json.get("html")
        if isinstance(html_candidate, str):
            html_table = _extract_html_tag_block(html_candidate, "table")
            if html_table:
                return html_table

        table_candidate = parsed_json.get("table")
        if isinstance(table_candidate, dict):
            nested_html = table_candidate.get("html")
            if isinstance(nested_html, str):
                html_table = _extract_html_tag_block(nested_html, "table")
                if html_table:
                    return html_table

    for match in _JSON_FENCE_PATTERN.finditer(output):
        content = match.group(1).strip()
        html_table = _extract_html_tag_block(content, "table")
        if html_table:
            return html_table

    markdown_table = _extract_markdown_table(output)
    if markdown_table:
        html_from_markdown = _markdown_table_to_html(markdown_table)
        if html_from_markdown:
            return html_from_markdown

    return ""
