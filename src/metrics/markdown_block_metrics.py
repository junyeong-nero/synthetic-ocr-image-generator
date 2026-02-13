from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable

import numpy as np

from evaluation.utils import extract_html_table
from metrics import TEDS
from metrics.edit_distance import cer


_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")
_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_DISPLAY_DOLLAR_RE = re.compile(r"\$\$([\s\S]+?)\$\$")
_DISPLAY_BRACKET_RE = re.compile(r"\\\[([\s\S]+?)\\\]")
_INLINE_DOLLAR_RE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
_INLINE_PAREN_RE = re.compile(r"\\\(([^\n]+?)\\\)")
_FORMULA_TOKEN_RE = re.compile(r"(\\[a-zA-Z]+|\\.|[A-Za-z0-9]+|[^\s])")


def _overlaps(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    return span_a[0] < span_b[1] and span_b[0] < span_a[1]


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and stripped.count("|") >= 2


def _find_markdown_table_spans(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    spans: list[tuple[int, int, str]] = []
    i = 0
    while i + 1 < len(lines):
        first = lines[i].strip()
        second = lines[i + 1].strip()
        if first.count("|") >= 2 and _TABLE_SEPARATOR_RE.match(second):
            start = offsets[i]
            j = i + 2
            while j < len(lines) and _is_table_row(lines[j]):
                j += 1
            end = offsets[j] if j < len(lines) else len(text)
            spans.append((start, end, "table"))
            i = j
            continue
        i += 1
    return spans


def _find_regex_spans(pattern: re.Pattern[str], text: str, kind: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), kind) for m in pattern.finditer(text)]


def _strip_formula_delimiters(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    if stripped.startswith("\\[") and stripped.endswith("\\]") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    if stripped.startswith("\\(") and stripped.endswith("\\)") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        return stripped[1:-1].strip()
    return stripped


def split_markdown_blocks(markdown: str) -> list[dict[str, Any]]:
    text = markdown if isinstance(markdown, str) else str(markdown)
    taken: list[tuple[int, int]] = []
    spans: list[tuple[int, int, str]] = []

    def add_spans(candidates: list[tuple[int, int, str]]) -> None:
        for start, end, kind in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
            if start >= end:
                continue
            if any(_overlaps((start, end), existing) for existing in taken):
                continue
            spans.append((start, end, kind))
            taken.append((start, end))

    add_spans(_find_regex_spans(_HTML_TABLE_RE, text, "table"))
    add_spans(_find_markdown_table_spans(text))
    add_spans(_find_regex_spans(_DISPLAY_DOLLAR_RE, text, "formula"))
    add_spans(_find_regex_spans(_DISPLAY_BRACKET_RE, text, "formula"))
    add_spans(_find_regex_spans(_INLINE_DOLLAR_RE, text, "formula"))
    add_spans(_find_regex_spans(_INLINE_PAREN_RE, text, "formula"))

    spans.sort(key=lambda item: item[0])
    blocks: list[dict[str, Any]] = []
    cursor = 0

    def append_text(raw_text: str) -> None:
        if raw_text.strip():
            blocks.append({"type": "text", "content": raw_text})

    for start, end, kind in spans:
        if cursor < start:
            append_text(text[cursor:start])
        raw = text[start:end]
        if kind == "table":
            table_html = extract_html_table(raw)
            blocks.append({"type": "table", "content": table_html if table_html else raw.strip()})
        else:
            blocks.append({"type": "formula", "content": _strip_formula_delimiters(raw)})
        cursor = end

    if cursor < len(text):
        append_text(text[cursor:])

    for idx, block in enumerate(blocks):
        block["order"] = idx
    return blocks


def normalize_markdown_text(text: str) -> str:
    lines: list[str] = []
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        plain_text = re.sub(r"[#$`*_>\-+|\[\]()!~]", "", stripped)
        plain_text = re.sub(r"\s+", " ", plain_text).strip()
        if plain_text:
            lines.append(plain_text)
    return "\n".join(lines)


def _safe_similarity_from_cer(reference: str, hypothesis: str) -> float:
    return max(0.0, 1.0 - cer(reference, hypothesis))


def _pairwise_type_score(
    gt_items: list[dict[str, Any]],
    pred_items: list[dict[str, Any]],
    scorer: Callable[[dict[str, Any] | None, dict[str, Any] | None], float],
) -> float:
    total = max(len(gt_items), len(pred_items))
    if total == 0:
        return 1.0

    scores: list[float] = []
    for idx in range(total):
        gt_item = gt_items[idx] if idx < len(gt_items) else None
        pred_item = pred_items[idx] if idx < len(pred_items) else None
        scores.append(float(scorer(gt_item, pred_item)))
    return float(np.mean(scores))


def _text_block_score(gt_item: dict[str, Any] | None, pred_item: dict[str, Any] | None) -> float:
    if gt_item is None or pred_item is None:
        return 0.0
    gt_text = normalize_markdown_text(str(gt_item.get("content", "")))
    pred_text = normalize_markdown_text(str(pred_item.get("content", "")))
    if not gt_text and not pred_text:
        return 1.0
    return _safe_similarity_from_cer(gt_text, pred_text)


def _table_block_score(gt_item: dict[str, Any] | None, pred_item: dict[str, Any] | None) -> float:
    if gt_item is None or pred_item is None:
        return 0.0

    gt_html = extract_html_table(str(gt_item.get("content", "")))
    pred_html = extract_html_table(str(pred_item.get("content", "")))

    if not gt_html and not pred_html:
        return 1.0
    if not gt_html or not pred_html:
        return 0.0

    if TEDS is None:
        return _safe_similarity_from_cer(gt_html, pred_html)

    teds = TEDS(structure_only=True)
    result = teds.evaluate(pred_html, gt_html)
    return float(result.get("teds", 0.0))


def _formula_tokens(text: str) -> list[str]:
    return [token for token in _FORMULA_TOKEN_RE.findall(text) if token.strip()]


def _bleu_score(reference_tokens: list[str], hypothesis_tokens: list[str], max_n: int = 4) -> float:
    if not reference_tokens and not hypothesis_tokens:
        return 1.0
    if not hypothesis_tokens:
        return 0.0

    weights = [1.0 / float(max_n)] * max_n
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        if len(hypothesis_tokens) < n:
            precisions.append(1.0 / (len(hypothesis_tokens) + 1.0))
            continue

        hyp_ngrams = Counter(
            tuple(hypothesis_tokens[i : i + n]) for i in range(0, len(hypothesis_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(reference_tokens[i : i + n]) for i in range(0, len(reference_tokens) - n + 1)
        )

        clipped = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        precisions.append((clipped + 1.0) / (total + 1.0))

    geo_mean = math.exp(sum(weight * math.log(max(precision, 1e-12)) for weight, precision in zip(weights, precisions)))
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)
    if hyp_len == 0:
        brevity_penalty = 0.0
    elif hyp_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (float(ref_len) / float(hyp_len)))
    return float(max(0.0, min(1.0, brevity_penalty * geo_mean)))


def _normalize_formula(text: str) -> str:
    normalized = _strip_formula_delimiters(text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _formula_block_score(gt_item: dict[str, Any] | None, pred_item: dict[str, Any] | None) -> float:
    if gt_item is None or pred_item is None:
        return 0.0

    gt_formula = _normalize_formula(str(gt_item.get("content", "")))
    pred_formula = _normalize_formula(str(pred_item.get("content", "")))
    if not gt_formula and not pred_formula:
        return 1.0
    if gt_formula == pred_formula:
        return 1.0

    bleu = _bleu_score(_formula_tokens(gt_formula), _formula_tokens(pred_formula), max_n=4)
    char_score = _safe_similarity_from_cer(gt_formula, pred_formula)
    return float((bleu + char_score) / 2.0)


def _block_similarity(block_a: dict[str, Any], block_b: dict[str, Any]) -> float:
    block_type = block_a.get("type")
    if block_type != block_b.get("type"):
        return 0.0
    if block_type == "text":
        return _text_block_score(block_a, block_b)
    if block_type == "table":
        return _table_block_score(block_a, block_b)
    if block_type == "formula":
        return _formula_block_score(block_a, block_b)
    return 0.0


def _matched_pairs_for_order(
    gt_blocks: list[dict[str, Any]], pred_blocks: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    unmatched_gt = set(range(len(gt_blocks)))
    matches: list[tuple[int, int]] = []

    for pred_idx, pred_block in enumerate(pred_blocks):
        best_gt_idx: int | None = None
        best_score = 0.0
        for gt_idx in sorted(unmatched_gt):
            score = _block_similarity(gt_blocks[gt_idx], pred_block)
            if score > best_score:
                best_score = score
                best_gt_idx = gt_idx
        if best_gt_idx is not None and best_score > 0.0:
            matches.append((pred_idx, best_gt_idx))
            unmatched_gt.remove(best_gt_idx)

    return matches


def _order_score(gt_blocks: list[dict[str, Any]], pred_blocks: list[dict[str, Any]]) -> float:
    total = max(len(gt_blocks), len(pred_blocks))
    if total == 0:
        return 1.0

    matched = _matched_pairs_for_order(gt_blocks, pred_blocks)
    if not matched:
        return 0.0

    matched.sort(key=lambda pair: pair[0])
    gt_sequence = [gt_idx for _, gt_idx in matched]
    if len(gt_sequence) <= 1:
        pair_order_score = 1.0
    else:
        concordant = 0
        total_pairs = 0
        for i in range(len(gt_sequence) - 1):
            for j in range(i + 1, len(gt_sequence)):
                total_pairs += 1
                if gt_sequence[i] < gt_sequence[j]:
                    concordant += 1
        pair_order_score = float(concordant / total_pairs) if total_pairs else 1.0

    coverage = float(len(matched) / total)
    return pair_order_score * coverage


def evaluate_markdown_blocks(pred_markdown: str, gt_markdown: str) -> dict[str, float]:
    pred_blocks = split_markdown_blocks(pred_markdown)
    gt_blocks = split_markdown_blocks(gt_markdown)

    gt_text = [block for block in gt_blocks if block.get("type") == "text"]
    pred_text = [block for block in pred_blocks if block.get("type") == "text"]
    gt_table = [block for block in gt_blocks if block.get("type") == "table"]
    pred_table = [block for block in pred_blocks if block.get("type") == "table"]
    gt_formula = [block for block in gt_blocks if block.get("type") == "formula"]
    pred_formula = [block for block in pred_blocks if block.get("type") == "formula"]

    text_score = _pairwise_type_score(gt_text, pred_text, _text_block_score)
    table_score = _pairwise_type_score(gt_table, pred_table, _table_block_score)
    formula_score = _pairwise_type_score(gt_formula, pred_formula, _formula_block_score)
    order_score = _order_score(gt_blocks, pred_blocks)

    overall_score = float((text_score + table_score + formula_score + order_score) / 4.0)

    return {
        "markdown_text_score": text_score,
        "markdown_table_teds": table_score,
        "markdown_formula_score": formula_score,
        "markdown_order_score": order_score,
        "markdown_overall_score": overall_score,
    }
