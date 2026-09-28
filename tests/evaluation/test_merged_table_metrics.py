"""evaluate_markdown_blocks on merged-cell HTML tables from the generator."""

import random
import re

import pytest

from src.generator.data_provider import DataProvider
from src.generator.table_generator import TableGenerator
from src.metrics.markdown_block_metrics import (
    _normalize_table_html,
    evaluate_markdown_blocks,
    split_markdown_blocks,
)

_SPAN_ATTR_RE = re.compile(r'\s(?:colspan|rowspan)="\d+"')


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def _merged_table(schema: str, seed: int = 0) -> str:
    random.seed(seed)
    generator = TableGenerator(
        data=DataProvider(lang="ko", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        table_schemas={schema: 1.0},
        merged_table_ratio=1.0,
    )
    section = generator.generate_sections(section_count=1, row_range=(3, 5), column_range=(4, 5))[0]
    return section.partition("\n\n")[2]


def _page(table_html: str) -> str:
    return f"# 보고서\n\n## 현황\n\n본문 문단입니다.\n\n{table_html}\n\n맺음 문단입니다.\n"


def _expanded_grid_html(table_html: str) -> str:
    """The table a span-unaware reader produces: every spanned slot repeated."""
    rows = re.findall(r"<tr>(.*?)</tr>", table_html)
    occupied: dict[tuple[int, int], str] = {}
    for r, row in enumerate(rows):
        c = 0
        for attrs, text in re.findall(r"<t[hd]([^>]*)>(.*?)</t[hd]>", row):
            while (r, c) in occupied:
                c += 1
            colspan = int((re.search(r'colspan="(\d+)"', attrs) or [0, 1])[1])
            rowspan = int((re.search(r'rowspan="(\d+)"', attrs) or [0, 1])[1])
            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied[(r + dr, c + dc)] = text
            c += colspan
    width = max(c for _, c in occupied) + 1
    lines = ["<table>"]
    for r in range(len(rows)):
        lines.append("<tr>" + "".join(f"<td>{occupied[(r, c)]}</td>" for c in range(width)) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


@pytest.mark.parametrize("schema", ["statistics", "financial", "budget", "schedule", "roster", "order"])
def test_identical_merged_table_scores_one(schema: str) -> None:
    gt = _page(_merged_table(schema))

    blocks = split_markdown_blocks(gt)
    assert [block["type"] for block in blocks] == ["text", "table", "text"]
    metrics = evaluate_markdown_blocks(gt, gt)
    assert metrics["markdown_table_teds"] == 1.0
    assert metrics["markdown_overall_score"] == 1.0


@pytest.mark.parametrize("schema", ["statistics", "financial", "budget", "schedule", "roster", "order"])
def test_flattened_spans_are_penalized(schema: str) -> None:
    table_html = _merged_table(schema)
    gt = _page(table_html)

    dropped_attrs = _page(_SPAN_ATTR_RE.sub("", table_html))
    expanded = _page(_expanded_grid_html(table_html))

    assert _SPAN_ATTR_RE.search(table_html)
    for prediction in (dropped_attrs, expanded):
        metrics = evaluate_markdown_blocks(prediction, gt)
        assert metrics["markdown_table_teds"] < 1.0
        assert metrics["markdown_text_score"] == 1.0


def test_numeric_alignment_style_does_not_affect_the_table_score() -> None:
    table_html = _merged_table("statistics")
    unstyled = table_html.replace(' style="text-align: right;"', "")

    assert unstyled != table_html
    assert _normalize_table_html(unstyled) == _normalize_table_html(table_html)
    assert evaluate_markdown_blocks(_page(unstyled), _page(table_html))["markdown_table_teds"] == 1.0


def test_teds_structure_scores_spans() -> None:
    pytest.importorskip("lxml")
    pytest.importorskip("apted")
    from src.metrics.table_edit_distance import TEDS

    table_html = _merged_table("statistics")
    gt = _normalize_table_html(table_html)
    teds = TEDS(structure_only=True)

    assert teds.evaluate(gt, gt)["teds"] == 1.0
    flattened = _normalize_table_html(_expanded_grid_html(table_html))
    assert teds.evaluate(flattened, gt)["teds"] < 1.0
