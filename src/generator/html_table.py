"""Merged-cell tables written as a raw HTML ``<table>`` block.

Markdown pipe tables cannot express row or column spans, so a share of the
schema tables (``content.merged_table_ratio``, see
:class:`src.generator.table_generator.TableGenerator`) is written into
``GT_markdown`` as raw HTML instead. python-markdown passes a raw HTML block
through to the rendered page unchanged, and mistune (``GT_json``) parses it
as one ``block_html`` token. The block never contains a blank line, so the
blank-line block splitters (``fit_markdown_to_sheet``, ``text_mutation``)
see one chunk.

Layouts reuse the :mod:`src.generator.table_schemas` columns and the row
values from ``TableGenerator._build_<schema>_rows``:

- column groups: a group header spans its leaf columns (``colspan``) and the
  ungrouped headers span both header rows (``rowspan="2"``). statistics puts
  상반기 | 하반기 under each year, financial 당기 | 전기 under 금액(백만원)
  and 금액 | 비율 under 증감, budget 금액 | 집행률 under 집행 (or 예산액 |
  집행액 under 금액), schedule 일자 | 시간 under 일시.
- row groups: consecutive rows with the same label share one ``rowspan``
  cell: several schedule events on one date, several roster people in one
  department (moved to the first column).
- summary row: order's 합계 label spans every column before 금액.

Numeric cells carry the same inline ``style="text-align: right;"`` that
python-markdown writes for a ``---:`` column, so they align the way pipe
tables do. Centring span headers and vertically centring row-group labels
is left to the renderer CSS (``layout_css.table_style_css``), which keeps
the ground truth free of presentational attributes.
"""

from __future__ import annotations

import html
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.generator.table_schemas import NUMERIC_COLUMN_TYPES, TableColumn, resolve_header

# python-markdown's attribute for a `---:` column.
RIGHT_ALIGN_STYLE = "text-align: right;"

# Longest run of rows that share one row-group label.
_MAX_ROW_GROUP = 3

# Share of a year's total that falls in the first half (statistics).
_FIRST_HALF_SHARE = (0.42, 0.58)


@dataclass(frozen=True)
class HtmlCell:
    """One ``<th>``/``<td>``; ``numeric`` cells are right-aligned."""

    text: str
    colspan: int = 1
    rowspan: int = 1
    numeric: bool = False


@dataclass(frozen=True)
class HtmlTable:
    """Header rows (``<thead>``) and body rows (``<tbody>``) over ``column_count`` leaf columns."""

    header_rows: Tuple[Tuple[HtmlCell, ...], ...]
    body_rows: Tuple[Tuple[HtmlCell, ...], ...]
    column_count: int

    def has_spans(self) -> bool:
        return any(
            cell.colspan > 1 or cell.rowspan > 1
            for row in self.header_rows + self.body_rows
            for cell in row
        )


def span_grid(rows: Sequence[Sequence[HtmlCell]], column_count: int) -> List[List[Tuple[int, int]]]:
    """Slot grid of one table section, each slot holding its cell's ``(row, index)``.

    Cells are placed the way a browser places them: left to right, skipping
    slots already covered by a ``rowspan`` from an earlier row. Raises
    ``ValueError`` when a cell overflows ``column_count``, spans overlap, a
    ``rowspan`` runs past the section, or a slot is left empty.
    """

    grid: List[List[Optional[Tuple[int, int]]]] = [[None] * column_count for _ in rows]
    for r, row in enumerate(rows):
        c = 0
        for index, cell in enumerate(row):
            while c < column_count and grid[r][c] is not None:
                c += 1
            if cell.colspan < 1 or cell.rowspan < 1:
                raise ValueError(f"row {r} cell {index}: spans must be >= 1")
            if c + cell.colspan > column_count:
                raise ValueError(f"row {r} cell {index}: overflows {column_count} columns")
            if r + cell.rowspan > len(rows):
                raise ValueError(f"row {r} cell {index}: rowspan runs past the section")
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    if grid[r + dr][c + dc] is not None:
                        raise ValueError(f"row {r} cell {index}: overlaps another span")
                    grid[r + dr][c + dc] = (r, index)
            c += cell.colspan
    for r, slots in enumerate(grid):
        if any(slot is None for slot in slots):
            raise ValueError(f"row {r}: empty slot")
    return [[slot for slot in slots if slot is not None] for slots in grid]


def render_html_table(table: HtmlTable) -> str:
    """HTML for ``table``: one line per row, no blank lines, escaped text."""

    span_grid(table.header_rows, table.column_count)
    span_grid(table.body_rows, table.column_count)
    lines = ["<table>"]
    if table.header_rows:
        lines.append("<thead>")
        lines.extend(_render_row(row, "th") for row in table.header_rows)
        lines.append("</thead>")
    lines.append("<tbody>")
    lines.extend(_render_row(row, "td") for row in table.body_rows)
    lines.append("</tbody>")
    lines.append("</table>")
    return "\n".join(lines)


def _render_row(row: Sequence[HtmlCell], tag: str) -> str:
    return "<tr>" + "".join(_render_cell(cell, tag) for cell in row) + "</tr>"


def _render_cell(cell: HtmlCell, tag: str) -> str:
    attrs = ""
    if cell.colspan > 1:
        attrs += f' colspan="{cell.colspan}"'
    if cell.rowspan > 1:
        attrs += f' rowspan="{cell.rowspan}"'
    if cell.numeric:
        attrs += f' style="{RIGHT_ALIGN_STYLE}"'
    return f"<{tag}{attrs}>{html.escape(cell.text, quote=False)}</{tag}>"


# ---------------------------------------------------------------------------
# Layouts built from schema columns and row values.
# ---------------------------------------------------------------------------

ClipText = Callable[[str, int], str]
Localized = Mapping[str, str]


@dataclass(frozen=True)
class _ColumnGroup:
    """A header spanning ``keys`` (adjacent columns); ``leaf_labels`` shorten their headers."""

    label: Localized
    keys: Tuple[str, ...]
    leaf_labels: Mapping[str, Localized] = field(default_factory=dict)


_HALF_YEAR_LABELS: Dict[str, Tuple[str, str]] = {
    "ko": ("상반기", "하반기"),
    "en": ("H1", "H2"),
    "ja": ("上半期", "下半期"),
}

_FINANCIAL_GROUPS = (
    _ColumnGroup(
        label={"ko": "금액(백만원)", "en": "Amount (KRW mn)", "ja": "金額（百万円）"},
        keys=("current", "prior"),
        leaf_labels={
            "current": {"ko": "당기", "en": "Current", "ja": "当期"},
            "prior": {"ko": "전기", "en": "Prior", "ja": "前期"},
        },
    ),
    _ColumnGroup(
        label={"ko": "증감", "en": "Change", "ja": "増減"},
        keys=("change", "change_rate"),
        leaf_labels={
            "change": {"ko": "금액", "en": "Amount", "ja": "金額"},
            "change_rate": {"ko": "비율", "en": "Rate", "ja": "率"},
        },
    ),
)

_BUDGET_EXECUTION_GROUP = _ColumnGroup(
    label={"ko": "집행", "en": "Executed", "ja": "執行"},
    keys=("executed", "execution_rate"),
    leaf_labels={
        "executed": {"ko": "금액", "en": "Amount", "ja": "金額"},
        "execution_rate": {"ko": "집행률", "en": "Rate", "ja": "執行率"},
    },
)

# Without an execution-rate column the two amounts share one group.
_BUDGET_AMOUNT_GROUP = _ColumnGroup(
    label={"ko": "금액", "en": "Amount", "ja": "金額"},
    keys=("budget", "executed"),
)

_SCHEDULE_DATETIME_GROUP = _ColumnGroup(
    label={"ko": "일시", "en": "Date & Time", "ja": "日時"},
    keys=("date", "time"),
)


def build_merged_table(
    schema_name: str,
    columns: Sequence[TableColumn],
    rows: Sequence[Mapping[str, str]],
    lang: str,
    clip_text: ClipText,
) -> Optional[HtmlTable]:
    """Merged-cell layout of a schema table, or ``None`` when nothing would span.

    ``columns`` are the schema columns picked for the table and ``rows`` the
    row dicts from the schema's row builder. Layouts that need randomness
    (row-group runs, half-year splits) draw from the global ``random``,
    which ``Generator`` seeds per sample.
    """

    builder = _LAYOUTS.get(schema_name)
    if builder is None or not columns or not rows:
        return None
    table = builder(list(columns), [dict(row) for row in rows], lang, clip_text)
    if table is None or not table.has_spans():
        return None
    return table


def _statistics_layout(columns, rows, lang, clip_text) -> Optional[HtmlTable]:
    label_column, year_columns = columns[0], columns[1:]
    if not year_columns:
        return None
    # Two leaf columns per year: keep the leaf count near the requested width.
    years = year_columns[: max(1, len(year_columns) // 2)]
    first_labels = {code: labels[0] for code, labels in _HALF_YEAR_LABELS.items()}
    second_labels = {code: labels[1] for code, labels in _HALF_YEAR_LABELS.items()}
    leaf_columns: List[TableColumn] = [label_column]
    groups: List[_ColumnGroup] = []
    for year in years:
        first = TableColumn(f"{year.key}_h1", year.type, first_labels)
        second = TableColumn(f"{year.key}_h2", year.type, second_labels)
        leaf_columns.extend([first, second])
        groups.append(_ColumnGroup(label=year.header, keys=(first.key, second.key)))
        for row in rows:
            row[first.key], row[second.key] = _split_total(row[year.key])
    header = _grouped_header(leaf_columns, groups, lang, clip_text)
    return HtmlTable(header, _body_rows(leaf_columns, rows), len(leaf_columns))


def _financial_layout(columns, rows, lang, clip_text) -> Optional[HtmlTable]:
    header = _grouped_header(columns, _FINANCIAL_GROUPS, lang, clip_text)
    return HtmlTable(header, _body_rows(columns, rows), len(columns))


def _budget_layout(columns, rows, lang, clip_text) -> Optional[HtmlTable]:
    keys = {column.key for column in columns}
    group = _BUDGET_EXECUTION_GROUP if "execution_rate" in keys else _BUDGET_AMOUNT_GROUP
    header = _grouped_header(columns, (group,), lang, clip_text)
    return HtmlTable(header, _body_rows(columns, rows), len(columns))


def _schedule_layout(columns, rows, lang, clip_text) -> Optional[HtmlTable]:
    # Several events share a date; list them in date/time order.
    _share_label_in_runs(rows, "date")
    rows.sort(key=lambda row: (row["date"], row["time"]))
    header = _grouped_header(columns, (_SCHEDULE_DATETIME_GROUP,), lang, clip_text)
    body = _merge_equal_runs(_body_rows(columns, rows), _column_index(columns, "date"))
    return HtmlTable(header, body, len(columns))


def _roster_layout(columns, rows, lang, clip_text) -> Optional[HtmlTable]:
    # Group people by department, with the department as the first column.
    department = next((column for column in columns if column.key == "department"), None)
    if department is None:
        return None
    columns = [department] + [column for column in columns if column.key != "department"]
    _share_label_in_runs(rows, "department")
    # Keep each department's people together, departments in first-seen order.
    first_seen: Dict[str, int] = {}
    for row in rows:
        first_seen.setdefault(row["department"], len(first_seen))
    rows.sort(key=lambda row: first_seen[row["department"]])
    header = _grouped_header(columns, (), lang, clip_text)
    body = _merge_equal_runs(_body_rows(columns, rows), 0)
    return HtmlTable(header, body, len(columns))


def _order_layout(columns, rows, lang, clip_text) -> Optional[HtmlTable]:
    # The last row is the 합계 summary row (see _build_order_rows).
    if len(rows) < 2 or len(columns) < 3 or columns[-1].key != "amount":
        return None
    header = _grouped_header(columns, (), lang, clip_text)
    body = list(_body_rows(columns, rows[:-1]))
    total = rows[-1]
    body.append(
        (
            HtmlCell(total[columns[0].key], colspan=len(columns) - 1),
            HtmlCell(total["amount"], numeric=True),
        )
    )
    return HtmlTable(header, tuple(body), len(columns))


_LAYOUTS: Dict[str, Callable[..., Optional[HtmlTable]]] = {
    "statistics": _statistics_layout,
    "financial": _financial_layout,
    "budget": _budget_layout,
    "schedule": _schedule_layout,
    "roster": _roster_layout,
    "order": _order_layout,
}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _localized(labels: Localized, lang: str) -> str:
    return labels.get(lang) or labels.get("en") or next(iter(labels.values()), "")


def _is_numeric(column: TableColumn) -> bool:
    return column.type in NUMERIC_COLUMN_TYPES


def _column_index(columns: Sequence[TableColumn], key: str) -> int:
    return next(index for index, column in enumerate(columns) if column.key == key)


def _grouped_header(
    columns: Sequence[TableColumn],
    groups: Sequence[_ColumnGroup],
    lang: str,
    clip_text: ClipText,
) -> Tuple[Tuple[HtmlCell, ...], ...]:
    """One header row, or two when a group's columns are all present and adjacent.

    A group whose columns are not all present (an optional column was
    dropped) is not drawn; its columns keep their full schema headers.
    """

    top: List[HtmlCell] = []
    leaves: List[HtmlCell] = []
    index = 0
    while index < len(columns):
        group = next(
            (
                candidate
                for candidate in groups
                if tuple(column.key for column in columns[index : index + len(candidate.keys)])
                == candidate.keys
            ),
            None,
        )
        if group is None:
            column = columns[index]
            top.append(
                HtmlCell(clip_text(resolve_header(column, lang), 24), numeric=_is_numeric(column))
            )
            index += 1
            continue
        top.append(HtmlCell(clip_text(_localized(group.label, lang), 24), colspan=len(group.keys)))
        for column in columns[index : index + len(group.keys)]:
            label = group.leaf_labels.get(column.key)
            text = _localized(label, lang) if label else resolve_header(column, lang)
            leaves.append(HtmlCell(clip_text(text, 24), numeric=_is_numeric(column)))
        index += len(group.keys)

    if not leaves:
        return (tuple(top),)
    # Ungrouped headers span both header rows; group headers are not numeric.
    top = [cell if cell.colspan > 1 else replace(cell, rowspan=2) for cell in top]
    return (tuple(top), tuple(leaves))


def _body_rows(
    columns: Sequence[TableColumn], rows: Sequence[Mapping[str, str]]
) -> Tuple[Tuple[HtmlCell, ...], ...]:
    return tuple(
        tuple(HtmlCell(row[column.key], numeric=_is_numeric(column)) for column in columns)
        for row in rows
    )


def _share_label_in_runs(rows: List[Dict[str, str]], key: str) -> None:
    """Copy each run's first ``key`` value over the run, so runs merge into row groups.

    Runs are 1-``_MAX_ROW_GROUP`` rows long; with two or more rows at least
    one run spans two rows, so the table always has a row group.
    """

    runs: List[int] = []
    remaining = len(rows)
    while remaining > 0:
        size = random.randint(1, min(_MAX_ROW_GROUP, remaining))
        runs.append(size)
        remaining -= size
    if len(rows) >= 2 and all(size == 1 for size in runs):
        runs = [2] + runs[2:]
    start = 0
    for size in runs:
        for row in rows[start + 1 : start + size]:
            row[key] = rows[start][key]
        start += size


def _merge_equal_runs(
    body: Sequence[Sequence[HtmlCell]], column_index: int
) -> Tuple[Tuple[HtmlCell, ...], ...]:
    """Merge consecutive equal cells in ``column_index`` into one ``rowspan`` cell.

    ``body`` must not have spans yet, so list positions are grid columns.
    """

    merged: List[List[Optional[HtmlCell]]] = [list(row) for row in body]
    start = 0
    while start < len(body):
        end = start + 1
        while end < len(body) and body[end][column_index].text == body[start][column_index].text:
            end += 1
        if end - start > 1:
            merged[start][column_index] = replace(body[start][column_index], rowspan=end - start)
            for r in range(start + 1, end):
                merged[r][column_index] = None
        start = end
    return tuple(tuple(cell for cell in row if cell is not None) for row in merged)


def _split_total(total_text: str) -> Tuple[str, str]:
    """Split a formatted annual total ("12,345") into two half-year values that add up."""

    total = int(total_text.replace(",", ""))
    first = int(round(total * random.uniform(*_FIRST_HALF_SHARE)))
    return f"{first:,}", f"{total - first:,}"
