"""Merged-cell HTML tables: well-formed HTML, span grid consistency, schema layouts."""

import random
import re
from html.parser import HTMLParser

import pytest

from src.generator.data_provider import DataProvider
from src.generator.html_table import HtmlCell, HtmlTable, render_html_table, span_grid
from src.generator.table_generator import TableGenerator
from src.generator.table_schemas import SCHEMAS

_ALLOWED_TAGS = {"table", "thead", "tbody", "tr", "th", "td"}
_RIGHT = "text-align: right;"


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def _generator(lang="ko", table_schemas=None, merged_table_ratio=None) -> TableGenerator:
    data = DataProvider(lang=lang, mix_ratio=0.0, use_corpus=False)
    kwargs = {} if merged_table_ratio is None else {"merged_table_ratio": merged_table_ratio}
    return TableGenerator(data=data, clip_text=_clip, table_schemas=table_schemas, **kwargs)


def _merged_html(schema: str, seed: int, lang: str = "ko", rows=(2, 6), cols=(3, 5)) -> str:
    random.seed(seed)
    generator = _generator(lang, {schema: 1.0}, merged_table_ratio=1.0)
    section = generator.generate_sections(section_count=1, row_range=rows, column_range=cols)[0]
    heading, _, body = section.partition("\n\n")
    assert heading.startswith("## ")
    return body


class _TableParser(HTMLParser):
    """Independent parser: checks tag balance and collects cells per section."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.sections: dict[str, list[list[dict]]] = {"thead": [], "tbody": []}
        self._section = None
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag not in _ALLOWED_TAGS:
            self.errors.append(f"unexpected <{tag}>")
        self.stack.append(tag)
        attrs = dict(attrs)
        if tag in ("thead", "tbody"):
            self._section = tag
        elif tag == "tr":
            self._row = []
            self.sections[self._section].append(self._row)
        elif tag in ("th", "td"):
            self._cell = {
                "tag": tag,
                "text": "",
                "colspan": int(attrs.get("colspan", 1)),
                "rowspan": int(attrs.get("rowspan", 1)),
                "style": attrs.get("style"),
            }
            self._row.append(self._cell)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}>")
        else:
            self.stack.pop()
        if tag in ("th", "td"):
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"] += data
        elif data.strip():
            self.errors.append(f"stray text {data!r}")


def _parse(table_html: str) -> _TableParser:
    parser = _TableParser()
    parser.feed(table_html)
    parser.close()
    assert parser.errors == []
    assert parser.stack == []
    return parser


def _grid_width(rows: list[list[dict]]) -> int:
    """Place cells like a browser does; assert a full rectangle with no overlap."""
    occupied: dict[tuple[int, int], dict] = {}
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in occupied:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    assert (r + dr, c + dc) not in occupied, "overlapping spans"
                    occupied[(r + dr, c + dc)] = cell
            c += cell["colspan"]
    assert all(r < len(rows) for r, _ in occupied), "rowspan runs past its section"
    widths = {sum(1 for (rr, _) in occupied if rr == r) for r in range(len(rows))}
    assert len(widths) == 1, f"ragged rows: {widths}"
    width = widths.pop()
    assert all((r, c) in occupied for r in range(len(rows)) for c in range(width)), "holes"
    return width


def _expand(rows: list[list[dict]]) -> list[list[dict]]:
    """Grid of cells (a spanned cell repeated in every slot it covers)."""
    occupied: dict[tuple[int, int], dict] = {}
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in occupied:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    occupied[(r + dr, c + dc)] = cell
            c += cell["colspan"]
    width = max(c for _, c in occupied) + 1
    return [[occupied[(r, c)] for c in range(width)] for r in range(len(rows))]


# ---------------------------------------------------------------------------
# Well-formed HTML and a consistent span grid for every schema.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["ko", "en", "ja"])
@pytest.mark.parametrize("schema", sorted(SCHEMAS))
def test_every_schema_emits_one_chunk_well_formed_table_with_spans(schema: str, lang: str) -> None:
    for seed in range(12):
        table_html = _merged_html(schema, seed, lang)

        assert table_html.startswith("<table>") and table_html.endswith("</table>")
        assert "\n\n" not in table_html  # one blank-line chunk
        assert all(line.startswith("<") for line in table_html.splitlines())
        parsed = _parse(table_html)
        header, body = parsed.sections["thead"], parsed.sections["tbody"]
        assert header and body
        assert _grid_width(header) == _grid_width(body)
        cells = [cell for row in header + body for cell in row]
        assert any(cell["colspan"] > 1 or cell["rowspan"] > 1 for cell in cells)
        assert all(cell["tag"] == "th" for row in header for cell in row)
        assert all(cell["tag"] == "td" for row in body for cell in row)
        assert all(cell["text"].strip() for cell in cells)


def test_statistics_groups_half_years_under_each_year() -> None:
    table_html = _merged_html("statistics", seed=3, cols=(5, 5))
    header = _parse(table_html).sections["thead"]

    assert len(header) == 2
    first, second = header
    assert first[0]["text"] == "구분" and first[0]["rowspan"] == 2
    years = first[1:]
    assert years and all(re.fullmatch(r"\d{4}년", cell["text"]) for cell in years)
    assert all(cell["colspan"] == 2 for cell in years)
    assert [cell["text"] for cell in second] == ["상반기", "하반기"] * len(years)
    year_numbers = [int(cell["text"][:4]) for cell in years]
    assert year_numbers == list(range(year_numbers[0], year_numbers[0] + len(years)))


def test_statistics_english_half_year_labels() -> None:
    header = _parse(_merged_html("statistics", seed=1, lang="en")).sections["thead"]
    assert {cell["text"] for cell in header[1]} == {"H1", "H2"}


def test_financial_groups_amounts_and_changes() -> None:
    texts = set()
    for seed in range(30):
        header = _parse(_merged_html("financial", seed, cols=(5, 5))).sections["thead"]
        texts |= {(cell["text"], cell["colspan"]) for cell in header[0]}
        assert [cell["text"] for cell in header[1]][:2] == ["당기", "전기"]
    assert ("금액(백만원)", 2) in texts
    assert ("증감", 2) in texts


def test_financial_single_change_column_keeps_its_full_header() -> None:
    header = _parse(_merged_html("financial", seed=0, cols=(4, 4))).sections["thead"]
    top = [(cell["text"], cell["colspan"], cell["rowspan"]) for cell in header[0]]
    assert top == [("항목", 1, 2), ("금액(백만원)", 2, 1), ("증감(백만원)", 1, 2)]


def test_budget_groups_execution_amount_and_rate() -> None:
    header = _parse(_merged_html("budget", seed=2, cols=(4, 4))).sections["thead"]
    top = [(cell["text"], cell["colspan"], cell["rowspan"]) for cell in header[0]]
    assert top == [("구분", 1, 2), ("예산액", 1, 2), ("집행", 2, 1)]
    assert [cell["text"] for cell in header[1]] == ["금액", "집행률"]


def test_schedule_merges_repeated_dates_into_rowspans_in_date_order() -> None:
    saw_rowspan = False
    for seed in range(20):
        body = _parse(_merged_html("schedule", seed, rows=(3, 6))).sections["tbody"]
        dates = [row[0]["text"] for row in _expand(body)]
        assert dates == sorted(dates)
        date_cells = [row[0] for row in body if re.fullmatch(r"\d{4}-\d{2}-\d{2}", row[0]["text"])]
        # every date appears once as a (possibly spanning) cell
        assert len({cell["text"] for cell in date_cells}) == len(date_cells)
        saw_rowspan |= any(cell["rowspan"] > 1 for cell in date_cells)
    assert saw_rowspan


def test_schedule_groups_date_and_time_under_one_header() -> None:
    header = _parse(_merged_html("schedule", seed=4, cols=(5, 5))).sections["thead"]
    assert (header[0][0]["text"], header[0][0]["colspan"]) == ("일시", 2)
    assert [cell["text"] for cell in header[1]] == ["일자", "시간"]


def test_roster_merges_department_rows_and_puts_department_first() -> None:
    for seed in range(10):
        parsed = _parse(_merged_html("roster", seed, rows=(3, 6)))
        assert parsed.sections["thead"][0][0]["text"] == "소속"
        body = parsed.sections["tbody"]
        assert any(row[0]["rowspan"] > 1 for row in body)


def test_order_total_row_label_spans_to_amount_column() -> None:
    for seed in range(10):
        body = _parse(_merged_html("order", seed, rows=(3, 6))).sections["tbody"]
        width = _grid_width(body)
        total = body[-1]
        assert total[0]["text"] == "합계" and total[0]["colspan"] == width - 1
        amounts = [int(row[-1]["text"].replace(",", "")) for row in body[:-1]]
        assert int(total[-1]["text"].replace(",", "")) == sum(amounts)


def test_numeric_cells_are_right_aligned_like_markdown_pipe_tables() -> None:
    parsed = _parse(_merged_html("statistics", seed=5))
    for row in parsed.sections["tbody"]:
        assert row[0]["style"] is None  # category label
        assert all(cell["style"] == _RIGHT for cell in row[1:])
        assert all(re.fullmatch(r"[\d,]+", cell["text"]) for cell in row[1:])
    assert all(cell["style"] == _RIGHT for cell in parsed.sections["thead"][1])
    assert all(cell["style"] is None for cell in parsed.sections["thead"][0])


def test_merged_tables_are_deterministic_for_a_seed() -> None:
    for schema in SCHEMAS:
        assert _merged_html(schema, seed=9) == _merged_html(schema, seed=9)


# ---------------------------------------------------------------------------
# Ratio gating and legacy behaviour.
# ---------------------------------------------------------------------------


def test_zero_ratio_keeps_schema_tables_byte_identical() -> None:
    schemas = {name: 1.0 for name in SCHEMAS}
    random.seed(21)
    plain = _generator("ko", schemas).generate_sections(section_count=8, row_range=(2, 5), column_range=(3, 5))
    random.seed(21)
    zero = _generator("ko", schemas, merged_table_ratio=0.0).generate_sections(
        section_count=8, row_range=(2, 5), column_range=(3, 5)
    )
    assert plain == zero
    assert not any("<table" in section for section in plain)


def test_ratio_without_table_schemas_keeps_legacy_tables() -> None:
    random.seed(4)
    legacy = _generator("ko").generate_sections(section_count=4, row_range=(2, 4), column_range=(3, 5))
    random.seed(4)
    merged = _generator("ko", merged_table_ratio=1.0).generate_sections(
        section_count=4, row_range=(2, 4), column_range=(3, 5)
    )
    assert legacy == merged


def test_ratio_sets_share_of_html_tables() -> None:
    random.seed(0)
    generator = _generator("ko", {"financial": 1.0, "order": 1.0}, merged_table_ratio=0.3)
    sections = generator.generate_sections(section_count=600, row_range=(2, 5), column_range=(3, 5))
    share = sum("<table>" in section for section in sections) / len(sections)
    assert 0.24 <= share <= 0.36


def test_layout_without_any_span_falls_back_to_pipe_table() -> None:
    # A one-row roster has no department run to merge.
    random.seed(0)
    generator = _generator("ko", {"roster": 1.0}, merged_table_ratio=1.0)
    section = generator.generate_sections(section_count=1, row_range=(1, 1), column_range=(3, 3))[0]
    assert "<table" not in section
    assert "| 성명 |" in section


# ---------------------------------------------------------------------------
# render_html_table / span_grid.
# ---------------------------------------------------------------------------


def test_render_escapes_cell_text() -> None:
    table = HtmlTable(
        header_rows=((HtmlCell("A & B"), HtmlCell("<C>")),),
        body_rows=((HtmlCell("x"), HtmlCell("1", numeric=True)),),
        column_count=2,
    )
    table_html = render_html_table(table)
    assert "<th>A &amp; B</th><th>&lt;C&gt;</th>" in table_html
    assert f'<td style="{_RIGHT}">1</td>' in table_html


def test_render_writes_span_attributes() -> None:
    table = HtmlTable(
        header_rows=(
            (HtmlCell("k", rowspan=2), HtmlCell("g", colspan=2)),
            (HtmlCell("a"), HtmlCell("b")),
        ),
        body_rows=((HtmlCell("x"), HtmlCell("1"), HtmlCell("2")),),
        column_count=3,
    )
    table_html = render_html_table(table)
    assert '<tr><th rowspan="2">k</th><th colspan="2">g</th></tr>' in table_html
    assert _grid_width(_parse(table_html).sections["thead"]) == 3


@pytest.mark.parametrize(
    "rows",
    [
        ((HtmlCell("a", colspan=3), HtmlCell("b")),),  # overflows the width
        ((HtmlCell("a", rowspan=2), HtmlCell("b")), (HtmlCell("c"), HtmlCell("d"))),  # overlap -> overflow
        ((HtmlCell("a"), HtmlCell("b")), (HtmlCell("c"),)),  # hole
        ((HtmlCell("a", rowspan=3), HtmlCell("b")), (HtmlCell("c"),)),  # runs past section
    ],
)
def test_span_grid_rejects_inconsistent_rows(rows) -> None:
    with pytest.raises(ValueError):
        span_grid(rows, 2)


def test_span_grid_maps_every_slot_to_its_origin_cell() -> None:
    rows = ((HtmlCell("a", rowspan=2), HtmlCell("b")), (HtmlCell("c"),))
    assert span_grid(rows, 2) == [[(0, 0), (0, 1)], [(0, 0), (1, 0)]]
