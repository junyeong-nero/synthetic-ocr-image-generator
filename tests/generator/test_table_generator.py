"""Schema-driven table generation: consistency, alignment, localization, determinism."""

import random

import pytest

from src.generator.data_provider import DataProvider
from src.generator.table_generator import TableGenerator
from src.generator.table_schemas import (
    FINANCIAL_BALANCE_SHEET_ITEMS,
    FINANCIAL_INCOME_STATEMENT_ITEMS,
)


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def _table_generator(lang: str = "ko", table_schemas=None) -> TableGenerator:
    data = DataProvider(lang=lang, mix_ratio=0.0, use_corpus=False)
    return TableGenerator(data=data, clip_text=_clip, table_schemas=table_schemas)


def _parse_table(section_markdown: str):
    lines = [line for line in section_markdown.splitlines() if line.strip() and not line.startswith("#")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    sep = [cell.strip() for cell in lines[1].strip("|").split("|")]
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]
    return header, sep, rows


def _parse_signed_int(cell: str) -> int:
    text = cell.strip()
    if text.startswith("△"):
        return -int(text[1:].replace(",", ""))
    if text.startswith("(") and text.endswith(")"):
        return -int(text[1:-1].replace(",", ""))
    if text.startswith("-"):
        return -int(text[1:].replace(",", ""))
    return int(text.replace(",", ""))


def _parse_signed_percent(cell: str) -> float:
    text = cell.strip()
    negative = False
    if text.startswith("△"):
        negative = True
        text = text[1:]
    elif text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    elif text.startswith("-"):
        negative = True
        text = text[1:]
    value = float(text.rstrip("%"))
    return -value if negative else value


# ---------------------------------------------------------------------------
# Legacy behaviour is unchanged when no table_schemas key is supplied.
# ---------------------------------------------------------------------------


def test_without_table_schemas_output_matches_plain_constructor() -> None:
    random.seed(7)
    with_none = _table_generator(table_schemas=None)
    sections_a = with_none.generate_sections(section_count=3, row_range=(2, 4), column_range=(3, 5))

    random.seed(7)
    data = DataProvider(lang="ko", mix_ratio=0.0, use_corpus=False)
    legacy = TableGenerator(data=data, clip_text=_clip)
    sections_b = legacy.generate_sections(section_count=3, row_range=(2, 4), column_range=(3, 5))

    assert sections_a == sections_b


def test_empty_table_schemas_mapping_keeps_legacy_output() -> None:
    random.seed(7)
    empty = _table_generator(table_schemas={})
    sections_a = empty.generate_sections(section_count=3, row_range=(2, 4), column_range=(3, 5))

    random.seed(7)
    legacy = _table_generator(table_schemas=None)
    sections_b = legacy.generate_sections(section_count=3, row_range=(2, 4), column_range=(3, 5))

    assert sections_a == sections_b


# ---------------------------------------------------------------------------
# Financial schema: 증감 = 당기 - 전기, 증감률 = 증감 / 전기.
# ---------------------------------------------------------------------------


def test_financial_schema_change_and_change_rate_are_consistent() -> None:
    generator = _table_generator(table_schemas={"financial": 1.0})
    for seed in range(15):
        random.seed(seed)
        sections = generator.generate_sections(section_count=1, row_range=(3, 5), column_range=(5, 5))
        header, sep, rows = _parse_table(sections[0])
        assert header == ["항목", "당기(백만원)", "전기(백만원)", "증감(백만원)", "증감률"]
        assert sep == ["---", "---:", "---:", "---:", "---:"]
        for item, current, prior, change, change_rate in rows:
            assert item.strip()
            current_v = _parse_signed_int(current)
            prior_v = _parse_signed_int(prior)
            change_v = _parse_signed_int(change)
            rate_v = _parse_signed_percent(change_rate)
            assert change_v == current_v - prior_v
            assert rate_v == pytest.approx(round(change_v / prior_v * 100, 1), abs=0.05)


def test_financial_schema_keeps_canonical_account_order() -> None:
    # Financial statement accounts have a real, top-to-bottom canonical
    # order (e.g. 매출액 before 매출총이익 before 당기순이익 on an income
    # statement); rows must be a contiguous, order-preserving run of one
    # statement, never a shuffle that could print 당기순이익 above 매출액.
    generator = _table_generator(table_schemas={"financial": 1.0})
    income_order = FINANCIAL_INCOME_STATEMENT_ITEMS["ko"]
    balance_order = FINANCIAL_BALANCE_SHEET_ITEMS["ko"]

    for seed in range(20):
        random.seed(seed)
        sections = generator.generate_sections(section_count=1, row_range=(3, 6), column_range=(3, 3))
        header, _sep, rows = _parse_table(sections[0])
        items = [row[0] for row in rows]

        reference = income_order if items[0] in income_order else balance_order
        indices = [reference.index(item) for item in items]

        # every item belongs to the same statement as the first row...
        assert all(item in reference for item in items)
        # ...appears in ascending canonical order...
        assert indices == sorted(indices)
        # ...and is one contiguous run (no skipped-then-revisited accounts).
        assert indices == list(range(indices[0], indices[0] + len(indices)))


def test_financial_schema_drops_optional_columns_to_fit_narrow_range() -> None:
    generator = _table_generator(table_schemas={"financial": 1.0})
    random.seed(1)
    sections = generator.generate_sections(section_count=5, row_range=(2, 2), column_range=(3, 3))
    for section in sections:
        header, sep, rows = _parse_table(section)
        assert header == ["항목", "당기(백만원)", "전기(백만원)"]
        assert sep == ["---", "---:", "---:"]


# ---------------------------------------------------------------------------
# Budget schema: 집행률 = 집행액 / 예산액.
# ---------------------------------------------------------------------------


def test_budget_schema_execution_rate_matches_ratio() -> None:
    generator = _table_generator(table_schemas={"budget": 1.0})
    for seed in range(15):
        random.seed(seed)
        sections = generator.generate_sections(section_count=1, row_range=(3, 3), column_range=(4, 4))
        header, sep, rows = _parse_table(sections[0])
        assert header == ["구분", "예산액", "집행액", "집행률"]
        assert sep == ["---", "---:", "---:", "---:"]
        for category, budget, executed, rate in rows:
            assert category.strip()
            budget_v = int(budget.replace(",", ""))
            executed_v = int(executed.replace(",", ""))
            rate_v = float(rate.rstrip("%"))
            assert rate_v == pytest.approx(round(executed_v / budget_v * 100, 1), abs=0.05)


# ---------------------------------------------------------------------------
# Order schema: 금액 = 수량 x 단가, and a 합계 row sums quantity/amount.
# ---------------------------------------------------------------------------


def test_order_schema_amount_equals_quantity_times_unit_price_with_total_row() -> None:
    generator = _table_generator(table_schemas={"order": 1.0})
    for seed in range(15):
        random.seed(seed)
        sections = generator.generate_sections(section_count=1, row_range=(4, 4), column_range=(5, 5))
        header, sep, rows = _parse_table(sections[0])
        assert header == ["품목", "규격", "수량", "단가", "금액"]
        assert sep == ["---", "---", "---:", "---:", "---:"]

        *data_rows, total_row = rows
        total_qty = 0
        total_amount = 0
        for item, spec, qty, price, amount in data_rows:
            assert item.strip()
            qty_v = int(qty.replace(",", ""))
            price_v = int(price.replace(",", ""))
            amount_v = int(amount.replace(",", ""))
            assert amount_v == qty_v * price_v
            total_qty += qty_v
            total_amount += amount_v

        assert total_row[0] == "합계"
        assert int(total_row[2].replace(",", "")) == total_qty
        assert int(total_row[4].replace(",", "")) == total_amount


# ---------------------------------------------------------------------------
# Text columns (schedule, roster) are not right-aligned; headers localize.
# ---------------------------------------------------------------------------


def test_schedule_schema_headers_and_alignment() -> None:
    generator = _table_generator(table_schemas={"schedule": 1.0})
    random.seed(6)
    sections = generator.generate_sections(section_count=1, row_range=(2, 2), column_range=(5, 5))
    header, sep, rows = _parse_table(sections[0])
    assert header == ["일자", "시간", "내용", "장소", "담당"]
    assert sep == ["---", "---", "---", "---", "---"]
    for row in rows:
        assert all(cell.strip() for cell in row)


def test_roster_schema_headers_localize_for_english() -> None:
    generator = _table_generator(lang="en", table_schemas={"roster": 1.0})
    random.seed(2)
    sections = generator.generate_sections(section_count=1, row_range=(2, 2), column_range=(4, 4))
    header, sep, _rows = _parse_table(sections[0])
    assert header == ["Name", "Department", "Position", "Phone"]
    assert sep == ["---", "---", "---", "---"]


def test_roster_schema_headers_localize_for_japanese() -> None:
    generator = _table_generator(lang="ja", table_schemas={"roster": 1.0})
    random.seed(2)
    sections = generator.generate_sections(section_count=1, row_range=(2, 2), column_range=(4, 4))
    header, _sep, _rows = _parse_table(sections[0])
    assert header == ["氏名", "所属", "職位", "連絡先"]


# ---------------------------------------------------------------------------
# Statistics schema: dynamic year columns, category rows, right alignment.
# ---------------------------------------------------------------------------


def test_statistics_schema_builds_year_columns_matching_column_range() -> None:
    generator = _table_generator(table_schemas={"statistics": 1.0})
    random.seed(11)
    sections = generator.generate_sections(section_count=1, row_range=(3, 3), column_range=(4, 4))
    header, sep, rows = _parse_table(sections[0])

    assert header[0] == "구분"
    assert len(header) == 4
    # remaining headers are 4-digit years with a ko "년" suffix
    for year_header in header[1:]:
        assert year_header.endswith("년")
        assert year_header[:-1].isdigit()
    assert sep == ["---", "---:", "---:", "---:"]

    for row in rows:
        category, *year_cells = row
        assert category.strip()
        for cell in year_cells:
            assert int(cell.replace(",", "")) >= 0


def test_statistics_schema_never_exceeds_the_narrowest_column_range() -> None:
    generator = _table_generator(table_schemas={"statistics": 1.0})
    random.seed(11)
    sections = generator.generate_sections(section_count=5, row_range=(2, 2), column_range=(2, 2))
    for section in sections:
        header, _sep, _rows = _parse_table(section)
        assert header[0] == "구분"
        assert len(header) == 2


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


def test_schema_table_generation_is_deterministic_with_same_seed() -> None:
    schemas = {"financial": 1.0, "order": 1.0, "schedule": 1.0, "roster": 1.0, "budget": 1.0}

    random.seed(42)
    sections_a = _table_generator(table_schemas=schemas).generate_sections(
        section_count=8, row_range=(2, 5), column_range=(3, 5)
    )

    random.seed(42)
    sections_b = _table_generator(table_schemas=schemas).generate_sections(
        section_count=8, row_range=(2, 5), column_range=(3, 5)
    )

    assert sections_a == sections_b
