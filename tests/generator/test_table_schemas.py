"""Coherent table schema definitions: headers, types, column selection."""

from src.generator.table_schemas import (
    NUMERIC_COLUMN_TYPES,
    SCHEMAS,
    required_column_count,
    resolve_header,
    select_schema_columns,
)


def test_schema_registry_has_the_schemas_from_the_brief() -> None:
    assert set(SCHEMAS) >= {
        "financial",
        "budget",
        "schedule",
        "roster",
        "order",
        "statistics",
    }


def test_financial_schema_column_order_and_types() -> None:
    schema = SCHEMAS["financial"]
    assert [c.key for c in schema.columns] == [
        "item",
        "current",
        "prior",
        "change",
        "change_rate",
    ]
    by_key = {c.key: c for c in schema.columns}
    assert by_key["item"].type == "label"
    assert by_key["current"].type == "amount"
    assert by_key["prior"].type == "amount"
    assert by_key["change"].type == "change"
    assert by_key["change_rate"].type == "percent"
    # only item is required to draw the smallest possible table
    assert required_column_count(schema) == 3


def test_budget_schedule_roster_order_column_order_matches_brief() -> None:
    assert [c.key for c in SCHEMAS["budget"].columns] == [
        "category",
        "budget",
        "executed",
        "execution_rate",
    ]
    assert [c.key for c in SCHEMAS["schedule"].columns] == [
        "date",
        "time",
        "content",
        "place",
        "owner",
    ]
    assert [c.key for c in SCHEMAS["roster"].columns] == [
        "name",
        "department",
        "position",
        "phone",
    ]
    assert [c.key for c in SCHEMAS["order"].columns] == [
        "item",
        "spec",
        "quantity",
        "unit_price",
        "amount",
    ]


def test_order_schema_requires_a_summary_row_and_at_least_two_rows() -> None:
    schema = SCHEMAS["order"]
    assert schema.has_summary_row is True
    assert schema.min_rows == 2


def test_numeric_column_types_are_right_aligned_candidates() -> None:
    schema = SCHEMAS["order"]
    numeric_keys = {c.key for c in schema.columns if c.type in NUMERIC_COLUMN_TYPES}
    assert numeric_keys == {"quantity", "unit_price", "amount"}
    text_keys = {c.key for c in schema.columns if c.type not in NUMERIC_COLUMN_TYPES}
    assert text_keys == {"item", "spec"}


def test_resolve_header_localizes_ko_en_ja_and_falls_back_to_en() -> None:
    item = {c.key: c for c in SCHEMAS["financial"].columns}["item"]
    assert resolve_header(item, "ko") == "항목"
    assert resolve_header(item, "en") == "Item"
    assert resolve_header(item, "ja") == "項目"
    # a language outside ko/en/ja falls back to the English header
    assert resolve_header(item, "fr") == resolve_header(item, "en")


def test_financial_current_header_carries_the_unit() -> None:
    current = {c.key: c for c in SCHEMAS["financial"].columns}["current"]
    assert resolve_header(current, "ko") == "당기(백만원)"


def test_select_schema_columns_drops_optional_columns_from_the_right_to_fit() -> None:
    schema = SCHEMAS["financial"]
    assert [c.key for c in select_schema_columns(schema, 5)] == [
        "item",
        "current",
        "prior",
        "change",
        "change_rate",
    ]
    assert [c.key for c in select_schema_columns(schema, 4)] == [
        "item",
        "current",
        "prior",
        "change",
    ]
    assert [c.key for c in select_schema_columns(schema, 3)] == [
        "item",
        "current",
        "prior",
    ]
    # never drops below the required columns even if asked for fewer
    assert [c.key for c in select_schema_columns(schema, 1)] == [
        "item",
        "current",
        "prior",
    ]


def test_select_schema_columns_never_exceeds_the_declared_columns() -> None:
    schema = SCHEMAS["budget"]
    assert len(select_schema_columns(schema, 99)) == len(schema.columns)
