"""Table section generator for markdown documents."""

import random
from typing import Callable, List, Mapping, Optional, Tuple

from src.generator.data_provider import DataProvider
from src.generator.html_table import build_merged_table, render_html_table
from src.generator.table_schemas import (
    NUMERIC_COLUMN_TYPES,
    SCHEMAS,
    TableColumn,
    required_column_count,
    resolve_curated_lang,
    resolve_header,
    select_schema_columns,
    summary_row_label,
    BUDGET_CATEGORIES,
    FINANCIAL_STATEMENTS,
    ORDER_SPECS,
    PERSON_NAMES,
    PLACE_NAMES,
    STATISTICS_CATEGORIES,
)

# Start-year range for the `statistics` schema's dynamic year columns. Kept
# as a fixed prior (not wall-clock time) so output stays deterministic from
# the seed alone.
_STATISTICS_START_YEAR_RANGE = (2018, 2023)
_STATISTICS_MAX_YEAR_COLUMNS = 6


class TableGenerator:
    """Builds markdown table sections."""

    def __init__(
        self,
        *,
        data: DataProvider,
        clip_text: Callable[[str, int], str],
        table_schemas: Optional[Mapping[str, float]] = None,
        merged_table_ratio: float = 0.0,
    ) -> None:
        self.data = data
        self.clip_text = clip_text
        self.table_schemas = {
            str(name): float(weight)
            for name, weight in (table_schemas or {}).items()
            if name in SCHEMAS and float(weight) > 0
        }
        # Share of schema tables written as a merged-cell HTML <table> (see
        # html_table.py). Only schema tables have a merged layout, so this
        # has no effect without `table_schemas`; 0 draws no random numbers.
        self.merged_table_ratio = min(1.0, max(0.0, float(merged_table_ratio or 0.0)))

    # ------------------------------------------------------------------
    # Legacy (no schema) generation - unchanged behaviour.
    # ------------------------------------------------------------------

    def _resolve_headers(self, column_count: int) -> List[str]:
        template_name = random.choice(["invoice", "schedule", "product", "contact"])
        base_headers = list(self.data.headers(template_name, count=column_count))
        if not base_headers:
            base_headers = [str(idx + 1) for idx in range(column_count)]

        headers: List[str] = []
        for idx in range(column_count):
            if idx < len(base_headers):
                headers.append(self.clip_text(str(base_headers[idx]), 24))
            else:
                headers.append(str(idx + 1))
        return headers

    def _build_cell_value(self, column_index: int, column_count: int) -> str:
        if column_index == 0:
            return self.clip_text(self.data.product_name(), 32)
        if column_index == 1:
            return str(self.data.quantity())
        if column_index == 2 and column_count >= 4:
            return self.data.format_currency(self.data.random_price())
        return self.clip_text(self.data.feature(), 36)

    def _generate_legacy_section(self, row_count: int, column_count: int) -> str:
        headers = self._resolve_headers(column_count)

        lines: List[str] = [f"## {self.clip_text(self.data.title(), 96)}", ""]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for _ in range(row_count):
            row_values = [
                self._build_cell_value(column_index=idx, column_count=column_count)
                for idx in range(column_count)
            ]
            lines.append("| " + " | ".join(row_values) + " |")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Schema-driven generation.
    # ------------------------------------------------------------------

    def _compatible_schema_names(self, row_range: Tuple[int, int], column_range: Tuple[int, int]) -> List[str]:
        """Schema names whose required rows/columns fit within the blueprint's ranges."""

        _row_min, row_max = row_range
        _col_min, col_max = column_range
        names: List[str] = []
        for name in self.table_schemas:
            schema = SCHEMAS[name]
            if schema.min_rows > row_max:
                continue
            if name == "statistics":
                # category column + at least one year column.
                if col_max < 2:
                    continue
            elif required_column_count(schema) > col_max:
                continue
            names.append(name)
        return names

    def _pick_schema_name(self, row_range: Tuple[int, int], column_range: Tuple[int, int]) -> Optional[str]:
        candidates = self._compatible_schema_names(row_range, column_range)
        if not candidates:
            return None
        weights = [self.table_schemas[name] for name in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def _generate_schema_section(self, row_range: Tuple[int, int], column_range: Tuple[int, int]) -> str:
        row_min, row_max = row_range
        col_min, col_max = column_range
        schema_name = self._pick_schema_name(row_range, column_range)
        if schema_name is None:
            row_count = random.randint(row_min, row_max)
            column_count = random.randint(col_min, col_max)
            return self._generate_legacy_section(row_count, column_count)

        lang = resolve_curated_lang(self.data.base_lang)
        column_count = random.randint(max(col_min, 1), col_max)

        if schema_name == "statistics":
            columns = self._build_statistics_columns(column_count)
        else:
            schema = SCHEMAS[schema_name]
            columns = select_schema_columns(schema, column_count)

        row_count = random.randint(row_min, row_max)
        row_count = max(row_count, SCHEMAS[schema_name].min_rows)

        row_dicts = self._build_row_dicts(schema_name, row_count, lang)
        if self.merged_table_ratio > 0 and random.random() < self.merged_table_ratio:
            merged = build_merged_table(schema_name, columns, row_dicts, lang, self.clip_text)
            if merged is not None:
                title = self.clip_text(self.data.title(), 96)
                return f"## {title}\n\n{render_html_table(merged)}"
        rows = _project_rows(row_dicts, columns)

        headers = [self.clip_text(resolve_header(column, lang), 24) for column in columns]
        separators = [
            "---:" if column.type in NUMERIC_COLUMN_TYPES else "---" for column in columns
        ]

        lines: List[str] = [f"## {self.clip_text(self.data.title(), 96)}", ""]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(separators) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines).strip()

    def _build_statistics_columns(self, column_count: int) -> List[TableColumn]:
        year_count = max(1, min(column_count - 1, _STATISTICS_MAX_YEAR_COLUMNS))
        start_year = random.randint(*_STATISTICS_START_YEAR_RANGE)
        columns: List[TableColumn] = [SCHEMAS["statistics"].columns[0]]
        for offset in range(year_count):
            year = start_year + offset
            columns.append(
                TableColumn(
                    key=f"year_{offset}",
                    type="amount",
                    header={"ko": f"{year}년", "en": str(year), "ja": f"{year}年"},
                )
            )
        return columns

    def _build_row_dicts(self, schema_name: str, row_count: int, lang: str) -> List[dict]:
        """Rows as ``{column key: cell text}`` dicts, every schema key filled."""
        builder = getattr(self, f"_build_{schema_name}_rows")
        return builder(row_count, lang)

    # -- per-schema row builders (values in schema-declared, unfiltered order) --

    def _build_financial_rows(self, row_count: int, lang: str) -> List[dict]:
        statement = random.choice(FINANCIAL_STATEMENTS)
        items = _pick_contiguous_run(statement[lang], row_count)
        rows: List[dict] = []
        for item in items:
            prior = random.randint(100, 50000)
            if random.random() < 0.08:
                prior = -prior
            delta_ratio = random.uniform(-0.35, 0.45)
            current = prior + int(round(prior * delta_ratio)) + random.randint(-50, 50)
            change = current - prior
            change_rate = round((change / prior) * 100, 1)
            rows.append(
                {
                    "item": self.clip_text(item, 32),
                    "current": _format_financial_amount(current, lang),
                    "prior": _format_financial_amount(prior, lang),
                    "change": _format_financial_amount(change, lang),
                    "change_rate": _format_financial_percent(change_rate, lang),
                }
            )
        return rows

    def _build_budget_rows(self, row_count: int, lang: str) -> List[dict]:
        categories = _sample_unique_cycle(BUDGET_CATEGORIES[lang], row_count)
        rows: List[dict] = []
        for category in categories:
            budget = random.randrange(1_000_000, 300_000_001, 10_000)
            execution_ratio = random.uniform(0.4, 1.05)
            executed = min(budget, int(round(budget * execution_ratio / 1000)) * 1000)
            rate = round((executed / budget) * 100, 1)
            rows.append(
                {
                    "category": self.clip_text(category, 32),
                    "budget": f"{budget:,}",
                    "executed": f"{executed:,}",
                    "execution_rate": f"{rate:.1f}%",
                }
            )
        return rows

    def _build_schedule_rows(self, row_count: int, lang: str) -> List[dict]:
        rows: List[dict] = []
        for _ in range(row_count):
            place = random.choice(PLACE_NAMES[lang])
            rows.append(
                {
                    "date": _random_date(),
                    "time": _random_time(),
                    "content": self.clip_text(self.data.feature(), 40),
                    "place": self.clip_text(place, 24),
                    "owner": self.clip_text(self.data.department(), 24),
                }
            )
        return rows

    def _build_roster_rows(self, row_count: int, lang: str) -> List[dict]:
        names = self._person_names(lang, row_count)
        rows: List[dict] = []
        for name in names:
            rows.append(
                {
                    "name": self.clip_text(name, 24),
                    "department": self.clip_text(self.data.department(), 24),
                    "position": self.clip_text(self.data.position(), 24),
                    "phone": _random_phone(lang),
                }
            )
        return rows

    def _person_names(self, lang: str, count: int) -> List[str]:
        # `DataProvider.name()` falls back to Faker when no `person_names`
        # corpus is loaded. Faker is seeded per sample by
        # `Generator._seed_for_sample`, but `TableGenerator` is also used
        # without going through `Generator` (tests, direct use), so tables
        # draw names/dates/phones from `random` to stay reproducible from
        # `random.seed()` alone (see the note above `PERSON_NAMES`); only
        # use `DataProvider.name()` when a corpus is actually present,
        # otherwise cycle the curated names so a small roster doesn't
        # repeat a name unnecessarily.
        if self.data.has_corpus("person_names"):
            return [self.data.name() for _ in range(count)]
        return _sample_unique_cycle(PERSON_NAMES[lang], count)

    def _build_order_rows(self, row_count: int, lang: str) -> List[dict]:
        data_row_count = max(1, row_count - 1)
        rows: List[dict] = []
        total_qty = 0
        total_amount = 0
        for _ in range(data_row_count):
            quantity = self.data.quantity(min_val=1, max_val=50)
            unit_price = self.data.random_price(min_val=500, max_val=200000, step=100)
            amount = quantity * unit_price
            total_qty += quantity
            total_amount += amount
            rows.append(
                {
                    "item": self.clip_text(self.data.product_name(), 32),
                    "spec": self.clip_text(random.choice(ORDER_SPECS[lang]), 16),
                    "quantity": f"{quantity:,}",
                    "unit_price": f"{unit_price:,}",
                    "amount": f"{amount:,}",
                }
            )
        rows.append(
            {
                "item": summary_row_label(lang),
                "spec": "-",
                "quantity": f"{total_qty:,}",
                "unit_price": "-",
                "amount": f"{total_amount:,}",
            }
        )
        return rows

    def _build_statistics_rows(self, row_count: int, lang: str) -> List[dict]:
        # Always fill every possible year_0.._STATISTICS_MAX_YEAR_COLUMNS-1
        # key; _project_rows() (and the merged layout) only reads back the
        # keys the selected columns actually need, so the exact year count
        # used elsewhere doesn't need to be threaded through here.
        categories = _sample_unique_cycle(STATISTICS_CATEGORIES[lang], row_count)
        rows: List[dict] = []
        for category in categories:
            row = {"category": self.clip_text(category, 24)}
            value = random.randint(500, 80000)
            for offset in range(_STATISTICS_MAX_YEAR_COLUMNS):
                row[f"year_{offset}"] = f"{value:,}"
                walk = int(value * random.uniform(-0.25, 0.25)) + random.randint(-50, 50)
                value = max(0, value + walk)
            rows.append(row)
        return rows

    # ------------------------------------------------------------------

    def generate_sections(
        self,
        *,
        section_count: int,
        row_range: Tuple[int, int],
        column_range: Tuple[int, int],
    ) -> List[str]:
        sections: List[str] = []

        for _ in range(max(0, section_count)):
            if self.table_schemas:
                sections.append(self._generate_schema_section(row_range, column_range))
            else:
                row_count = random.randint(*row_range)
                column_count = random.randint(*column_range)
                sections.append(self._generate_legacy_section(row_count, column_count))

        return sections


def _project_rows(rows_by_key: List[dict], columns: List[TableColumn]) -> List[List[str]]:
    column_keys = [column.key for column in columns]
    return [[row[key] for key in column_keys] for row in rows_by_key]


def _sample_unique_cycle(pool: List[str], count: int) -> List[str]:
    """``count`` values drawn from ``pool`` without immediate repeats.

    Shuffles the pool and cycles through it, reshuffling on wrap, so labels
    stay unique within a table as long as possible while remaining seed
    deterministic.
    """

    if not pool:
        return [""] * count
    shuffled = list(pool)
    random.shuffle(shuffled)
    result: List[str] = []
    while len(result) < count:
        needed = count - len(result)
        if needed >= len(shuffled):
            result.extend(shuffled)
            random.shuffle(shuffled)
        else:
            result.extend(shuffled[:needed])
    return result[:count]


def _pick_contiguous_run(items: List[str], count: int) -> List[str]:
    """``count`` consecutive items from ``items``, keeping their order.

    Financial statement accounts have a canonical top-to-bottom order (e.g.
    매출액 before 매출총이익 before 당기순이익), so rows are a contiguous
    slice of one statement rather than a shuffled sample: 당기순이익 never
    ends up printed above 매출액.
    """

    if not items:
        return [""] * count
    if count >= len(items):
        return list(items)
    start = random.randint(0, len(items) - count)
    return items[start : start + count]


def _random_phone(lang: str) -> str:
    if lang == "ko":
        return f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
    if lang == "ja":
        return f"090-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
    return f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"


def _random_date() -> str:
    year = random.randint(2019, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _random_time() -> str:
    hour = random.randint(8, 18)
    minute = random.choice([0, 10, 15, 20, 30, 40, 45, 50])
    return f"{hour:02d}:{minute:02d}"


def _format_financial_amount(value: int, lang: str) -> str:
    if value < 0:
        magnitude = f"{abs(value):,}"
        if lang == "ko":
            return f"△{magnitude}"
        if lang == "en":
            return f"({magnitude})"
        return f"-{magnitude}"
    return f"{value:,}"


def _format_financial_percent(value: float, lang: str) -> str:
    magnitude = f"{abs(value):.1f}%"
    if value < 0:
        if lang == "ko":
            return f"△{magnitude}"
        if lang == "en":
            return f"({magnitude})"
        return f"-{magnitude}"
    return magnitude
