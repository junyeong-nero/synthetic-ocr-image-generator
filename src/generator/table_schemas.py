"""Coherent table schemas for realistic table content.

A :class:`TableSchema` describes a table whose columns have a *semantic*
type (an amount, a percentage, a phone number, ...) instead of the legacy
"column 0 is a product name, column 1 is a quantity" convention. Column
values generated for a schema are internally consistent (e.g. an order's
``금액`` column always equals ``수량 x 단가``); see
:mod:`src.generator.table_generator` for the row-generation logic that uses
these definitions.

Header labels are localized for ``ko``/``en``/``ja``; any other language
falls back to the English header (see :func:`resolve_header`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Column types that are rendered right-aligned (`---:`) because their values
# are numbers meant to be compared vertically. Everything else (labels,
# dates, names, free text, ...) stays left-aligned.
NUMERIC_COLUMN_TYPES = frozenset({"integer", "amount", "percent", "change"})

SUPPORTED_HEADER_LANGS = ("ko", "en", "ja")


@dataclass(frozen=True)
class TableColumn:
    """One column of a :class:`TableSchema`.

    ``required`` columns are always rendered; optional columns are dropped
    (from the right, see :func:`select_schema_columns`) when the blueprint's
    column range is narrower than the schema's full column count.
    """

    key: str
    type: str
    header: Dict[str, str]
    required: bool = True


@dataclass(frozen=True)
class TableSchema:
    """A named, coherent table shape."""

    name: str
    columns: Tuple[TableColumn, ...]
    # Minimum row count this schema needs to render meaningfully (e.g. an
    # `order` table needs at least one data row plus its summary row).
    min_rows: int = 1
    has_summary_row: bool = False


def resolve_header(column: TableColumn, lang: str) -> str:
    """Localized header text for ``column``, falling back to English."""

    resolved_lang = lang if lang in SUPPORTED_HEADER_LANGS else "en"
    return column.header.get(resolved_lang) or column.header.get("en", column.key)


def required_column_count(schema: TableSchema) -> int:
    """Number of columns that cannot be dropped to fit a narrow width."""

    return sum(1 for column in schema.columns if column.required)


def select_schema_columns(schema: TableSchema, column_count: int) -> List[TableColumn]:
    """Columns to render for a target width, dropping optional ones first.

    Optional columns are dropped from the right (the last optional column
    still present) until the schema fits ``column_count``, but the result
    never shrinks below the schema's required columns and never exceeds its
    declared columns.
    """

    columns = list(schema.columns)
    required = required_column_count(schema)
    target = max(required, min(column_count, len(columns)))

    result = list(columns)
    while len(result) > target:
        for index in range(len(result) - 1, -1, -1):
            if not result[index].required:
                result.pop(index)
                break
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Curated label lists (ko/en/ja). These are domain vocabulary the DataProvider
# corpus does not carry (account items, budget categories, place names, order
# specs, statistics categories), so they live here rather than in
# language_data.py, which holds general-purpose prose data.
# ---------------------------------------------------------------------------

# Financial statement accounts are listed top-to-bottom the way a real
# financial statement lists them (income statement: revenue down to net
# income; balance sheet: assets, then liabilities, then equity, each with
# its subtotal). `table_generator._build_financial_rows` picks one of these
# two statements and a contiguous run from it, rather than shuffling, so
# emitted rows keep their canonical relative order (e.g. 매출액 always comes
# before 매출총이익, never after 당기순이익).
FINANCIAL_INCOME_STATEMENT_ITEMS: Dict[str, List[str]] = {
    "ko": [
        "매출액", "매출원가", "매출총이익", "판매비와관리비", "영업이익",
        "영업외수익", "영업외비용", "법인세비용차감전순이익", "법인세비용",
        "당기순이익",
    ],
    "en": [
        "Revenue", "Cost of Goods Sold", "Gross Profit",
        "Selling & Administrative Expenses", "Operating Income",
        "Non-operating Income", "Non-operating Expenses",
        "Income Before Tax", "Income Tax Expense", "Net Income",
    ],
    "ja": [
        "売上高", "売上原価", "売上総利益", "販売費及び一般管理費", "営業利益",
        "営業外収益", "営業外費用", "税引前当期純利益", "法人税等",
        "当期純利益",
    ],
}

FINANCIAL_BALANCE_SHEET_ITEMS: Dict[str, List[str]] = {
    "ko": [
        "유동자산", "비유동자산", "자산총계", "유동부채", "비유동부채",
        "부채총계", "자본총계",
    ],
    "en": [
        "Current Assets", "Non-current Assets", "Total Assets",
        "Current Liabilities", "Non-current Liabilities",
        "Total Liabilities", "Total Equity",
    ],
    "ja": [
        "流動資産", "固定資産", "資産合計", "流動負債", "固定負債",
        "負債合計", "純資産合計",
    ],
}

# The two statements `_build_financial_rows` chooses between.
FINANCIAL_STATEMENTS: Tuple[Dict[str, List[str]], ...] = (
    FINANCIAL_INCOME_STATEMENT_ITEMS,
    FINANCIAL_BALANCE_SHEET_ITEMS,
)

BUDGET_CATEGORIES: Dict[str, List[str]] = {
    "ko": [
        "인건비", "운영비", "사업비", "시설비", "홍보비", "교육훈련비",
        "여비교통비", "자산취득비", "예비비", "일반수용비",
    ],
    "en": [
        "Personnel Costs", "Operating Expenses", "Program Costs",
        "Facility Costs", "Publicity Costs", "Training Costs",
        "Travel Expenses", "Capital Acquisition", "Contingency Reserve",
        "General Supplies",
    ],
    "ja": [
        "人件費", "運営費", "事業費", "施設費", "広報費", "教育訓練費",
        "旅費交通費", "資産取得費", "予備費", "一般需用費",
    ],
}

PLACE_NAMES: Dict[str, List[str]] = {
    "ko": [
        "대회의실", "제1회의실", "제2회의실", "세미나실", "강당",
        "본관 3층 회의실", "별관 2층 소회의실", "교육장", "상황실",
        "영상회의실",
    ],
    "en": [
        "Main Conference Room", "Meeting Room 1", "Meeting Room 2",
        "Seminar Room", "Auditorium", "3F Conference Room, Main Bldg",
        "2F Meeting Room, Annex", "Training Center", "Situation Room",
        "Video Conference Room",
    ],
    "ja": [
        "大会議室", "第1会議室", "第2会議室", "セミナー室", "講堂",
        "本館3階会議室", "別館2階小会議室", "研修室", "状況室",
        "ビデオ会議室",
    ],
}

ORDER_SPECS: Dict[str, List[str]] = {
    "ko": ["A4", "1EA", "1SET", "1BOX(10EA)", "500매", "1롤", "1박스", "中", "大", "소형"],
    "en": [
        "A4", "1EA", "1SET", "1BOX(10EA)", "500 sheets", "1 roll", "1 box",
        "Medium", "Large", "Small",
    ],
    "ja": ["A4", "1個", "1セット", "1箱(10個)", "500枚", "1本", "1箱", "中", "大", "小"],
}


# `DataProvider.name()`/`.phone_number()`/`.date()`/`.time()` fall back to
# Faker when no corpus is loaded, and Faker's own RNG is independent of the
# per-sample `random.seed()` the rest of the generator uses. `Generator.
# _seed_for_sample` (src/generator/generator.py) does call `faker.
# seed_instance(sample_seed)` on every sample, which keeps Faker output
# reproducible when going through `Generator.generate_single()` -- but
# `TableGenerator`/`DataProvider` are also used directly (unit tests, other
# callers) without going through `Generator` at all, and there `random.
# seed()` alone would not make Faker calls reproducible. So the `roster`/
# `schedule` schemas use `PERSON_NAMES` here instead of `DataProvider.
# name()`'s Faker fallback, and `table_generator` formats phone/date/time
# from `random` directly, to stay reproducible from `random.seed()` alone
# regardless of caller.
PERSON_NAMES: Dict[str, List[str]] = {
    "ko": [
        "김민준", "이서연", "박도윤", "최지우", "정하은", "강시우", "조유나",
        "윤재현", "장수빈", "임하람", "한지호", "오세아",
    ],
    "en": [
        "James Carter", "Emily Johnson", "Michael Lee", "Sarah Davis",
        "David Kim", "Laura Martinez", "Daniel Brown", "Jessica Wilson",
        "Matthew Clark", "Amanda Lewis", "Christopher Young", "Rachel Adams",
    ],
    "ja": [
        "佐藤陽翔", "鈴木陽菜", "高橋蓮", "田中結衣", "伊藤大和",
        "渡辺美咲", "山本悠真", "中村さくら", "小林颯太", "加藤美月",
        "吉田翼", "山田葵",
    ],
}

STATISTICS_CATEGORIES: Dict[str, List[str]] = {
    "ko": ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "경기", "강원", "제주"],
    "en": ["Seoul", "Busan", "Daegu", "Incheon", "Gwangju", "Daejeon", "Ulsan", "Gyeonggi", "Gangwon", "Jeju"],
    "ja": ["ソウル", "釜山", "大邱", "仁川", "光州", "大田", "蔚山", "京畿", "江原", "済州"],
}

SUMMARY_ROW_LABEL: Dict[str, str] = {"ko": "합계", "en": "Total", "ja": "合計"}


def resolve_curated_lang(lang: str) -> str:
    """Curated lists cover ko/en/ja; anything else uses the English list."""

    return lang if lang in ("ko", "en", "ja") else "en"


def summary_row_label(lang: str) -> str:
    return SUMMARY_ROW_LABEL.get(resolve_curated_lang(lang), SUMMARY_ROW_LABEL["en"])


# ---------------------------------------------------------------------------
# Schema registry.
# ---------------------------------------------------------------------------

SCHEMAS: Dict[str, TableSchema] = {
    "financial": TableSchema(
        name="financial",
        columns=(
            TableColumn("item", "label", {"ko": "항목", "en": "Item", "ja": "項目"}),
            TableColumn(
                "current", "amount",
                {"ko": "당기(백만원)", "en": "Current (KRW mn)", "ja": "当期（百万円）"},
            ),
            TableColumn(
                "prior", "amount",
                {"ko": "전기(백만원)", "en": "Prior (KRW mn)", "ja": "前期（百万円）"},
            ),
            TableColumn(
                "change", "change",
                {"ko": "증감(백만원)", "en": "Change (KRW mn)", "ja": "増減（百万円）"},
                required=False,
            ),
            TableColumn(
                "change_rate", "percent",
                {"ko": "증감률", "en": "Change Rate", "ja": "増減率"},
                required=False,
            ),
        ),
    ),
    "budget": TableSchema(
        name="budget",
        columns=(
            TableColumn("category", "label", {"ko": "구분", "en": "Category", "ja": "区分"}),
            TableColumn("budget", "amount", {"ko": "예산액", "en": "Budget", "ja": "予算額"}),
            TableColumn("executed", "amount", {"ko": "집행액", "en": "Executed", "ja": "執行額"}),
            TableColumn(
                "execution_rate", "percent",
                {"ko": "집행률", "en": "Execution Rate", "ja": "執行率"},
                required=False,
            ),
        ),
    ),
    "schedule": TableSchema(
        name="schedule",
        columns=(
            TableColumn("date", "date", {"ko": "일자", "en": "Date", "ja": "日付"}),
            TableColumn("time", "time", {"ko": "시간", "en": "Time", "ja": "時間"}, required=False),
            TableColumn("content", "text", {"ko": "내용", "en": "Content", "ja": "内容"}),
            TableColumn("place", "label", {"ko": "장소", "en": "Location", "ja": "場所"}, required=False),
            TableColumn(
                "owner", "department",
                {"ko": "담당", "en": "Owner", "ja": "担当"},
                required=False,
            ),
        ),
    ),
    "roster": TableSchema(
        name="roster",
        columns=(
            TableColumn("name", "person", {"ko": "성명", "en": "Name", "ja": "氏名"}),
            TableColumn("department", "department", {"ko": "소속", "en": "Department", "ja": "所属"}),
            TableColumn(
                "position", "position",
                {"ko": "직위", "en": "Position", "ja": "職位"},
                required=False,
            ),
            TableColumn(
                "phone", "phone",
                {"ko": "연락처", "en": "Phone", "ja": "連絡先"},
                required=False,
            ),
        ),
    ),
    "order": TableSchema(
        name="order",
        columns=(
            TableColumn("item", "label", {"ko": "품목", "en": "Item", "ja": "品目"}),
            TableColumn(
                "spec", "text",
                {"ko": "규격", "en": "Spec", "ja": "規格"},
                required=False,
            ),
            TableColumn("quantity", "integer", {"ko": "수량", "en": "Qty", "ja": "数量"}),
            TableColumn("unit_price", "amount", {"ko": "단가", "en": "Unit Price", "ja": "単価"}),
            TableColumn("amount", "amount", {"ko": "금액", "en": "Amount", "ja": "金額"}),
        ),
        min_rows=2,
        has_summary_row=True,
    ),
    "statistics": TableSchema(
        name="statistics",
        # Year columns are appended dynamically (see
        # table_generator.build_statistics_columns) based on the requested
        # column width; this entry only carries the fixed label column.
        columns=(
            TableColumn("category", "label", {"ko": "구분", "en": "Category", "ja": "区分"}),
        ),
        min_rows=1,
    ),
}
