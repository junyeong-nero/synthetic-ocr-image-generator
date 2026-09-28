from corpus_llm.constants import CATEGORIES
from corpus_llm.parsing import parse_response
from corpus_llm.pipeline import save_corpus

_GENRE_CATEGORIES = (
    "report_lines",
    "meeting_notes",
    "notice_paragraphs",
    "contract_clauses",
    "academic_abstracts",
    "financial_commentary",
)


def test_parse_response_strips_leading_indices_from_paragraphs() -> None:
    response = """00. 첫 번째 문단입니다. 두 번째 문장도 있습니다.

01. 다음 문단입니다. 역시 번호 없이 저장되어야 합니다.
"""

    assert parse_response(response, "paragraphs") == [
        "첫 번째 문단입니다. 두 번째 문장도 있습니다.",
        "다음 문단입니다. 역시 번호 없이 저장되어야 합니다.",
    ]


def test_parse_response_strips_markdown_wrapped_indices_from_paragraphs() -> None:
    response = """**87.** Bold marker paragraph should also be cleaned.

_12._ Another paragraph should lose the markdown-wrapped index.
"""

    assert parse_response(response, "paragraphs") == [
        "Bold marker paragraph should also be cleaned.",
        "Another paragraph should lose the markdown-wrapped index.",
    ]


def test_save_corpus_rewrites_existing_numbered_paragraph_entries(tmp_path) -> None:
    lang_dir = tmp_path / "ko"
    lang_dir.mkdir(parents=True, exist_ok=True)
    output_file = lang_dir / "paragraphs.txt"
    output_file.write_text(
        "00. 기존 문단입니다. 번호가 제거되어야 합니다.\n",
        encoding="utf-8",
    )

    saved_count = save_corpus(
        ["01. 새 문단입니다. 이 줄도 번호 없이 저장됩니다."],
        category="paragraphs",
        lang="ko",
        output_dir=tmp_path,
    )

    saved_lines = {
        line.strip() for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()
    }

    assert saved_count == 2
    assert saved_lines == {
        "기존 문단입니다. 번호가 제거되어야 합니다.",
        "새 문단입니다. 이 줄도 번호 없이 저장됩니다.",
    }


def test_genre_categories_have_ko_en_ja_prompts() -> None:
    for category in _GENRE_CATEGORIES:
        assert category in CATEGORIES
        prompts = CATEGORIES[category]["prompts"]
        for lang in ("ko", "en", "ja"):
            assert lang in prompts
            assert "{count}" in prompts[lang]


def test_parse_response_treats_report_lines_as_one_item_per_line() -> None:
    response = """3분기 매출 목표 대비 108% 달성함
신규 거래처 5개사 발굴 완료함

재고 관리 시스템 고도화 진행 중임
"""

    assert parse_response(response, "report_lines") == [
        "3분기 매출 목표 대비 108% 달성함",
        "신규 거래처 5개사 발굴 완료함",
        "재고 관리 시스템 고도화 진행 중임",
    ]


def test_parse_response_groups_meeting_notes_like_paragraphs() -> None:
    response = """차기 프로젝트 일정에 대해 논의함.
2주간 일정을 조정하여 다음 달 초에 착수하기로 결정함.

예산 초과분에 대한 대응 방안을 검토함.
"""

    assert parse_response(response, "meeting_notes") == [
        "차기 프로젝트 일정에 대해 논의함. 2주간 일정을 조정하여 다음 달 초에 착수하기로 결정함.",
        "예산 초과분에 대한 대응 방안을 검토함.",
    ]
