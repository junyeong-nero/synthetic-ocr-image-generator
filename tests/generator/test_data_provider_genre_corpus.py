"""Genre-aware corpus lookups on `DataProvider` (Task 11).

`paragraph(genre=...)` lets callers ask for genre-flavored text (report
lines, meeting notes, ...) with a fallback to the general `paragraphs.txt`
corpus when the genre file is absent -- and `paragraph()` with no genre must
behave exactly as before (legacy unchanged).
"""

from pathlib import Path

from src.generator.data_provider import DataProvider


def _write_corpus(tmp_path: Path, lang: str, filename: str, lines: list[str]) -> Path:
    lang_dir = tmp_path / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    (lang_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_paragraph_with_genre_uses_genre_corpus_when_present(tmp_path) -> None:
    corpus_dir = _write_corpus(
        tmp_path, "ko", "paragraphs.txt", ["일반 위키 문단입니다."]
    )
    _write_corpus(
        tmp_path,
        "ko",
        "report_lines.txt",
        ["3분기 매출 목표 대비 108% 달성함", "신규 거래처 5개사 발굴 완료함"],
    )
    provider = DataProvider(lang="ko", corpus_dir=corpus_dir)

    result = provider.paragraph(genre="report_lines")

    assert result in {"3분기 매출 목표 대비 108% 달성함", "신규 거래처 5개사 발굴 완료함"}


def test_paragraph_with_genre_falls_back_to_general_corpus_when_absent(tmp_path) -> None:
    corpus_dir = _write_corpus(
        tmp_path, "ko", "paragraphs.txt", ["일반 위키 문단입니다."]
    )
    provider = DataProvider(lang="ko", corpus_dir=corpus_dir)

    result = provider.paragraph(genre="report_lines")

    assert result == "일반 위키 문단입니다."


def test_paragraph_without_genre_argument_stays_legacy(tmp_path) -> None:
    corpus_dir = _write_corpus(
        tmp_path, "ko", "paragraphs.txt", ["일반 위키 문단입니다."]
    )
    _write_corpus(
        tmp_path, "ko", "report_lines.txt", ["3분기 매출 목표 대비 108% 달성함"]
    )
    provider = DataProvider(lang="ko", corpus_dir=corpus_dir)

    # No genre requested: must ignore the genre corpus entirely, exactly like
    # before this feature existed.
    assert provider.paragraph() == "일반 위키 문단입니다."


def test_has_corpus_reports_genre_files() -> None:
    provider = DataProvider(lang="ko", use_corpus=False)
    assert provider.has_corpus("report_lines") is False
