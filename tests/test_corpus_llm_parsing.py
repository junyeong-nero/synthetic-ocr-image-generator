from corpus_llm.parsing import parse_response
from corpus_llm.pipeline import save_corpus


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
