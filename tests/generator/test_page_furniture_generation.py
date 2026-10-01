"""Page furniture and continuation pages through Generator.generate_single."""

import random
from pathlib import Path
from typing import List

import pytest
from PIL import Image

import src.generator.generator as generator_module
from src.generator.data_provider import DataProvider
from src.generator.generator import Generator
from src.generator.markdown_content import MarkdownDataGenerator
from src.generator.markdown_renderers import PlaywrightMarkdownRenderer

_PAGE = (
    "page:\n"
    "  aspect_ratio: 1.414\n"
    "  margins_mm: {top: 35, bottom: 15, left: 30, right: 30}\n"
)
_FURNITURE = "  header: {p: 1.0}\n  footer: {p: 1.0}\n  page_number: dashed\n"


def _write_profile(tmp_path: Path, body: str) -> str:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "id: t\nfamily_mix: {business: 1.0}\n"
        + body
        + "\ncapture_channels:\n  born_digital:\n    weight: 1.0\n    dpi: 150\n",
        encoding="utf-8",
    )
    return str(path)


def _generator(tmp_path: Path, name: str = "gen") -> Generator:
    # DataProvider shuffles its corpus with the global RNG when it is built, so
    # equal generators need an equal RNG state (the determinism test relies on it).
    random.seed(0)
    root = tmp_path / name
    font_dir = root / "fonts"
    font_dir.mkdir(parents=True)
    (font_dir / "Body.ttf").write_bytes(b"")
    corpus_dir = root / "corpus"
    (corpus_dir / "ko").mkdir(parents=True)
    (corpus_dir / "ko" / "paragraphs.txt").write_text(
        "첫 번째 문단입니다. 두 번째 문장입니다. 세 번째 문장입니다.\n"
        "네 번째 문단입니다. 다섯 번째 문장입니다.\n",
        encoding="utf-8",
    )
    generator = Generator(output_dir=str(root / "out"), font_dir=str(font_dir), lang="ko")
    generator.data_generator = MarkdownDataGenerator(
        "ko", data_provider=DataProvider(lang="ko", corpus_dir=corpus_dir)
    )
    return generator


class _Renders:
    """Record every render: the markdown, the HTML document and the style."""

    def __init__(self) -> None:
        self.markdown: List[str] = []
        self.html: List[str] = []
        self.sheet_heights: List[object] = []


@pytest.fixture
def renders(monkeypatch) -> _Renders:
    record = _Renders()

    class _CapturingRenderer(PlaywrightMarkdownRenderer):
        # Pages are "as tall as" 9px per markdown character, so long
        # documents overflow the sheet and exercise fit-to-sheet trimming.
        px_per_char = 0

        def render(self, markdown_text, image_assets=None):
            record.markdown.append(markdown_text)
            record.html.append(self._build_html_document(markdown_text))
            record.sheet_heights.append(self.style.sheet_height)
            width = self.style.margin_left + self.style.content_width + self.style.margin_right
            height = max(self.style.sheet_height or 0, len(markdown_text) * _CapturingRenderer.px_per_char, 10)
            return Image.new("RGB", (width, height), (255, 255, 255))

    record.renderer = _CapturingRenderer
    monkeypatch.setattr(generator_module, "PlaywrightMarkdownRenderer", _CapturingRenderer)
    return record


def _generate(tmp_path: Path, profile_body: str, *, name: str = "gen", seed: int = 1, template="business_report"):
    generator = _generator(tmp_path, name)
    generator._configure_generation(
        seed=seed, distribution_profile=_write_profile(tmp_path / name, profile_body), template=template
    )
    return generator.generate_single(sample_index=0)


def _body(html: str) -> str:
    return html.split("<body>", 1)[1]


def _content(html: str) -> str:
    body = _body(html)
    return body[body.index('<div class="markdown-body">'):]


def test_furniture_is_drawn_but_not_in_gt(tmp_path, renders) -> None:
    _image, metadata = _generate(tmp_path, _PAGE + _FURNITURE)

    furniture = metadata["page_furniture"]
    assert set(furniture) == {"header", "footer", "page_number"}
    assert all(isinstance(value, str) and value for value in furniture.values())
    assert furniture["page_number"] == "- 1 -"

    html = renders.html[-1]
    body = _body(html)
    assert '<div class="page-sheet">' in body
    # Everything recorded in page_furniture is drawn, outside the markdown body.
    furniture_html = body[: body.index('<div class="markdown-body">')] + body[body.index("page-footer") - 40:]
    for part in furniture["header"].split("\n") + [furniture["footer"], furniture["page_number"]]:
        assert f">{part}<" in furniture_html.replace("&amp;", "&")
    # GT is exactly the rendered markdown body, which carries no furniture.
    assert metadata["GT_markdown"] == renders.markdown[-1]
    assert "- 1 -" not in metadata["GT_markdown"]
    assert "page-furniture" not in _content(html).split("page-footer")[0].rsplit("<div", 1)[0]
    # The sheet container is one page (width x aspect) tall.
    page_width = metadata["image_width"]
    assert renders.sheet_heights[-1] is not None
    assert metadata["image_height"] == pytest.approx(page_width * 1.414, abs=2)


def test_profile_without_furniture_keys_keeps_legacy_page(tmp_path, renders) -> None:
    _image, metadata = _generate(tmp_path, _PAGE)

    assert "page_furniture" not in metadata
    assert "continuation_page" not in metadata
    assert "start_mid_paragraph" not in metadata
    assert all("page-sheet" not in html for html in renders.html)
    assert metadata["GT_markdown"].startswith("# ")


def test_furniture_keys_that_sample_off_record_empty_strings(tmp_path, renders) -> None:
    _image, metadata = _generate(
        tmp_path, _PAGE + "  header: {p: 0.0}\n  footer: {p: 0.0}\n  page_number: none\n"
    )

    assert metadata["page_furniture"] == {"header": "", "footer": "", "page_number": ""}
    assert all("page-sheet" not in html for html in renders.html)


def test_pil_renderer_draws_no_furniture_and_records_none(tmp_path, monkeypatch) -> None:
    class _StubPil:
        def __init__(self, _font_path, style):
            self.style = style

        def render(self, markdown_text):
            assert self.style.page_furniture is None
            return Image.new("RGB", (600, 800), (255, 255, 255))

    monkeypatch.setattr(generator_module, "MarkdownRenderer", _StubPil)
    generator = _generator(tmp_path)
    generator._configure_generation(
        seed=1,
        distribution_profile=_write_profile(tmp_path, _PAGE + _FURNITURE),
        template="business_report",
        markdown_renderer="pil",
    )

    _image, metadata = generator.generate_single(sample_index=0)

    assert metadata["page_furniture"] == {"header": "", "footer": "", "page_number": ""}


def test_continuation_page_generation_is_consistent(tmp_path, renders) -> None:
    body = (
        _PAGE
        + "  page_number: plain\n"
        + "content:\n  continuation_page: {p: 1.0}\n  start_mid_paragraph: {p: 0.5}\n"
    )
    for seed in range(6):
        _image, metadata = _generate(tmp_path, body, name=f"g{seed}", seed=seed)

        assert metadata["continuation_page"] is True
        assert isinstance(metadata["start_mid_paragraph"], bool)
        gt = metadata["GT_markdown"]
        assert not gt.startswith("# ")
        assert int(metadata["page_furniture"]["page_number"]) >= 2
        chunks = [chunk for chunk in gt.strip().split("\n\n") if chunk.strip()]
        blocks = [c for c in chunks if not (c.startswith("#") and "\n" not in c)]
        assert len(blocks) == len(metadata["merge_order"])
        assert metadata["block_types"] == metadata["merge_order"]
        assert metadata["section_count"] == sum(1 for line in gt.splitlines() if line.startswith("## "))


def test_fit_to_sheet_trims_inside_the_furniture_sheet(tmp_path, renders) -> None:
    renders.renderer.px_per_char = 4
    body = _PAGE + _FURNITURE + "content:\n  extra_blocks_per_section: 3\n  paragraph_parts: 4\n"

    _image, metadata = _generate(tmp_path, body)

    assert metadata["page_trimmed"] is True
    assert len(renders.html) > 1
    # Every re-render keeps the furniture, and GT is the last rendered body.
    assert all('<div class="page-sheet">' in html for html in renders.html)
    assert metadata["GT_markdown"] == renders.markdown[-1]
    chunks = [chunk for chunk in metadata["GT_markdown"].strip().split("\n\n") if chunk.strip()]
    blocks = [c for c in chunks if not (c.startswith("#") and "\n" not in c)]
    assert len(blocks) == len(metadata["merge_order"])


def test_furniture_generation_is_deterministic(tmp_path, renders) -> None:
    body = _PAGE + _FURNITURE + "content:\n  continuation_page: {p: 0.5}\n"
    _image_a, first = _generate(tmp_path, body, name="a", seed=3)
    html_a = renders.html[-1]
    _image_b, second = _generate(tmp_path, body, name="b", seed=3)

    assert first == second
    assert renders.html[-1] == html_a.replace(str(tmp_path / "a"), str(tmp_path / "b"))
