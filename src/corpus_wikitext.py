"""Import WikiText-format dumps (e.g. Korean WikiText) as generator corpus files.

WikiText files mark article titles as `` = Title = `` and section headers as
`` = = Section = = ``; everything else is running text, one paragraph per
line. The importer writes ``paragraphs.txt`` and ``titles.txt`` under
``<output_dir>/<lang>/``, which ``DataProvider`` picks up automatically.
"""

from __future__ import annotations

import io
import logging
import random
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

KOWIKITEXT_URLS = {
    split: (
        "https://github.com/lovit/kowikitext/releases/download/"
        f"kowikitext.20200920.v2/kowikitext_20200920.{split}.zip"
    )
    for split in ("train", "dev", "test")
}

_HEADER_RE = re.compile(r"^\s*(=+)\s*(.*?)\s*=+\s*$")
# Markup stripping leaves empty or punctuation-only parentheses, e.g. "(, )".
_EMPTY_PAREN_RE = re.compile(r"[(（]\s*[,;:·、\s]*\s*[)）]")
_DANGLING_COMMA_PAREN_RE = re.compile(r"\s*[,;]\s*([)）])")
_MISSING_SENTENCE_SPACE_RE = re.compile(r"([다요음함임됨][.!?])(?=[가-힣A-Za-z\"'(])")
_NAMESPACE_TITLE_RE = re.compile(r"^[^\s:]{1,12}:")
# Wiki list / indent markup ("# item", "* item", ": quote") that survived dumping.
_LEADING_MARKUP_RE = re.compile(r"^(?:[#*:;|>\-–]+\s*)+")
_INLINE_MARKUP_RE = re.compile(r"(?<=[\s.])[#*:;]+\s+")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


@dataclass(frozen=True)
class WikitextCorpus:
    paragraphs: List[str]
    titles: List[str]


def clean_text(text: str) -> str:
    text = _LEADING_MARKUP_RE.sub("", text.strip())
    text = _INLINE_MARKUP_RE.sub("", text)
    text = _EMPTY_PAREN_RE.sub("", text)
    text = _DANGLING_COMMA_PAREN_RE.sub(r"\1", text)
    text = _MISSING_SENTENCE_SPACE_RE.sub(r"\1 ", text)
    text = text.replace("..", ".")
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def iter_wikitext_lines(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            yield from handle


def parse_wikitext(
    lines: Iterable[str],
    *,
    min_chars: int = 60,
    max_chars: int = 700,
    min_title_chars: int = 2,
    max_title_chars: int = 40,
) -> WikitextCorpus:
    paragraphs: List[str] = []
    titles: List[str] = []
    seen_paragraphs: set[str] = set()
    seen_titles: set[str] = set()

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        header = _HEADER_RE.match(line)
        if header:
            title = clean_text(header.group(2).replace("=", " "))
            # Namespace pages such as "분류:..." or "틀:..." are not document titles.
            if _NAMESPACE_TITLE_RE.match(title):
                continue
            if min_title_chars <= len(title) <= max_title_chars and title not in seen_titles:
                seen_titles.add(title)
                titles.append(title)
            continue
        text = clean_text(line)
        if not (min_chars <= len(text) <= max_chars) or text in seen_paragraphs:
            continue
        # Skip list/table debris: mostly digits or symbols.
        letters = sum(ch.isalpha() for ch in text)
        if letters < len(text) * 0.5:
            continue
        seen_paragraphs.add(text)
        paragraphs.append(text)
    return WikitextCorpus(paragraphs=paragraphs, titles=titles)


def download_kowikitext(split: str, cache_dir: Path) -> Path:
    if split not in KOWIKITEXT_URLS:
        raise ValueError(f"Unknown kowikitext split '{split}'. Choose from {sorted(KOWIKITEXT_URLS)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"kowikitext_20200920.{split}"
    if target.exists():
        return target
    logger.info("Downloading %s", KOWIKITEXT_URLS[split])
    with urllib.request.urlopen(KOWIKITEXT_URLS[split]) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(name for name in archive.namelist() if name.endswith(split))
        target.write_bytes(archive.read(member))
    return target


def write_corpus(
    corpus: WikitextCorpus,
    output_dir: Path,
    lang: str,
    *,
    max_paragraphs: Optional[int] = None,
    seed: int = 0,
) -> Tuple[Path, Path]:
    lang_dir = output_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    paragraphs = list(corpus.paragraphs)
    random.Random(seed).shuffle(paragraphs)
    if max_paragraphs:
        paragraphs = paragraphs[:max_paragraphs]
    paragraphs_path = lang_dir / "paragraphs.txt"
    titles_path = lang_dir / "titles.txt"
    paragraphs_path.write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    titles_path.write_text("\n".join(corpus.titles) + "\n", encoding="utf-8")
    return paragraphs_path, titles_path
