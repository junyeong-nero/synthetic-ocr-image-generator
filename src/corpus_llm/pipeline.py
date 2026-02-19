import logging
import random
from pathlib import Path
from typing import List, Optional, Sequence

from corpus_llm.constants import CATEGORIES
from corpus_llm.parsing import parse_response
from corpus_llm.prompting import build_prompt
from corpus_llm.providers import LLMProvider

logger = logging.getLogger(__name__)


async def generate_category(
    provider: LLMProvider,
    category: str,
    lang: str,
    count: int,
    batch_size: int = 100,
    lang_name: Optional[str] = None,
) -> List[str]:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    category_info = CATEGORIES[category]
    all_items: List[str] = []
    remaining = count

    while remaining > 0:
        batch = min(batch_size, remaining)
        prompt = build_prompt(category_info, category, lang, batch, lang_name)

        logger.info("Generating %s items for %s (%s)...", batch, category, lang)
        try:
            response = await provider.generate(prompt)
            items = parse_response(response, category)
            items = list(dict.fromkeys(items))
            all_items.extend(items)

            logger.info("  Got %s items (total: %s)", len(items), len(all_items))
            remaining -= batch
        except Exception as exc:
            logger.error("Error generating %s: %s", category, exc)
            break

    all_items = list(dict.fromkeys(all_items))
    return all_items[:count]


def save_corpus(items: Sequence[str], category: str, lang: str, output_dir: Path) -> int:
    lang_dir = output_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    output_file = lang_dir / f"{category}.txt"

    existing = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as file_handle:
            existing = {line.strip() for line in file_handle if line.strip()}

    all_items = list(existing | set(items))
    random.shuffle(all_items)

    with open(output_file, "w", encoding="utf-8") as file_handle:
        for item in all_items:
            file_handle.write(item + "\n")

    logger.info("Saved %s items to %s", len(all_items), output_file)
    return len(all_items)


async def run_generation(
    provider: LLMProvider,
    categories: Sequence[str],
    lang: str,
    count: int,
    batch_size: int,
    output_dir: Path,
    lang_name: Optional[str] = None,
) -> int:
    total_saved = 0

    for category in categories:
        items = await generate_category(
            provider=provider,
            category=category,
            lang=lang,
            count=count,
            batch_size=batch_size,
            lang_name=lang_name,
        )

        if items:
            total_saved += save_corpus(items, category, lang, output_dir)

    logger.info("Total items generated: %s", total_saved)
    return 0
