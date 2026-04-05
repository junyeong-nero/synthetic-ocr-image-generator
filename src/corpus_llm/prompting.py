from typing import Any, Dict, Optional

from src.corpus_llm.constants import LANGUAGE_LABELS


def resolve_language_label(lang: str, lang_name: Optional[str] = None) -> str:
    if lang_name:
        return lang_name

    normalized = lang.lower().replace("_", "-")
    base_code = normalized.split("-", 1)[0]
    return LANGUAGE_LABELS.get(base_code, lang)


def build_prompt(
    category_info: Dict[str, Any],
    category: str,
    lang: str,
    count: int,
    lang_name: Optional[str] = None,
) -> str:
    prompts = category_info["prompts"]
    typed_prompts = prompts if isinstance(prompts, dict) else {}

    if lang in typed_prompts:
        return str(typed_prompts[lang]).format(count=count)

    fallback_prompt = typed_prompts.get("en") or next(iter(typed_prompts.values()), "")
    language_label = resolve_language_label(lang, lang_name)

    return (
        "You are generating OCR corpus entries.\n"
        f"Category: {category}\n"
        f"Target language: {language_label} (code: {lang})\n"
        f"Generate exactly {count} unique items.\n"
        "Adapt names, addresses, and wording to the target language and locale naturally.\n"
        "Return plain text only, one item per line, with no numbering or commentary.\n\n"
        "Category guidance:\n"
        f"{fallback_prompt.format(count=count)}"
    )
