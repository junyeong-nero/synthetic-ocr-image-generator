from faker.config import AVAILABLE_LOCALES


FAKER_LOCALES: dict[str, str] = {
    "ko": "ko_KR",
    "en": "en_US",
    "ja": "ja_JP",
    "hi": "hi_IN",
}


DEFAULT_FAKER_LOCALE_BY_BASE: dict[str, str] = {
    "af": "en_ZA",
    "ar": "ar_EG",
    "bn": "en_IN",
    "de": "de_DE",
    "en": "en_US",
    "es": "es_ES",
    "fa": "en_US",
    "fr": "fr_FR",
    "gu": "en_IN",
    "he": "he_IL",
    "hi": "hi_IN",
    "id": "id_ID",
    "it": "it_IT",
    "ja": "ja_JP",
    "ko": "ko_KR",
    "mr": "en_IN",
    "nl": "nl_NL",
    "pa": "en_IN",
    "pl": "pl_PL",
    "pt": "pt_BR",
    "ru": "ru_RU",
    "ta": "en_IN",
    "te": "en_IN",
    "th": "th_TH",
    "tr": "tr_TR",
    "uk": "uk_UA",
    "ur": "en_PK",
    "vi": "vi_VN",
    "zh": "zh_CN",
}


AVAILABLE_LOCALE_LOOKUP: dict[str, str] = {
    locale.lower(): locale for locale in AVAILABLE_LOCALES
}


def normalize_lang_code(lang: str) -> str:
    return lang.strip().lower().replace("_", "-")


def base_lang_code(lang: str) -> str:
    return normalize_lang_code(lang).split("-", 1)[0]


def resolve_faker_locale(lang: str) -> str:
    normalized = normalize_lang_code(lang)
    base = base_lang_code(lang)

    candidates: list[str] = []
    if "-" in normalized:
        base_code, region = normalized.split("-", 1)
        candidates.append(f"{base_code}_{region.upper()}")
    candidates.append(normalized.replace("-", "_"))

    if normalized in FAKER_LOCALES:
        candidates.append(FAKER_LOCALES[normalized])
    if base in FAKER_LOCALES:
        candidates.append(FAKER_LOCALES[base])
    if base in DEFAULT_FAKER_LOCALE_BY_BASE:
        candidates.append(DEFAULT_FAKER_LOCALE_BY_BASE[base])

    prefix_matches = sorted(
        locale for locale in AVAILABLE_LOCALES if locale.lower().startswith(f"{base}_")
    )
    candidates.extend(prefix_matches)
    candidates.append("en_US")

    seen = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        if key in AVAILABLE_LOCALE_LOOKUP:
            return AVAILABLE_LOCALE_LOOKUP[key]

    return "en_US"
