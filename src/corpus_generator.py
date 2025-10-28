import re
import yaml
from typing import List
from datasets import load_dataset


LANG_CONFIG = {
    "ko": {
        "dataset_id": "20231101.ko",
        "char_regex": r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.?!]",
    },
    "en": {
        "dataset_id": "20231101.en",
        "char_regex": r"[^a-zA-Z0-9\s.?!]",
    },
    "ja": {
        "dataset_id": "20231101.ja",
        "char_regex": r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF0-9\s.?!]",
    },
    "ar": {
        "dataset_id": "20231101.ar",
        "char_regex": r"[^\u0600-\u06FF0-9\s.?!]",
    },
    "hi": {
        "dataset_id": "20231101.hi",
        "char_regex": r"[^\u0900-\u097F0-9\s.?!]",
    },
    "vi": {
        "dataset_id": "20231101.vi",
        "char_regex": r"[^a-zA-Z\u00C0-\u017F0-9\s.?!]",
    },
    "th": {
        "dataset_id": "20231101.th",
        "char_regex": r"[^\u0E00-\u0E7F0-9\s.?!]",
    },
}


def clean_wiki_text(text: str, lang: str) -> str:
    """Cleans and removes unnecessary markup and special characters from Wikipedia text."""
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r" ", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r" ", text)
    text = re.sub(r"https?://[^ ]+", "", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"==+\s*(.*?)\s*==+", r" .", text)

    if lang in LANG_CONFIG:
        text = re.sub(LANG_CONFIG[lang]["char_regex"], "", text)

    text = " ".join(text.split()).strip()
    return text


# --------------------------------------------------------------------------
# 1.2 Function to Create a Corpus File from Wikipedia
# --------------------------------------------------------------------------
def create_corpus_from_wiki(output_path: str, lang: str, num_sentences: int = 5000):
    """
    Collects sentences from the Wikimedia dataset for a specified language to create a corpus file.
    """
    if lang not in LANG_CONFIG:
        print(
            f"Error: Language '{lang}' is not supported. Supported languages are: {list(LANG_CONFIG.keys())}"
        )
        return

    lang_settings = LANG_CONFIG[lang]
    print(
        f"Starting to create '{output_path}' for language '{lang}'. Target sentences: {num_sentences:,}"
    )

    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            lang_settings["dataset_id"],
            split="train",
            streaming=True,
        )
        shuffled_dataset = dataset.shuffle(buffer_size=10000)
    except Exception as e:
        print(f"Error loading dataset for language '{lang}': {e}")
        return

    collected_sentences: List[str] = []
    for data in shuffled_dataset:
        if len(collected_sentences) >= num_sentences:
            break

        cleaned_text = clean_wiki_text(data["text"], lang)
        sentences = re.split(r"(?<=[.?!])\s+", cleaned_text)

        for sentence in sentences:
            s = sentence.strip()
            if 10 < len(s) < 100:
                collected_sentences.append(s)
                if len(collected_sentences) % 100 == 0:
                    print(
                        f"... {len(collected_sentences):,} / {num_sentences:,} sentences collected"
                    )
                if len(collected_sentences) >= num_sentences:
                    break

    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in collected_sentences:
            f.write(sentence + "\n")

    print(
        f"Saved a total of {len(collected_sentences):,} sentences to '{output_path}'."
    )


def create_all_chars_corpus(output_path: str):
    """
    Creates a corpus file containing all theoretically possible Hangul syllable characters
    by combining initial, medial, and final consonants. Also includes individual consonants and vowels.

    :param output_path: The file path to save the corpus.
    """
    print(f"Starting to create the complete Hangul character corpus '{output_path}'.")

    CHOSUNG = [
        "ㄱ",
        "ㄲ",
        "ㄴ",
        "ㄷ",
        "ㄸ",
        "ㄹ",
        "ㅁ",
        "ㅂ",
        "ㅃ",
        "ㅅ",
        "ㅆ",
        "ㅇ",
        "ㅈ",
        "ㅉ",
        "ㅊ",
        "ㅋ",
        "ㅌ",
        "ㅍ",
        "ㅎ",
    ]
    JUNGSUNG = [
        "ㅏ",
        "ㅐ",
        "ㅑ",
        "ㅒ",
        "ㅓ",
        "ㅔ",
        "ㅕ",
        "ㅖ",
        "ㅗ",
        "ㅘ",
        "ㅙ",
        "ㅚ",
        "ㅛ",
        "ㅜ",
        "ㅝ",
        "ㅞ",
        "ㅟ",
        "ㅠ",
        "ㅡ",
        "ㅢ",
        "ㅣ",
    ]
    JONGSUNG = [
        "",
        "ㄱ",
        "ㄲ",
        "ㄳ",
        "ㄴ",
        "ㄵ",
        "ㄶ",
        "ㄷ",
        "ㄹ",
        "ㄺ",
        "ㄻ",
        "ㄼ",
        "ㄽ",
        "ㄾ",
        "ㄿ",
        "ㅀ",
        "ㅁ",
        "ㅂ",
        "ㅄ",
        "ㅅ",
        "ㅆ",
        "ㅇ",
        "ㅈ",
        "ㅊ",
        "ㅋ",
        "ㅌ",
        "ㅍ",
        "ㅎ",
    ]

    all_korean_chars = []

    for i, _ in enumerate(CHOSUNG):
        for j, _ in enumerate(JUNGSUNG):
            for k, _ in enumerate(JONGSUNG):
                code_point = (i * 21 * 28) + (j * 28) + k + 0xAC00
                all_korean_chars.append(chr(code_point))

    all_korean_chars.extend(CHOSUNG)
    all_korean_chars.extend(JUNGSUNG)

    with open(output_path, "w", encoding="utf-8") as f:
        for char in all_korean_chars:
            f.write(char + "\n")

    total_chars = len(all_korean_chars)
    print(f"Saved a total of {total_chars:,} Hangul characters to '{output_path}'.")
    print(
        f"(Syllables: {11172:,}, Consonants: {len(CHOSUNG):,}, Vowels: {len(JUNGSUNG):,})"
    )
