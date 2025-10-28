import re
from typing import List
from datasets import load_dataset


# --------------------------------------------------------------------------
# 1.1 Text Cleaning Function
# --------------------------------------------------------------------------
def clean_wiki_text(text: str) -> str:
    """Cleans and removes unnecessary markup and special characters from Wikipedia text."""
    # [[Link|Display Text]] -> Display Text
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r" ", text)
    # [[Link]] -> Whitespace
    text = re.sub(r"\[\[([^\]]+)\]\]", r" ", text)
    # Remove URLs
    text = re.sub(r"https?://[^ ]+", "", text)
    # Remove bold/italic markup ('{2,5})
    text = re.sub(r"'{2,5}", "", text)
    # Remove section titles (== Title ==)
    text = re.sub(r"==+\s*(.*?)\s*==+", r" .", text)
    # Remove characters other than Hangul, numbers, spaces, and basic punctuation.
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.?!]", "", text)
    # Convert multiple spaces to a single space and strip whitespace.
    text = " ".join(text.split()).strip()
    return text


# --------------------------------------------------------------------------
# 1.2 Function to Create a Corpus File from Wikipedia
# --------------------------------------------------------------------------
def create_corpus_from_wiki(output_path: str, num_sentences: int = 5000):
    """
    Collects Korean sentences from the Wikimedia dataset to create a corpus file.

    :param output_path: The file path to save the corpus.
    :param num_sentences: The target number of sentences to collect.
    """
    print(f"Starting to create '{output_path}'. Target number of sentences: {num_sentences:,}")

    try:
        # Load the dataset in streaming mode
        dataset = load_dataset(
            "wikimedia/wikipedia", "20231101.ko", split="train", streaming=True
        )
        # Shuffle the data using a buffer (limited in streaming mode)
        shuffled_dataset = dataset.shuffle(buffer_size=10000)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    collected_sentences: List[str] = []
    for data in shuffled_dataset:
        if len(collected_sentences) >= num_sentences:
            break

        cleaned_text = clean_wiki_text(data["text"])
        # Split into sentences (based on spaces after periods, question marks, exclamation marks)
        sentences = re.split(r"(?<=[.?!])\s+", cleaned_text)

        for sentence in sentences:
            s = sentence.strip()
            # Collect only sentences with a length between 10 and 100 characters
            if 10 < len(s) < 100:
                collected_sentences.append(s)
                if len(collected_sentences) % 100 == 0:
                    print(
                        f"... {len(collected_sentences):,} / {num_sentences:,} sentences collected"
                    )
                if len(collected_sentences) >= num_sentences:
                    break

    # Save sentences to the file
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in collected_sentences:
            f.write(sentence + "\n")

    print(
        f"Saved a total of {len(collected_sentences):,} sentences to '{output_path}'."
    )


# --------------------------------------------------------------------------
# 1.3 [Added Function] Function to Create a Corpus of All Korean Characters
# --------------------------------------------------------------------------
def create_korean_char_corpus(output_path: str):
    """
    Creates a corpus file containing all theoretically possible Hangul syllable characters
    by combining initial, medial, and final consonants. Also includes individual consonants and vowels.

    :param output_path: The file path to save the corpus.
    """
    print(f"Starting to create the complete Hangul character corpus '{output_path}'.")

    # Initial consonants (Choseong, 19)
    CHOSUNG = [
        "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ]
    # Medial vowels (Jungseong, 21)
    JUNGSUNG = [
        "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
    ]
    # Final consonants (Jongseong, 28, including empty string for no final)
    JONGSUNG = [
        "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ]

    all_korean_chars = []

    # 1. Generate all Hangul syllable combinations (11,172 characters)
    # Unicode Hangul code point calculation: (chosung_index * 21 * 28) + (jungsung_index * 28) + jongsung_index + 0xAC00 ('Ga')
    for i, _ in enumerate(CHOSUNG):
        for j, _ in enumerate(JUNGSUNG):
            for k, _ in enumerate(JONGSUNG):
                code_point = (i * 21 * 28) + (j * 28) + k + 0xAC00
                all_korean_chars.append(chr(code_point))

    # 2. Add independent consonants and vowels
    all_korean_chars.extend(CHOSUNG)
    all_korean_chars.extend(JUNGSUNG)

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        for char in all_korean_chars:
            f.write(char + "\n")

    total_chars = len(all_korean_chars)
    print(f"Saved a total of {total_chars:,} Hangul characters to '{output_path}'.")
    print(f"(Syllables: {11172:,}, Consonants: {len(CHOSUNG):,}, Vowels: {len(JUNGSUNG):,})")


# --- Example Usage ---
if __name__ == "__main__":
    # Method 1: Create a sentence-based corpus from Wikipedia
    create_corpus_from_wiki("data/corpus.txt", num_sentences=10000)

    # Method 2: Create a character-based corpus consisting of all Hangul characters
    create_korean_char_corpus("data/korean_char_corpus.txt")
