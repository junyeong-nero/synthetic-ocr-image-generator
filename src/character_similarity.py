import os
import cv2
import numpy as np
import logging
import random
from typing import Dict, List, Any

from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from utils import save_json, read_json, read_txt
from generator.basic_generator import _generate_text_image

logger = logging.getLogger(__name__)


def calculate_ssim_with_pil_images(imageA_pil, imageB_pil):
    """Calculates the SSIM score between two Pillow Image objects."""
    # Convert Pillow images to grayscale numpy arrays for SSIM calculation
    imageA = cv2.cvtColor(np.array(imageA_pil), cv2.COLOR_RGB2GRAY)
    imageB = cv2.cvtColor(np.array(imageB_pil), cv2.COLOR_RGB2GRAY)

    # Resize imageB to match imageA's dimensions if they are different
    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

    # Calculate the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(imageA, imageB, full=True)
    return score


def build_similarity_database(char_list, font_path, db_path, threshold=0.5):
    """
    Builds a similarity database based on SSIM for a given list of characters.

    Args:
        char_list (list): A list of characters to compare.
        font_path (str): The path to the font file.
        db_path (str): The path to the JSON file where the results will be saved.
        threshold (float): The similarity score threshold for saving the relationship.
    """
    logger.info("Starting to build the similarity database...")
    similarity_db = {}

    # Pre-generate character images to avoid redundant generation
    logger.info("1. Pre-generating character images...")
    char_images = {
        char: _generate_text_image(
            char,
            font_path,
            background_color=(255, 255, 255),
            font_size=24,
            bold=False,
            tilt=0,
            shadow=False,
            distortion=False,
            blur=False,
            contrast=False,
        )
        for char in tqdm(char_list, desc="Generating images")
    }

    # Calculate the similarity for each pair of characters
    logger.info("\n2. Calculating similarity for character pairs...")
    for i in tqdm(range(len(char_list)), desc="Comparing characters"):
        char1 = char_list[i]
        img1 = char_images[char1]

        for j in range(i + 1, len(char_list)):
            char2 = char_list[j]
            img2 = char_images[char2]

            score = calculate_ssim_with_pil_images(img1, img2)

            # If the score is above the threshold, save the relationship in both directions
            if score >= threshold:
                if char1 not in similarity_db:
                    similarity_db[char1] = {}
                similarity_db[char1][char2] = score

                if char2 not in similarity_db:
                    similarity_db[char2] = {}
                similarity_db[char2][char1] = score

    # Save the completed database to a JSON file
    save_json(similarity_db, db_path)
    logger.info("Database build complete.")
    return similarity_db


# --- Step 2: Searching for similar characters using the database --- 


def find_similar_chars(query_char, db, top_n=5):
    """
    Finds characters similar to a given character from the database.

    Args:
        query_char (str): The character to search for.
        db (dict): The pre-built similarity database.
        top_n (int): The maximum number of results to return.

    Returns:
        list: A list of tuples, each containing a similar character and its similarity score.
    """
    if query_char not in db:
        return []

    # Sort the similar characters by score in descending order
    similar_items = sorted(
        db[query_char].items(), key=lambda item: item[1], reverse=True
    )
    return similar_items[:top_n]


# --- Step 3: Generating typos using the database --- 


def generate_sentence_typos(texts: List[str], db: Dict[str, Any], top_n: int = 1) -> List[str]:
    """
    주어진 텍스트 목록에 대해 유사 문자를 기반으로 오타를 생성합니다.

    Args:
        texts: 오타를 생성할 원본 문자열의 리스트.
        db: 유사 문자 데이터베이스.
        top_n: 각 문자에 대해 고려할 유사 문자의 최대 개수.

    Returns:
        생성된 모든 오타 문장들의 리스트.
    """

    def _generate_typos(word: str) -> List[str]:
        """하나의 단어에 대한 오타 후보들을 생성합니다."""
        if not word:  # 빈 문자열 처리
            return [""]

        n = len(word)
        # 단어의 맨 앞/뒤가 아닌, 문자 사이에서 오타를 생성하려면 randint(0, n-1) 사용
        index = random.randint(0, n - 1)

        # 숫자인 경우 오타를 생성하지 않음
        if word[index].isnumeric():
            return [word]

        original_char = word[index]
        word_list: List[str] = list(word)

        similar_chars: List[str] = find_similar_chars(original_char, db, top_n=top_n)

        result: List[str] = []
        for similar_char in similar_chars:
            word_list[index] = similar_char[0]
            result.append("".join(word_list))

        # 원래 단어도 후보에 포함시키려면 아래 주석 해제
        # if original_char not in similar_chars:
        #     result.append(word)

        return result

    all_generated_sentences: List[str] = []
    for text in texts:
        words: List[str] = text.split()

        # 각 단어에 대한 오타 후보들의 리스트 (예: [["안녕", "안녕"], ["하세여", "하새요"]])
        words_candidate: List[List[str]] = [_generate_typos(word) for word in words]

        # 생성된 오타 단어들의 모든 조합을 생성
        # (이 부분은 조합이 기하급수적으로 늘어날 수 있어 주의가 필요합니다)
        sentences: List[str] = [""]
        for typo_words in words_candidate:
            new_sentences: List[str] = [
                f"{sentence} {word}".strip()
                for sentence in sentences
                for word in typo_words
            ]
            sentences = new_sentences
        all_generated_sentences.extend(sentences)

    return all_generated_sentences

def inject_document_typos(text: str, db: Dict[str, Any], typo_rate: float = 0.05, top_n: int = 1) -> str:
    """
    Injects typos into a given text based on character similarity.
    A typo is introduced on a word-by-word basis with a given probability (typo_rate).
    """
    words = text.split(' ')
    new_words = []
    for word in words:
        if random.random() < typo_rate and len(word) > 1:
            # Introduce a typo in this word
            char_index_to_change = random.randint(0, len(word) - 1)
            original_char = word[char_index_to_change]

            if original_char.isnumeric() or original_char.isspace():
                new_words.append(word)
                continue

            similar_chars = find_similar_chars(original_char, db, top_n=top_n)
            if similar_chars:
                # Replace with a similar character
                new_char = random.choice(similar_chars)[0]
                word_list = list(word)
                word_list[char_index_to_change] = new_char
                new_words.append("".join(word_list))
            else:
                # No similar character found, keep original
                new_words.append(word)
        else:
            # No typo for this word
            new_words.append(word)
    return " ".join(new_words)

def generate_similar_chars_db(
    corpus_path="data/corpus.txt",
    db_path="data/char_similarity_db.json",
    font_path="/System/Library/Fonts/Supplemental/AppleGothic.ttf",
):
    """
    Main function to generate and query the character similarity database.
    """
    # Read the character corpus and create a unique list of characters
    chars = read_txt(corpus_path)
    chars = list(set(list(chars)))

    logger.info(f"Total unique characters: {len(chars)}")
    build_similarity_database(chars, font_path, db_path, threshold=0.6)


if __name__ == "__main__":
    generate_similar_chars_db()