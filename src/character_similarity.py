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


def generate_sentence_typos(
    texts: List[str], db: Dict[str, Any], typo_ratio: float = 0.15
) -> List[tuple[str, str]]:
    """
    주어진 텍스트 목록에 대해 유사 문자를 기반으로 오타를 생성합니다.
    전체 단어 중 `typo_ratio` 비율만큼의 단어에 오타를 생성합니다.

    Args:
        texts: 오타를 생성할 원본 문자열의 리스트.
        db: 유사 문자 데이터베이스.
        typo_ratio: 전체 단어 대비 오타를 생성할 단어의 비율.

    Returns:
        (원본 문장, 오타 문장) 튜플의 리스트.
    """
    generated_sentences_with_original = []
    for text in texts:
        words = text.split()
        if not words:
            generated_sentences_with_original.append((text, text))
            continue

        num_words_to_change = int(len(words) * typo_ratio)
        # typo_ratio가 0보다 크면 최소 1개의 오타를 생성
        if num_words_to_change == 0 and typo_ratio > 0:
            num_words_to_change = 1

        # 단어 수보다 많은 오타를 생성하지 않도록 보장
        num_words_to_change = min(num_words_to_change, len(words))

        indices_to_change = random.sample(range(len(words)), num_words_to_change)

        new_words = list(words)
        for index in indices_to_change:
            word = words[index]
            if not word or len(word) <= 1:
                continue

            # 단어 내에서 변경할 문자의 인덱스를 무작위로 선택
            char_index = random.randint(0, len(word) - 1)
            original_char = word[char_index]

            if original_char.isnumeric() or original_char.isspace():
                continue

            # 유사 문자를 찾아 무작위로 하나 선택
            similar_chars = find_similar_chars(original_char, db, top_n=5)
            if similar_chars:
                similar_char, _ = random.choice(similar_chars)
                word_list = list(word)
                word_list[char_index] = similar_char
                new_words[index] = "".join(word_list)

        generated_sentences_with_original.append((text, " ".join(new_words)))

    return generated_sentences_with_original


def inject_document_typos(
    text: str, db: Dict[str, Any], typo_rate: float = 0.05, top_n: int = 1
) -> str:
    """
    Injects typos into a given text based on character similarity.
    A typo is introduced on a word-by-word basis with a given probability (typo_rate).
    """
    words = text.split(" ")
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
