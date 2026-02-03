import cv2
import numpy as np
import logging
import random
import unicodedata
from typing import Dict, List, Any, Iterable, Tuple

from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from utils import save_json, read_txt

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


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _preprocess_image_for_ssim(image_pil, size: int = 32) -> np.ndarray:
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


def _low_res_embedding(image_gray: np.ndarray, size: int = 8) -> np.ndarray:
    if image_gray.shape[0] != size or image_gray.shape[1] != size:
        image_gray = cv2.resize(image_gray, (size, size), interpolation=cv2.INTER_AREA)
    vec = image_gray.astype(np.float32).flatten()
    return _normalize_vector(vec)


def _iter_valid_chars(text: str) -> Iterable[str]:
    for ch in text:
        if ch.isspace():
            continue
        if unicodedata.category(ch)[0] == "C":
            continue
        yield ch


def build_similarity_database(
    char_list, font_path, db_path, threshold=0.5, top_k=8
):
    """
    Builds a similarity database based on SSIM for a given list of characters.

    Args:
        char_list (list): A list of characters to compare.
        font_path (str): The path to the font file.
        db_path (str): The path to the JSON file where the results will be saved.
        threshold (float): The similarity score threshold for saving the relationship.
        top_k (int): The maximum number of similar characters to store per character.
    """
    from generator.effects import render_text_with_effects

    logger.info("Starting to build the similarity database...")
    similarity_db: Dict[str, List[Tuple[str, float]]] = {}

    logger.info("1. Pre-generating character images...")
    char_images = {}
    ssim_images: List[np.ndarray] = []
    embeddings: List[np.ndarray] = []

    for char in tqdm(char_list, desc="Generating images"):
        img = render_text_with_effects(
            text=char,
            font_path=font_path,
            background_color=(255, 255, 255),
            font_size=24,
        )
        char_images[char] = img
        ssim_img = _preprocess_image_for_ssim(img, size=32)
        ssim_images.append(ssim_img)
        embeddings.append(_low_res_embedding(ssim_img, size=8))

    if not embeddings:
        save_json(similarity_db, db_path)
        logger.info("Database build complete (empty input).")
        return similarity_db

    emb_matrix = np.stack(embeddings, axis=0)

    # Calculate the similarity for each character using a cheap filter, then SSIM for candidates
    logger.info("\n2. Calculating similarity for character candidates...")
    for i, char1 in tqdm(list(enumerate(char_list)), desc="Comparing characters"):
        sims = emb_matrix @ emb_matrix[i]
        sims[i] = -1.0
        candidate_count = min(len(char_list) - 1, max(top_k * 3, top_k))
        if candidate_count <= 0:
            similarity_db[char1] = []
            continue

        candidate_idx = np.argpartition(-sims, candidate_count - 1)[:candidate_count]
        candidates: List[Tuple[str, float]] = []

        img1 = ssim_images[i]
        for j in candidate_idx:
            score, _ = ssim(img1, ssim_images[j], full=True)
            if score >= threshold:
                candidates.append((char_list[j], float(score)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        similarity_db[char1] = candidates[:top_k]

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

    entry = db[query_char]
    if isinstance(entry, dict):
        similar_items = sorted(
            entry.items(), key=lambda item: item[1], reverse=True
        )
        return similar_items[:top_n]

    if isinstance(entry, list):
        similar_items = []
        for item in entry:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                similar_items.append((item[0], item[1]))
        similar_items.sort(key=lambda item: item[1], reverse=True)
        return similar_items[:top_n]

    return []


# --- Step 3: Generating typos using the database ---


def generate_sentence_typos(
    texts: List[str], db: Dict[str, Any], typo_ratio: float = 0.15
) -> List[tuple[str, str]]:
    """
    Generates typos for a given list of texts based on character similarity.
    Typos are generated for a ratio of words defined by `typo_ratio`.

    Args:
        texts: A list of original strings to generate typos for.
        db: The character similarity database.
        typo_ratio: The ratio of words in which to generate typos.

    Returns:
        A list of tuples, each containing the original sentence and the sentence with typos.
    """
    generated_sentences_with_original = []
    for text in texts:
        words = text.split()
        if not words:
            generated_sentences_with_original.append((text, text))
            continue

        num_words_to_change = int(len(words) * typo_ratio)
        if num_words_to_change == 0 and typo_ratio > 0:
            num_words_to_change = 1

        # Ensure we don't try to change more words than exist.
        num_words_to_change = min(num_words_to_change, len(words))

        new_words = list(words)
        changes_made = 0
        max_passes = 3
        pass_count = 0

        while changes_made < num_words_to_change and pass_count < max_passes:
            indices = list(range(len(words)))
            random.shuffle(indices)
            for index in indices:
                if changes_made >= num_words_to_change:
                    break

                word = new_words[index]
                if not word or len(word) <= 1:
                    continue

                # Try multiple characters within the word to increase success rate.
                changed = False
                max_char_tries = min(5, len(word))
                for _ in range(max_char_tries):
                    char_index = random.randint(0, len(word) - 1)
                    original_char = word[char_index]
                    if original_char.isnumeric() or original_char.isspace():
                        continue

                    similar_chars = find_similar_chars(original_char, db, top_n=5)
                    if similar_chars:
                        similar_char, _ = random.choice(similar_chars)
                        word_list = list(word)
                        word_list[char_index] = similar_char
                        new_words[index] = "".join(word_list)
                        changed = True
                        break

                if changed:
                    changes_made += 1

            pass_count += 1

        generated_sentences_with_original.append((text, " ".join(new_words)))

    return generated_sentences_with_original


def generate_similar_chars_db(
    corpus_path="data/corpus.txt",
    db_path="data/char_similarity_db.json",
    font_path="/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    threshold: float = 0.6,
    top_k: int = 8,
):
    """
    Main function to generate and query the character similarity database.
    """
    # Read the character corpus and create a unique list of characters
    chars_text = read_txt(corpus_path) or ""
    chars = sorted(set(_iter_valid_chars(chars_text)))

    logger.info(f"Total unique characters: {len(chars)}")
    build_similarity_database(chars, font_path, db_path, threshold=threshold, top_k=top_k)


if __name__ == "__main__":
    generate_similar_chars_db()
