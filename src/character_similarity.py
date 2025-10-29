import os
import cv2
import numpy as np

from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from utils import save_json, read_json, read_txt
from image_generator.basic_generator import _generate_text_image


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
    print("Starting to build the similarity database...")
    similarity_db = {}

    # Pre-generate character images to avoid redundant generation
    print("1. Pre-generating character images...")
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
    print("\n2. Calculating similarity for character pairs...")
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
    print("Database build complete.")
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

    print(f"Total unique characters: {len(chars)}")
    build_similarity_database(chars, font_path, db_path, threshold=0.6)


if __name__ == "__main__":
    generate_similar_chars_db()
