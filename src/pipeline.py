from pathlib import Path

from corpus_generator import create_corpus_from_wiki
from image_generator import (
    generate_document,
    generate_text,
    generate_table,
    generate_needle,
)
from utils import upload_subset_to_hub


def pipeline(
    corpus_path: str,
    repo_id: str,
    num_word_images: int,
    num_doc_images: int,
    num_table_images: int,
    num_needle_images: int,
    output_dir: str,
):
    """
    Executes the full data generation and upload pipeline.

    This function orchestrates the process of:
    1. Creating a text corpus if it doesn't exist.
    2. Generating synthetic images of words, documents, and tables.
    3. Uploading the generated datasets to the Hugging Face Hub.

    Args:
        corpus_path (str): Path to the text corpus file.
        repo_id (str): The Hugging Face Hub repository ID (e.g., "username/dataset-name").
        num_word_images (int): The number of word images to generate.
        num_doc_images (int): The number of document images to generate.
        num_table_images (int): The number of table images to generate.
        word_output_dir (str): The directory to save generated word images.
        doc_output_dir (str): The directory to save generated document images.
        table_output_dir (str): The directory to save generated table images.
    """

    word_output_dir = output_dir + "/images_word"
    table_output_dir = output_dir + "/images_table"
    doc_output_dir = output_dir + "/images_document"
    needle_output_dir = output_dir + "/images_needle_in_a_haystack"

    # If the corpus file doesn't exist, create it from Wikipedia.
    corpus_dir = Path(corpus_path)
    if not corpus_dir.exists():
        create_corpus_from_wiki(output_path=corpus_path, num_sentences=5000)

    # STEP 1: Generate single sentence images
    word_output_dir = generate_text(
        corpus_path=corpus_path,
        num_images=num_word_images,
        output_dir=word_output_dir,
    )

    if word_output_dir is None:
        print("Failed to generate single sentence images. Aborting pipeline.")
        return

    # STEP 2: Generate document images
    doc_output_dir = generate_document(
        corpus_path=corpus_path, num_images=num_doc_images, output_dir=doc_output_dir
    )

    if doc_output_dir is None:
        print("Failed to generate document images. Aborting pipeline.")
        return

    # STEP 3: Generate table images
    table_output_dir = generate_table(
        corpus_path=corpus_path,
        num_images=num_table_images,
        output_dir=table_output_dir,
    )

    if table_output_dir is None:
        print("Failed to generate table images. Aborting pipeline.")
        return

    needle_output_dir = generate_needle(
        corpus_path=corpus_path,
        num_images=num_needle_images,
        output_dir=needle_output_dir,
    )

    if needle_output_dir is None:
        print("Failed to generate needle in a haystack images. Aborting pipeline.")
        return

    # STEP 4: Upload all generated datasets to the Hugging Face Hub
    print(" Starting Upload to Hugging Face Hub ".center(50, "-"))
    upload_subset_to_hub(word_output_dir, repo_id, config_name="word")
    upload_subset_to_hub(doc_output_dir, repo_id, config_name="document")
    upload_subset_to_hub(table_output_dir, repo_id, config_name="table")
    upload_subset_to_hub(needle_output_dir, repo_id, config_name="needle-in-a-haystack")

    print(" All tasks completed! ".center(50, "="))
    print(f"Check your dataset on the Hub: https://huggingface.co/datasets/{repo_id}")
