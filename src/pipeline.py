from pathlib import Path

from corpus_generator import create_corpus_from_wiki
from image_generator import (
    generate_document,
    generate_text,
    generate_table,
    generate_needle,
)

from character_similarity import generate_similar_chars_db
from utils import upload_subset_to_hub


def pipeline(
    corpus_path: str,
    repo_id: str,
    num_word_images: int,
    num_doc_images: int,
    num_table_images: int,
    num_needle_images: int,
    output_dir: str,
    lang: str,
    **kwargs,
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
        num_needle_images (int): The number of "needle in a haystack" images to generate.
        output_dir (str): The directory to save all generated content.
        lang (str): The language code for corpus generation (e.g., "ko", "en").
    """
    # --- Path Initialization using pathlib ---
    # 기본 출력 디렉토리를 Path 객체로 변환
    base_output_path = Path(output_dir)

    # 각 데이터셋 유형별 출력 디렉토리 경로 정의
    word_output_path = base_output_path / "images_word"
    table_output_path = base_output_path / "images_table"
    doc_output_path = base_output_path / "images_document"
    needle_output_path = base_output_path / "images_needle_in_a_haystack"
    corpus_file_path = Path(corpus_path)

    # --- Directory Creation ---
    # 정의된 모든 출력 디렉토리와 코퍼스 디렉토리를 미리 생성
    print(f"Ensuring output directories exist in '{base_output_path}'...")
    word_output_path.mkdir(parents=True, exist_ok=True)
    table_output_path.mkdir(parents=True, exist_ok=True)
    doc_output_path.mkdir(parents=True, exist_ok=True)
    needle_output_path.mkdir(parents=True, exist_ok=True)
    if corpus_file_path.parent:
        corpus_file_path.parent.mkdir(parents=True, exist_ok=True)

    # If the corpus file doesn't exist, create it from Wikipedia.
    if not corpus_file_path.exists():
        create_corpus_from_wiki(
            output_path=str(corpus_file_path), lang=lang, num_sentences=5000
        )

    # STEP 1: Generate single sentence images
    # 생성 함수에 Path 객체를 문자열로 변환하여 전달
    generated_word_dir = generate_text(
        corpus_path=str(corpus_file_path),
        num_images=num_word_images,
        output_dir=str(word_output_path),
    )

    if generated_word_dir is None:
        print("Failed to generate single sentence images. Aborting pipeline.")
        return

    # STEP 2: Generate document images
    generated_doc_dir = generate_document(
        corpus_path=str(corpus_file_path),
        num_images=num_doc_images,
        output_dir=str(doc_output_path),
    )

    if generated_doc_dir is None:
        print("Failed to generate document images. Aborting pipeline.")
        return

    # STEP 3: Generate table images
    generated_table_dir = generate_table(
        corpus_path=str(corpus_file_path),
        num_images=num_table_images,
        output_dir=str(table_output_path),
    )

    if generated_table_dir is None:
        print("Failed to generate table images. Aborting pipeline.")
        return

    # STEP 4: Generate character similarity database
    db_path = base_output_path / f"char_similarity_db_{lang}.json"
    generate_similar_chars_db(
        corpus_path=str(corpus_file_path),
        db_path=str(db_path),
    )

    # STEP 5: Generate "needle in a haystack" images
    generated_needle_dir = generate_needle(
        corpus_path=str(corpus_file_path),
        db_path=db_path,
        num_images=num_needle_images,
        output_dir=str(needle_output_path),
    )

    if generated_needle_dir is None:
        print("Failed to generate needle in a haystack images. Aborting pipeline.")
        return

    # STEP 6: Upload all generated datasets to the Hugging Face Hub
    print(" Starting Upload to Hugging Face Hub ".center(50, "-"))
    upload_subset_to_hub(generated_word_dir, repo_id, config_name="word")
    upload_subset_to_hub(generated_doc_dir, repo_id, config_name="document")
    upload_subset_to_hub(generated_table_dir, repo_id, config_name="table")
    upload_subset_to_hub(
        generated_needle_dir, repo_id, config_name="needle-in-a-haystack"
    )

    print(" All tasks completed! ".center(50, "="))
    print(f"Check your dataset on the Hub: https://huggingface.co/datasets/{repo_id}")
