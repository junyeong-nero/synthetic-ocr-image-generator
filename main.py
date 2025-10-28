import sys
from pathlib import Path

sys.path.insert(0, "src")
from pipeline import pipeline


if __name__ == "__main__":

    # corpus directory (auto-generated from wikipedia)
    CORPUS_FILE_PATH = "data/corpus.txt"

    # repository-id
    HF_REPO_ID = "junyeong-nero/synthetic-ocr-bench"

    # number of images
    NUM_WORD_IMAGES = 1000
    NUM_DOCUMENT_IMAGES = 1000
    NUM_TABLE_IMAGES = 1000

    # output directory
    SINGLE_LINE_OUTPUT_DIR = "data/images_word"
    DOC_OUTPUT_DIR = "data/images_document"
    TABLE_OUTPUT_DIR = "data/images_table"

    pipeline(
        corpus_path=CORPUS_FILE_PATH,
        repo_id=HF_REPO_ID,
        num_word_images=NUM_WORD_IMAGES,
        num_doc_images=NUM_DOCUMENT_IMAGES,
        num_table_images=NUM_TABLE_IMAGES,
        word_output_dir=SINGLE_LINE_OUTPUT_DIR,
        table_output_dir=TABLE_OUTPUT_DIR,
        doc_output_dir=DOC_OUTPUT_DIR,
    )
