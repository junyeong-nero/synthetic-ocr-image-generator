import sys
from pathlib import Path

sys.path.insert(0, "src")
from image_generator.needle_in_a_haystack_generator import (
    generate_needle_in_a_haystack_images,
)


if __name__ == "__main__":
    # corpus directory (auto-generated from wikipedia)
    CORPUS_FILE_PATH = "data/corpus.txt"

    generate_needle_in_a_haystack_images(
        corpus_path=CORPUS_FILE_PATH,
        num_images=1000,
        output_dir="data/images_needle_in_a_haystack",
    )
