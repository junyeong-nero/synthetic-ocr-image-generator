import sys
from pathlib import Path

sys.path.insert(0, "src")

from ssim_calculator import run
from image_generator.table_generator import generate_table_images

if __name__ == "__main__":
    generate_table_images("data/corpus.txt", output_dir="data/images_table")
    # run(
    #     corpus_path="data/corpus.txt",
    # )
