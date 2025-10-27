import sys
from pathlib import Path

sys.path.insert(0, "src")

from ssim_calculator import run

if __name__ == "__main__":
    run(
        corpus_path="data/korean_char_corpus.txt",
    )
