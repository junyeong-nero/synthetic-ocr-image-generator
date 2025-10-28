import sys
from pathlib import Path
import yaml

sys.path.insert(0, "src")
from pipeline import pipeline


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline(
        corpus_path=config["corpus_file_path"],
        repo_id=config["hf_repo_id"],
        num_word_images=config["image_generation"]["num_word_images"],
        num_doc_images=config["image_generation"]["num_document_images"],
        num_table_images=config["image_generation"]["num_table_images"],
        num_needle_images=config["image_generation"]["num_needle_in_a_haystack_images"],
        output_dir=config["output_dir"],
    )
