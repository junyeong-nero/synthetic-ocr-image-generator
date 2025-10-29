import sys
from pathlib import Path
import yaml
import logging

sys.path.insert(0, "src")
from pipeline import pipeline


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline(**config)
