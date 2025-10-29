import sys
from pathlib import Path
import yaml

sys.path.insert(0, "src")
from pipeline import pipeline


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline(**config)
