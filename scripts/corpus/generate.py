#!/usr/bin/env python3

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from corpus_llm import run_cli
from env_utils import load_env_file

load_env_file()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    sys.exit(asyncio.run(run_cli()))