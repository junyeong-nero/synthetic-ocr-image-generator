import re
import json
import logging
import os
import platform
import random
import sys
from importlib import metadata as importlib_metadata

import pandas as pd

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from datasets import (
    Dataset,
    Features,
    Value,
    Image as HFImage,
)
from huggingface_hub import whoami

logger = logging.getLogger(__name__)


def extract_tag(text: str, tag="char") -> list:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return re.findall(pattern, text)


def read_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error: File '{file_path}' is not a valid JSON format.")
        return None


def save_json(data, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data successfully saved to '{file_path}'.")
    except Exception as e:
        logger.error(f"An error occurred while saving the file: {e}")


def read_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        return None


def save_txt(file_path, text):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Data successfully saved to '{file_path}'.")
    except Exception as e:
        logger.error(f"An error occurred while saving the file: {e}")


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _get_package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def get_environment_metadata() -> Dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    env["packages"] = {
        "torch": _get_package_version("torch"),
        "transformers": _get_package_version("transformers"),
        "datasets": _get_package_version("datasets"),
        "numpy": _get_package_version("numpy"),
        "pandas": _get_package_version("pandas"),
        "uv": _get_package_version("uv"),
    }

    try:
        import torch

        env["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None))
            and torch.backends.mps.is_available(),
        }
    except Exception:
        env["torch"] = {
            "cuda_available": False,
            "mps_available": False,
        }

    return env


def upload_subset_to_hub(repo_id: str, subset_dir: Path, config_name: str):
    logger.info(f"\n▶ Starting upload of subset '{config_name}' to '{repo_id}'...")

    try:
        try:
            whoami()
        except Exception:
            raise ConnectionError(
                "Hugging Face login is required. Please run 'huggingface-cli login'."
            )

        metadata_path = subset_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"'{metadata_path}' not found. Aborting upload.")

        feature_dict, column_names = _parse_metadata_schema(metadata_path)
        if feature_dict is None:
            return

        features = Features(feature_dict)
        logger.info(f"  Detected columns and types: {features}")

        all_data = _load_metadata_records(metadata_path, feature_dict)
        if not all_data:
            logger.warning("No valid data to process. Aborting upload.")
            return

        logger.info(f"  '{config_name}' subset: Found {len(all_data):,} valid data entries.")

        df = pd.DataFrame(all_data)
        dataset = Dataset.from_pandas(df, features=features)
        dataset.push_to_hub(repo_id, config_name=config_name)

        logger.info(f"✔ Subset '{config_name}' uploaded successfully!")

    except (ConnectionError, FileNotFoundError, KeyError) as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(
            f"Error: An unexpected error occurred while uploading subset '{config_name}': {e}",
            exc_info=True,
        )


def _parse_metadata_schema(metadata_path: Path) -> Tuple[Optional[Dict], List[str]]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            logger.warning(f"'{metadata_path}' is empty. Skipping upload.")
            return None, []

        sample_data = json.loads(first_line)
        if "file_name" not in sample_data:
            raise KeyError("Required key 'file_name' not found in 'metadata.jsonl'.")

        feature_dict: Dict[str, Any] = {"image": HFImage()}
        column_names = ["image"]

        type_mapping = {
            bool: "bool",
            int: "int64",
            float: "float32",
            str: "string",
            tuple: "string",
            list: "string",
        }

        for key, value in sample_data.items():
            if key == "file_name":
                continue

            value_type = type(value)
            hf_type = type_mapping.get(value_type, "string")
            if value_type not in type_mapping:
                logger.warning(f"Unsupported type for '{key}' ({value_type}). Treating as string.")
            feature_dict[key] = Value(hf_type)
            column_names.append(key)

        return feature_dict, column_names


def _load_metadata_records(metadata_path: Path, feature_dict: Dict) -> List[Dict[str, Any]]:
    all_data = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            record = {"image": str(data["file_name"])}
            for key in feature_dict.keys():
                if key != "image":
                    value = data.get(key)
                    # Convert dict/list/tuple to JSON string for HF Dataset compatibility
                    if isinstance(value, (dict, list, tuple)):
                        value = json.dumps(value, ensure_ascii=False)
                    record[key] = value
            all_data.append(record)
    return all_data
