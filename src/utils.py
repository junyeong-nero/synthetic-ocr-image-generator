import re
import json
import logging
import pandas as pd

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from datasets import (
    Dataset,
    Features,
    Value,
    Image as HFImage,
)
from huggingface_hub import HfFolder

logger = logging.getLogger(__name__)


def extract_tag(text: str, tag="char") -> list:
    """
    Extracts content from a specified tag in a string.

    Args:
        text (str): The string to search within.
        tag (str): The tag to extract content from.

    Returns:
        list: A list of strings found within the specified tags.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text)
    return matches


def read_json(file_path):
    """
    Reads a JSON file and returns a Python object.

    Args:
        file_path (str): The path to the JSON file to read.

    Returns:
        dict or list: A Python object containing the contents of the JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error: File '{file_path}' is not a valid JSON format.")
        return None


import json


def save_json(data, file_path):
    """
    Saves a Python object to a JSON file.

    Args:
        data (dict or list): The Python object to save.
        file_path (str): The path to the JSON file to save.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data successfully saved to '{file_path}'.")
    except Exception as e:
        logger.error(f"An error occurred while saving the file: {e}")


def read_txt(file_path):
    """
    Reads all lines from a text file and returns them as a single string.

    Args:
        file_path (str): The path to the text file to read.

    Returns:
        str: A string containing the entire content of the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        return None


def save_txt(file_path, text):
    """
    Writes a string to a text file.

    Args:
        text (str): The string to write to the file.
        file_path (str): The path to the text file to save.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Data successfully saved to '{file_path}'.")
    except Exception as e:
        logger.error(f"An error occurred while saving the file: {e}")


def upload_subset_to_hub(repo_id: str, subset_dir: Path, config_name: str):
    """
    Uploads data from a specified directory to a specific config (subset) on the Hub.
    Dynamically detects all fields in metadata.jsonl to use as columns.

    Args:
        repo_id (str): The Hugging Face repository ID (e.g., 'user/repo-name').
        subset_dir (Path): The directory path containing images and metadata.jsonl.
        config_name (str): The config name for the dataset (e.g., 'sentence_typos').
    """
    logger.info(f"\n▶ Starting upload of subset '{config_name}' to '{repo_id}'...")

    try:
        if HfFolder.get_token() is None:
            raise ConnectionError(
                "Hugging Face login is required. Please run 'huggingface-cli login'."
            )

        metadata_path = subset_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"'{metadata_path}' not found. Aborting upload."
            )

        # --- [Modified Part 1] ---
        # 1. Read the first line of metadata.jsonl to dynamically determine columns and data types.
        feature_dict = {}
        column_names = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                logger.warning(
                    f"'{metadata_path}' is empty. Skipping upload."
                )
                return

            sample_data = json.loads(first_line)
            if "file_name" not in sample_data:
                raise KeyError("Required key 'file_name' not found in 'metadata.jsonl'.")

            # 'file_name' is always treated as the 'image' column
            feature_dict["image"] = HFImage()
            column_names.append("image")

            # Check types for the remaining keys and construct Features
            for key, value in sample_data.items():
                if key == "file_name":
                    continue

                value_type = type(value)
                if value_type is bool:
                    feature_dict[key] = Value("bool")
                elif value_type is Tuple or value_type is List:
                    feature_dict[key] = Value("string")
                elif value_type is int:
                    feature_dict[key] = Value("int64")
                elif value_type is float:
                    feature_dict[key] = Value("float32")
                elif value_type is str:
                    feature_dict[key] = Value("string")
                else:
                    # Warn and treat unsupported types as strings
                    logger.warning(
                        f"Unsupported type for '{key}' ({value_type}). Treating as string."
                    )
                    feature_dict[key] = Value("string")

                column_names.append(key)

        features = Features(feature_dict)
        logger.info(f"  Detected columns and types: {features}")
        # --- [Modification Complete] ---

        # 2. Collect all data into a list of dictionaries
        all_data: List[Dict[str, Any]] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                record = {"image": str(data["file_name"])}
                for key in feature_dict.keys():
                    if key != "image":
                        record[key] = data.get(key)

                all_data.append(record)

        if not all_data:
            logger.warning("No valid data to process. Aborting upload.")
            return

        logger.info(
            f"  '{config_name}' subset: Found {len(all_data):,} valid data entries."
        )

        # 3. Create a Hugging Face Dataset object
        df = pd.DataFrame(all_data)
        dataset = Dataset.from_pandas(df, features=features)

        # Upload to the Hugging Face Hub (specifying config_name)
        dataset.push_to_hub(repo_id, config_name=config_name)

        logger.info(f"✔ Subset '{config_name}' uploaded successfully!")

    except (ConnectionError, FileNotFoundError, KeyError) as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(
            f"Error: An unexpected error occurred while uploading subset '{config_name}': {e}",
            exc_info=True,
        )
