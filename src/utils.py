import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MARKDOWN_AST_PARSER = None


def _get_markdown_ast_parser():
    global _MARKDOWN_AST_PARSER
    if _MARKDOWN_AST_PARSER is not None:
        return _MARKDOWN_AST_PARSER

    import mistune

    plugin_candidates = [
        ["table", "task_lists", "strikethrough"],
        ["table", "strikethrough"],
        ["table"],
        [],
    ]
    for plugins in plugin_candidates:
        try:
            _MARKDOWN_AST_PARSER = mistune.create_markdown(
                renderer="ast",
                plugins=plugins,
            )
            return _MARKDOWN_AST_PARSER
        except Exception:
            continue

    raise RuntimeError("Failed to initialize Mistune markdown AST parser")


def markdown_to_json_ast(markdown_text: str) -> List[Dict[str, Any]]:
    parser = _get_markdown_ast_parser()
    tokens = parser(markdown_text or "")
    return tokens if isinstance(tokens, list) else [{"type": "raw", "raw": markdown_text or ""}]


def extract_tag(text: str, tag="char") -> list:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return re.findall(pattern, text)


def read_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error: File '{file_path}' is not a valid JSON format.")
        return None


def save_json(data, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logger.info(f"Data successfully saved to '{file_path}'.")
    except Exception as exc:
        logger.error(f"An error occurred while saving the file: {exc}")


def read_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found.")
        return None


def save_txt(file_path, text):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
        logger.info(f"Data successfully saved to '{file_path}'.")
    except Exception as exc:
        logger.error(f"An error occurred while saving the file: {exc}")


def upload_subset_to_hub(
    repo_id: str,
    subset_dir: Path,
    config_name: str,
    split: str = "train",
    reuse_existing_schema: bool = False,
):
    from generation.hub_dataset import upload_subset_to_hub as _upload_subset_to_hub

    return _upload_subset_to_hub(
        repo_id=repo_id,
        subset_dir=subset_dir,
        config_name=config_name,
        split=split,
        reuse_existing_schema=reuse_existing_schema,
    )


def upload_dataset_readme_to_hub(
    repo_id: str,
    readme_content: str,
    commit_message: str = "docs: update dataset card",
) -> None:
    from generation.hub_dataset import (
        upload_dataset_readme_to_hub as _upload_dataset_readme_to_hub,
    )

    return _upload_dataset_readme_to_hub(
        repo_id=repo_id,
        readme_content=readme_content,
        commit_message=commit_message,
    )