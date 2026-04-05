import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from huggingface_hub import HfApi, whoami

from src.utils import markdown_to_json_ast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datasets import Features

_FEATURE_TYPE_MAPPING = {
    bool: "bool",
    int: "int64",
    float: "float32",
    str: "string",
    tuple: "string",
    list: "string",
    dict: "string",
}


def upload_subset_to_hub(
    repo_id: str,
    config_name: str,
    subset_dir: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    split: str = "train",
    reuse_existing_schema: bool = False,
    selected_indices: Optional[set[int]] = None,
    max_shard_size: str = "256MB",
) -> None:
    logger.info(
        f"\n▶ Starting upload of subset '{config_name}' split '{split}' to '{repo_id}'..."
    )

    try:
        _ensure_hf_login()
        resolved_metadata_path = _resolve_metadata_path(
            subset_dir=subset_dir,
            metadata_path=metadata_path,
        )
        sample_row = _read_first_normalized_metadata_row(
            resolved_metadata_path,
            selected_indices=selected_indices,
        )
        if sample_row is None:
            logger.warning("No valid data to process. Aborting upload.")
            return

        features = _build_features(_infer_feature_dict(sample_row))
        if reuse_existing_schema:
            existing_features = _get_existing_features(repo_id, config_name, split)
            if existing_features is not None:
                features = existing_features
                logger.info(f"  Reusing existing Hub feature schema: {features}")
            else:
                logger.info(f"  Detected columns and types: {features}")
        else:
            logger.info(f"  Using inferred local feature schema: {features}")

        entry_count = _count_selected_rows(
            resolved_metadata_path,
            selected_indices=selected_indices,
        )
        logger.info(
            f"  '{config_name}' subset ({split}): Found {entry_count:,} valid data entries."
        )

        dataset_cls = _get_dataset_class()
        dataset = dataset_cls.from_generator(
            _iter_upload_records,
            features=features,
            gen_kwargs={
                "metadata_path": resolved_metadata_path,
                "features": features,
                "selected_indices": selected_indices,
            },
        )
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            split=split,
            max_shard_size=max_shard_size,
        )

        logger.info(f"✔ Subset '{config_name}' uploaded successfully!")

    except (ConnectionError, FileNotFoundError, KeyError) as exc:
        logger.error(f"Error: {exc}")
    except Exception as exc:
        logger.error(
            f"Error: An unexpected error occurred while uploading subset '{config_name}': {exc}",
            exc_info=True,
        )


def upload_dataset_readme_to_hub(
    repo_id: str,
    readme_content: str,
    commit_message: str = "docs: update dataset card",
) -> None:
    logger.info("\n▶ Uploading dataset README.md to '%s'...", repo_id)
    try:
        _ensure_hf_login()
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
        logger.info("✔ README.md uploaded successfully!")
    except Exception as exc:
        logger.error("Failed to upload README.md: %s", exc, exc_info=True)


def _to_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw": value}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    if value is None:
        return {}
    return {"value": value}


def _json_block(payload: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _normalize_gt_fields(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)

    def _drop_excluded_keys() -> None:
        normalized.pop("format", None)
        normalized.pop("similarity_db_path", None)
        normalized.pop("original_markdown", None)
        normalized.pop("similar_char_ratio", None)
        normalized.pop("add_noise", None)
        normalized.pop("add_blur", None)

    def _drop_legacy_gt_keys() -> None:
        normalized.pop("ground_truth", None)
        normalized.pop("markdown", None)
        normalized.pop("json", None)

    gt_markdown = normalized.get("GT_markdown")
    gt_json = normalized.get("GT_json")
    if gt_markdown is not None and gt_json is not None:
        _drop_legacy_gt_keys()
        _drop_excluded_keys()
        return normalized

    if gt_markdown is not None and gt_json is None:
        markdown_text = str(gt_markdown)
        normalized["GT_markdown"] = markdown_text
        normalized["GT_json"] = markdown_to_json_ast(markdown_text)
        _drop_legacy_gt_keys()
        _drop_excluded_keys()
        return normalized

    if "typo_text" in normalized:
        text = str(normalized.get("typo_text", ""))
        payload = {
            "text": text,
            "original_text": str(normalized.get("original_text", "")),
        }
        normalized["GT_markdown"] = text
        normalized["GT_json"] = payload
        _drop_excluded_keys()
        return normalized

    if "html" in normalized and "json" in normalized:
        table_json = _to_json_object(normalized.get("json"))
        table_markdown = str(normalized.get("html", "")).strip()
        if not table_markdown:
            table_markdown = _json_block(table_json)
        normalized["GT_markdown"] = table_markdown
        normalized["GT_json"] = table_json
        _drop_legacy_gt_keys()
        _drop_excluded_keys()
        return normalized

    if "ground_truth" in normalized:
        payload = _to_json_object(normalized.get("ground_truth"))
        normalized["GT_markdown"] = _json_block(payload)
        normalized["GT_json"] = payload
        _drop_legacy_gt_keys()
        _drop_excluded_keys()
        return normalized

    if "markdown" in normalized:
        markdown_text = str(normalized.get("markdown", ""))
        normalized["GT_markdown"] = markdown_text
        normalized["GT_json"] = markdown_to_json_ast(markdown_text)
        _drop_legacy_gt_keys()
        _drop_excluded_keys()
        return normalized

    if "json" in normalized:
        payload = _to_json_object(normalized.get("json"))
        normalized["GT_markdown"] = _json_block(payload)
        normalized["GT_json"] = payload
        _drop_legacy_gt_keys()
        _drop_excluded_keys()
        return normalized

    normalized["GT_markdown"] = ""
    normalized["GT_json"] = {}
    _drop_excluded_keys()
    return normalized


def _ensure_hf_login() -> None:
    try:
        whoami()
    except Exception as exc:
        raise ConnectionError(
            "Hugging Face login is required. Please run 'huggingface-cli login'."
        ) from exc


def _resolve_metadata_path(
    *,
    subset_dir: Optional[Path],
    metadata_path: Optional[Path],
) -> Path:
    if metadata_path is None:
        if subset_dir is None:
            raise FileNotFoundError("Either subset_dir or metadata_path must be provided.")
        metadata_path = subset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"'{metadata_path}' not found. Aborting upload.")
    return metadata_path


def _iter_normalized_metadata_rows(metadata_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(metadata_path, "r", encoding="utf-8") as file:
        for row_index, line in enumerate(file):
            payload = line.strip()
            if not payload:
                continue
            try:
                raw = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in metadata.jsonl at line {row_index + 1}"
                ) from exc
            yield row_index, _normalize_gt_fields(raw)


def _read_first_normalized_metadata_row(
    metadata_path: Path,
    *,
    selected_indices: Optional[set[int]],
) -> Optional[dict[str, Any]]:
    for row_index, row in _iter_normalized_metadata_rows(metadata_path):
        if selected_indices is not None and row_index not in selected_indices:
            continue
        return row
    return None


def _infer_feature_dict(sample_data: dict[str, Any]) -> dict[str, Any]:
    if "file_name" not in sample_data:
        raise KeyError("Required key 'file_name' not found in 'metadata.jsonl'.")

    feature_dict: dict[str, Any] = {"image": _get_hf_image_feature()}
    for key, value in sample_data.items():
        if key == "file_name":
            continue

        value_type = type(value)
        hf_type = _FEATURE_TYPE_MAPPING.get(value_type, "string")
        if value_type not in _FEATURE_TYPE_MAPPING:
            logger.warning(
                "Unsupported type for '%s' (%s). Treating as string.",
                key,
                value_type,
            )
        feature_dict[key] = _get_hf_value_feature(hf_type)
    return feature_dict


def _iter_upload_records(
    metadata_path: Path,
    features: "Features",
    selected_indices: Optional[set[int]] = None,
) -> Iterator[dict[str, Any]]:
    for row_index, data in _iter_normalized_metadata_rows(metadata_path):
        if selected_indices is not None and row_index not in selected_indices:
            continue
        yield _build_upload_record(data, features, row_index=row_index)


def _build_upload_record(
    data: dict[str, Any],
    features: "Features",
    *,
    row_index: int,
) -> dict[str, Any]:
    file_name = data.get("file_name")
    if not file_name:
        raise KeyError(f"Required key 'file_name' missing at metadata row {row_index + 1}.")

    record: dict[str, Any] = {"image": str(file_name)}
    for key, feature in features.items():
        if key == "image":
            continue
        value = data.get(key)
        if value is None:
            value = _default_value_for_feature(key, feature, data)
        record[key] = _serialize_feature_value(value)
    return record


def _serialize_feature_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _get_existing_features(repo_id: str, config_name: str, split: str) -> Optional["Features"]:
    split_candidates: list[str] = []
    for name in [split, "train", "validation", "test"]:
        if name and name not in split_candidates:
            split_candidates.append(name)

    load_dataset = _get_load_dataset()
    for split_name in split_candidates:
        try:
            sample = load_dataset(repo_id, name=config_name, split=f"{split_name}[:1]")
        except Exception:
            continue
        if sample is not None and sample.features is not None:
            return sample.features
    return None


def _count_selected_rows(
    metadata_path: Path,
    *,
    selected_indices: Optional[set[int]],
) -> int:
    if selected_indices is None:
        count = 0
        with open(metadata_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    count += 1
        return count
    return len(selected_indices)


def _default_value_for_feature(key: str, feature: Any, record: dict[str, Any]) -> Any:
    if key == "ground_truth":
        return str(record.get("GT_markdown", record.get("markdown", "")))
    if key == "markdown":
        return str(record.get("GT_markdown", record.get("markdown", "")))
    if key == "json":
        value = record.get("GT_json", record.get("json", {}))
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    if key == "GT_markdown":
        return str(record.get("GT_markdown", record.get("markdown", "")))
    if key == "GT_json":
        return json.dumps(record.get("GT_json", {}), ensure_ascii=False)
    if key == "elements_count":
        source = str(record.get("GT_markdown", record.get("markdown", "")))
        return max(1, source.count("\n") + 1)
    if key == "font_size":
        return 14

    dtype = getattr(feature, "dtype", "string")
    if dtype.startswith("int"):
        return 0
    if dtype.startswith("float"):
        return 0.0
    if dtype == "bool":
        return False
    return ""


def _get_dataset_class():
    from datasets import Dataset

    return Dataset


def _get_load_dataset():
    from datasets import load_dataset

    return load_dataset


def _build_features(feature_dict: dict[str, Any]) -> "Features":
    from datasets import Features

    return Features(feature_dict)


def _get_hf_image_feature():
    from datasets import Image as HFImage

    return HFImage()


def _get_hf_value_feature(dtype: str):
    from datasets import Value

    return Value(dtype)
