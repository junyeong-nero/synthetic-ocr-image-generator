import json
from pathlib import Path
from typing import Any, Callable, Sequence

from evaluation.types import InferenceResult


def load_batch_info(batch_info_path: Path) -> dict[str, Any]:
    try:
        with open(batch_info_path, "r", encoding="utf-8") as file:
            batch_info = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"Ignoring invalid batch metadata at {batch_info_path}: {exc}")
        return {}

    if not isinstance(batch_info, dict):
        print(f"Ignoring invalid batch metadata at {batch_info_path}: expected object")
        return {}
    return batch_info


def load_batch_errors(batch_dir: Path) -> dict[str, str]:
    error_path = batch_dir / "batch_errors.json"
    if not error_path.exists():
        return {}

    try:
        with open(error_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"Ignoring invalid batch error file at {error_path}: {exc}")
        return {}

    if not isinstance(data, dict):
        print(f"Ignoring invalid batch error format at {error_path}: expected object")
        return {}
    return {str(key): str(value) for key, value in data.items()}


def build_batch_results(
    prediction_map: dict[str, str],
    indices: list[int],
    ground_truths: Sequence[Any],
    *,
    latency_ms: float = 0,
    error_map: dict[str, str] | None = None,
    is_empty_prediction: Callable[[Any], bool],
) -> list[InferenceResult]:
    results: list[InferenceResult] = []
    for idx in indices:
        custom_id = str(idx)
        prediction = prediction_map.get(custom_id)
        error = None
        if prediction is None:
            prediction = ""
            error = "Missing batch response"
        elif is_empty_prediction(prediction):
            error = "Empty prediction in batch response"
        if error_map and custom_id in error_map:
            error = error_map[custom_id]

        results.append(
            InferenceResult(
                index=idx,
                prediction=prediction,
                ground_truth=ground_truths[idx],
                latency_ms=latency_ms,
                error=error,
            )
        )
    return results
