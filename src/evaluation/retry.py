from typing import Any, Callable

from PIL import Image


def normalize_predictions(predictions: Any) -> list[str]:
    return [pred if isinstance(pred, str) else str(pred) for pred in list(predictions)]


def ensure_batch_size(predictions: list[str], expected_size: int) -> None:
    if len(predictions) != expected_size:
        raise RuntimeError(
            "Model returned mismatched batch size: "
            f"expected {expected_size}, got {len(predictions)}"
        )


def is_empty_prediction(prediction: Any) -> bool:
    if prediction is None:
        return True
    return isinstance(prediction, str) and prediction.strip() == ""


def empty_prediction_positions(predictions: list[str]) -> list[int]:
    return [index for index, prediction in enumerate(predictions) if is_empty_prediction(prediction)]


def prepare_retry_batch(
    empty_positions: list[int],
    batch_prompts: list[str],
    batch_images: list[Image.Image],
) -> tuple[list[str], list[Image.Image]]:
    retry_prompts = [batch_prompts[position] for position in empty_positions]
    retry_images = [batch_images[position] for position in empty_positions]
    return retry_prompts, retry_images


def finalize_retry_errors(predictions: list[str], errors: dict[int, str]) -> dict[int, str]:
    for position, prediction in enumerate(predictions):
        if is_empty_prediction(prediction):
            errors[position] = errors.get(position) or "Empty prediction after retries"
    return errors


def apply_retry_results(
    current_predictions: list[str],
    empty_positions: list[int],
    retried_predictions: list[str],
    errors: dict[int, str],
) -> bool:
    if len(retried_predictions) != len(empty_positions):
        for position in empty_positions:
            errors[position] = "Empty prediction retry returned mismatched batch size"
        return False
    for position, retry_prediction in zip(empty_positions, retried_predictions):
        current_predictions[position] = retry_prediction
    return True
