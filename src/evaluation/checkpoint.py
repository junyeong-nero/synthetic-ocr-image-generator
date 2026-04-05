from pathlib import Path

from src.evaluation.config import EvaluationConfig


def build_checkpoint_context(config: EvaluationConfig) -> dict[str, str]:
    return {
        "dataset_id": config.dataset_id,
        "split": config.split,
        "model_id": config.model.model_id,
        "backend": config.model.backend.value,
    }


def resolve_checkpoint_path(output_dir: Path) -> Path | None:
    checkpoint_path = output_dir / "checkpoints.json"
    if checkpoint_path.exists():
        return checkpoint_path

    legacy_path = output_dir / "checkpoint.json"
    if legacy_path.exists():
        return legacy_path

    return None
