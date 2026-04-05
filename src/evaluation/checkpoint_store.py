import json
from pathlib import Path

from src.evaluation.checkpoint import build_checkpoint_context, resolve_checkpoint_path
from src.evaluation.types import RunnerState


def load_or_create_state(config, output_dir: Path, checkpoint_path: Path) -> RunnerState:
    expected_context = build_checkpoint_context(config)
    checkpoint_to_load = resolve_checkpoint_path(output_dir) or checkpoint_path

    if config.resume_from_checkpoint and checkpoint_to_load.exists():
        try:
            state = RunnerState.load(checkpoint_to_load)
            if state.context and state.context != expected_context:
                print(
                    "Checkpoint context mismatch; ignoring existing checkpoint and "
                    "starting fresh."
                )
                return RunnerState(context=expected_context)
            state.context = expected_context
            if state.completed:
                print(f"Resuming from checkpoint: {len(state.completed)} samples completed")
            return state
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            corrupt_path = checkpoint_to_load.with_suffix(".corrupt.json")
            try:
                checkpoint_to_load.replace(corrupt_path)
                print(
                    f"Failed to load checkpoint: {e}. Corrupt checkpoint moved to "
                    f"{corrupt_path}. Starting fresh."
                )
            except OSError:
                print(f"Failed to load checkpoint: {e}. Starting fresh.")

    return RunnerState(context=expected_context)


def save_checkpoint(state: RunnerState, checkpoint_path: Path) -> None:
    state.save(checkpoint_path)
