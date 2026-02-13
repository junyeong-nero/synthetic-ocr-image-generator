"""Evaluation runner with checkpointing support."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from tqdm import tqdm

from evaluation.config import EvaluationConfig
from evaluation.types import InferenceResult, RunnerState
from models.base import VLMModel


class EvaluationRunner:
    """
    Batch inference runner with checkpointing support.

    Handles batch processing, progress tracking, and resumable evaluation.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        model: VLMModel,
    ):
        self.config = config
        self.model = model

        # Setup output directory and checkpoint
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoints.json"
        self.legacy_checkpoint_path = self.output_dir / "checkpoint.json"

        # Load or create state
        self.state = self._load_or_create_state()
        self._completed_indices = set(self.state.completed)
        self._empty_prediction_max_retries = 1
        self._empty_prediction_retry_backoff_seconds = 0.5

    @staticmethod
    def _normalize_predictions(predictions: Any) -> list[str]:
        return [pred if isinstance(pred, str) else str(pred) for pred in list(predictions)]

    @staticmethod
    def _ensure_batch_size(predictions: list[str], expected_size: int) -> None:
        if len(predictions) != expected_size:
            raise RuntimeError(
                "Model returned mismatched batch size: "
                f"expected {expected_size}, got {len(predictions)}"
            )

    def _remaining_indices(self, total_samples: int) -> list[int]:
        return [i for i in range(total_samples) if i not in self._completed_indices]

    def _append_checkpoint_result(self, result: InferenceResult) -> None:
        if result.index in self._completed_indices:
            return
        self._completed_indices.add(result.index)
        self.state.completed.append(result.index)
        self.state.results.append(result.to_dict())

    @staticmethod
    def _is_empty_prediction(prediction: Any) -> bool:
        if prediction is None:
            return True
        return isinstance(prediction, str) and prediction.strip() == ""

    async def _retry_empty_predictions_async(
        self,
        predictions: list[str],
        batch_prompts: list[str],
        batch_images: list[Image.Image],
    ) -> tuple[list[str], dict[int, str]]:
        errors: dict[int, str] = {}
        current = list(predictions)

        for attempt in range(1, self._empty_prediction_max_retries + 1):
            empty_positions = [
                pos for pos, pred in enumerate(current) if self._is_empty_prediction(pred)
            ]
            if not empty_positions:
                break

            retry_prompts = [batch_prompts[pos] for pos in empty_positions]
            retry_images = [batch_images[pos] for pos in empty_positions]

            try:
                if hasattr(self.model, "run_async"):
                    retried = await self.model.run_async(retry_prompts, retry_images)
                else:
                    retried = self.model.run(retry_prompts, retry_images)
                retried_list = self._normalize_predictions(retried)
            except Exception as exc:
                for pos in empty_positions:
                    errors[pos] = f"Empty prediction retry failed: {exc}"
                break

            if len(retried_list) != len(empty_positions):
                for pos in empty_positions:
                    errors[pos] = (
                        "Empty prediction retry returned mismatched batch size"
                    )
                break

            for pos, retried_pred in zip(empty_positions, retried_list):
                current[pos] = retried_pred

            if attempt < self._empty_prediction_max_retries:
                await asyncio.sleep(
                    self._empty_prediction_retry_backoff_seconds * attempt
                )

        for pos, pred in enumerate(current):
            if self._is_empty_prediction(pred):
                errors[pos] = errors.get(pos) or "Empty prediction after retries"

        return current, errors

    def _retry_empty_predictions_sync(
        self,
        predictions: list[str],
        batch_prompts: list[str],
        batch_images: list[Image.Image],
    ) -> tuple[list[str], dict[int, str]]:
        errors: dict[int, str] = {}
        current = list(predictions)

        for attempt in range(1, self._empty_prediction_max_retries + 1):
            empty_positions = [
                pos for pos, pred in enumerate(current) if self._is_empty_prediction(pred)
            ]
            if not empty_positions:
                break

            retry_prompts = [batch_prompts[pos] for pos in empty_positions]
            retry_images = [batch_images[pos] for pos in empty_positions]

            try:
                retried = self.model.run(retry_prompts, retry_images)
                retried_list = self._normalize_predictions(retried)
            except Exception as exc:
                for pos in empty_positions:
                    errors[pos] = f"Empty prediction retry failed: {exc}"
                break

            if len(retried_list) != len(empty_positions):
                for pos in empty_positions:
                    errors[pos] = (
                        "Empty prediction retry returned mismatched batch size"
                    )
                break

            for pos, retried_pred in zip(empty_positions, retried_list):
                current[pos] = retried_pred

            if attempt < self._empty_prediction_max_retries:
                time.sleep(self._empty_prediction_retry_backoff_seconds * attempt)

        for pos, pred in enumerate(current):
            if self._is_empty_prediction(pred):
                errors[pos] = errors.get(pos) or "Empty prediction after retries"

        return current, errors

    def _checkpoint_context(self) -> dict[str, Any]:
        return {
            "dataset_id": self.config.dataset_id,
            "split": self.config.split,
            "model_id": self.config.model.model_id,
            "backend": self.config.model.backend.value,
        }

    def _load_or_create_state(self) -> RunnerState:
        """Load checkpoint if exists and resuming is enabled."""
        expected_context = self._checkpoint_context()
        checkpoint_to_load = self.checkpoint_path
        if not checkpoint_to_load.exists() and self.legacy_checkpoint_path.exists():
            checkpoint_to_load = self.legacy_checkpoint_path

        if self.config.resume_from_checkpoint and checkpoint_to_load.exists():
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

    def _save_checkpoint(self) -> None:
        """Save current state to checkpoint."""
        self.state.save(self.checkpoint_path)

    def _merge_results(self, current_results: list[InferenceResult]) -> list[InferenceResult]:
        """Merge checkpointed results with current run results by sample index."""
        merged: dict[int, InferenceResult] = {
            int(r["index"]): InferenceResult(**r) for r in self.state.results
        }
        for result in current_results:
            merged[result.index] = result
        return sorted(merged.values(), key=lambda r: r.index)

    async def run_async(
        self,
        images: Sequence[Image.Image],
        ground_truths: Sequence[Any],
        prompts: Sequence[str],
    ) -> list[InferenceResult]:
        """
        Run async inference with batching and checkpointing.

        Args:
            images: List of PIL Images.
            ground_truths: List of ground truth values.
            prompts: List of prompts.

        Returns:
            List of InferenceResult objects.
        """
        if self.config.batch_api:
            return await self._run_batch_api(images, ground_truths, prompts)

        results: list[InferenceResult] = []

        # Get remaining indices (not yet completed)
        remaining = self._remaining_indices(len(images))

        if not remaining:
            print("All samples already processed. Loading from checkpoint.")
            return [
                InferenceResult(**r) for r in self.state.results
            ]

        print(f"Processing {len(remaining)} remaining samples...")

        # Process in batches
        batch_size = self.config.batch_size
        progress = tqdm(
            range(0, len(remaining), batch_size),
            desc="Evaluating",
            unit="batch",
        )

        for batch_start in progress:
            batch_indices = remaining[batch_start : batch_start + batch_size]
            batch_images = [images[i] for i in batch_indices]
            batch_prompts = [prompts[i] for i in batch_indices]
            batch_gts = [ground_truths[i] for i in batch_indices]

            try:
                start_time = time.time()

                if hasattr(self.model, "run_async"):
                    predictions = await self.model.run_async(batch_prompts, batch_images)
                else:
                    predictions = self.model.run(batch_prompts, batch_images)

                predictions_list = self._normalize_predictions(predictions)
                self._ensure_batch_size(predictions_list, len(batch_indices))

                predictions_list, retry_errors = await self._retry_empty_predictions_async(
                    predictions_list,
                    batch_prompts,
                    batch_images,
                )

            except Exception as e:
                print(f"Error processing batch: {e}")
                for idx in batch_indices:
                    result = InferenceResult(
                        index=idx,
                        prediction="",
                        ground_truth=ground_truths[idx],
                        latency_ms=0,
                        error=str(e),
                    )
                    results.append(result)
                continue

            latency = (time.time() - start_time) * 1000 / len(batch_indices)

            for pos, (idx, pred, gt) in enumerate(
                zip(batch_indices, predictions_list, batch_gts)
            ):
                result = InferenceResult(
                    index=idx,
                    prediction=pred,
                    ground_truth=gt,
                    latency_ms=latency,
                    error=retry_errors.get(pos),
                )
                results.append(result)
                self._append_checkpoint_result(result)

            self._save_checkpoint()
            progress.set_postfix({"latency": f"{latency:.1f}ms"})

        return self._merge_results(results)

    def run(
        self,
        images: Sequence[Image.Image],
        ground_truths: Sequence[Any],
        prompts: Sequence[str],
    ) -> list[InferenceResult]:
        """
        Run synchronous inference with batching and checkpointing.

        Args:
            images: List of PIL Images.
            ground_truths: List of ground truth values.
            prompts: List of prompts.

        Returns:
            List of InferenceResult objects.
        """
        if self.config.batch_api:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._run_batch_api(images, ground_truths, prompts))
            raise RuntimeError("Batch API sync run() called inside event loop; use run_async")

        results: list[InferenceResult] = []

        # Get remaining indices
        remaining = self._remaining_indices(len(images))

        if not remaining:
            print("All samples already processed. Loading from checkpoint.")
            return [InferenceResult(**r) for r in self.state.results]

        print(f"Processing {len(remaining)} remaining samples...")

        # Process in batches
        batch_size = self.config.batch_size
        progress = tqdm(
            range(0, len(remaining), batch_size),
            desc="Evaluating",
            unit="batch",
        )

        for batch_start in progress:
            batch_indices = remaining[batch_start : batch_start + batch_size]
            batch_images = [images[i] for i in batch_indices]
            batch_prompts = [prompts[i] for i in batch_indices]
            batch_gts = [ground_truths[i] for i in batch_indices]

            try:
                start_time = time.time()

                predictions = self.model.run(batch_prompts, batch_images)
                predictions_list = self._normalize_predictions(predictions)
                self._ensure_batch_size(predictions_list, len(batch_indices))

                predictions_list, retry_errors = self._retry_empty_predictions_sync(
                    predictions_list,
                    batch_prompts,
                    batch_images,
                )

            except Exception as e:
                print(f"Error processing batch: {e}")
                for idx in batch_indices:
                    result = InferenceResult(
                        index=idx,
                        prediction="",
                        ground_truth=ground_truths[idx],
                        latency_ms=0,
                        error=str(e),
                    )
                    results.append(result)
                continue

            latency = (time.time() - start_time) * 1000 / len(batch_indices)

            for pos, (idx, pred, gt) in enumerate(
                zip(batch_indices, predictions_list, batch_gts)
            ):
                result = InferenceResult(
                    index=idx,
                    prediction=pred,
                    ground_truth=gt,
                    latency_ms=latency,
                    error=retry_errors.get(pos),
                )
                results.append(result)
                self._append_checkpoint_result(result)

            self._save_checkpoint()
            progress.set_postfix({"latency": f"{latency:.1f}ms"})

        return self._merge_results(results)

    async def _run_batch_api(
        self,
        images: Sequence[Image.Image],
        ground_truths: Sequence[Any],
        prompts: Sequence[str],
    ) -> list[InferenceResult]:
        batch_dir = self.output_dir / "batch"
        batch_info_path = batch_dir / "batch_info.json"

        remaining = self._remaining_indices(len(images))

        if not remaining:
            print("All samples already processed. Loading from checkpoint.")
            return [InferenceResult(**r) for r in self.state.results]

        if batch_info_path.exists() and hasattr(self.model, "resume_batch_async"):
            try:
                with open(batch_info_path, "r", encoding="utf-8") as f:
                    batch_info = json.load(f)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"Ignoring invalid batch metadata at {batch_info_path}: {exc}")
                batch_info = {}

            batch_id = batch_info.get("batch_id") if isinstance(batch_info, dict) else None
            batch_status = batch_info.get("status") if isinstance(batch_info, dict) else None
            if batch_id and batch_status not in {"failed", "cancelled", "expired"}:
                print(f"Resuming batch: {batch_id}")
                try:
                    prediction_map = await self.model.resume_batch_async(
                        batch_id=batch_id,
                        output_dir=batch_dir,
                        poll_interval=self.config.batch_poll_seconds,
                        timeout=self.config.batch_timeout_seconds,
                    )
                except NotImplementedError as exc:
                    raise RuntimeError(
                        "Batch API requested but model does not support batch."
                    ) from exc
                error_map = self._load_batch_errors(batch_dir)
                return self._finalize_batch_results(
                    prediction_map,
                    remaining,
                    ground_truths,
                    error_map=error_map,
                )

        if not hasattr(self.model, "run_batch_async"):
            raise RuntimeError("Batch API requested but model does not support batch.")

        print(f"Submitting batch for {len(remaining)} samples...")

        batch_indices = remaining
        batch_images = [images[i] for i in batch_indices]
        batch_prompts = [prompts[i] for i in batch_indices]
        custom_ids = [str(i) for i in batch_indices]

        start_time = time.time()
        try:
            prediction_map = await self.model.run_batch_async(
                batch_prompts,
                batch_images,
                custom_ids,
                output_dir=batch_dir,
                completion_window=self.config.batch_completion_window,
                poll_interval=self.config.batch_poll_seconds,
                timeout=self.config.batch_timeout_seconds,
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                "Batch API requested but model does not support batch."
            ) from exc
        latency = (time.time() - start_time) * 1000 / len(batch_indices)
        error_map = self._load_batch_errors(batch_dir)
        return self._finalize_batch_results(
            prediction_map,
            batch_indices,
            ground_truths,
            latency,
            error_map=error_map,
        )

    def _finalize_batch_results(
        self,
        prediction_map: dict[str, str],
        indices: list[int],
        ground_truths: Sequence[Any],
        latency_ms: float = 0,
        error_map: dict[str, str] | None = None,
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for idx in indices:
            custom_id = str(idx)
            prediction = prediction_map.get(custom_id)
            error = None
            if prediction is None:
                prediction = ""
                error = "Missing batch response"
            elif self._is_empty_prediction(prediction):
                error = "Empty prediction in batch response"
            if error_map and custom_id in error_map:
                error = error_map[custom_id]

            result = InferenceResult(
                index=idx,
                prediction=prediction,
                ground_truth=ground_truths[idx],
                latency_ms=latency_ms,
                error=error,
            )
            if idx not in self._completed_indices:
                results.append(result)
            self._append_checkpoint_result(result)

        self._save_checkpoint()
        all_results = [InferenceResult(**r) for r in self.state.results]
        all_results.sort(key=lambda r: r.index)
        return all_results

    def _load_batch_errors(self, batch_dir: Path) -> dict[str, str]:
        error_path = batch_dir / "batch_errors.json"
        if not error_path.exists():
            return {}
        try:
            with open(error_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Ignoring invalid batch error file at {error_path}: {exc}")
            return {}

        if not isinstance(data, dict):
            print(f"Ignoring invalid batch error format at {error_path}: expected object")
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def clear_checkpoint(self) -> None:
        """Clear the checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self.state = RunnerState(context=self._checkpoint_context())
        self._completed_indices = set()
        print("Checkpoint cleared.")
