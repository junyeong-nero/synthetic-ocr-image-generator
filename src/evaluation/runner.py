"""Evaluation runner with checkpointing support."""

import asyncio
import time
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from tqdm import tqdm

from evaluation.batch_api import build_batch_results, load_batch_errors, load_batch_info
from evaluation.checkpoint import build_checkpoint_context
from evaluation.checkpoint_store import load_or_create_state, save_checkpoint
from evaluation.config import EvaluationConfig, InferenceBackend
from evaluation.retry import (
    apply_retry_results,
    empty_prediction_positions,
    ensure_batch_size,
    finalize_retry_errors,
    is_empty_prediction,
    normalize_predictions,
    prepare_retry_batch,
)
from evaluation.types import InferenceResult, RunnerState
from models.base import VLMModel


class EvaluationRunner:
    _API_BACKENDS = {
        InferenceBackend.OPENAI,
        InferenceBackend.ANTHROPIC,
        InferenceBackend.GOOGLE,
        InferenceBackend.UPSTAGE,
    }

    def __init__(
        self,
        config: EvaluationConfig,
        model: VLMModel,
    ):
        self.config = config
        self.model = model

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoints.json"

        self.state = load_or_create_state(self.config, self.output_dir, self.checkpoint_path)
        self._completed_indices = set(self.state.completed)
        self._empty_prediction_max_retries = (
            5 if self.config.model.backend in self._API_BACKENDS else 1
        )
        self._empty_prediction_retry_backoff_seconds = 0.5

    @staticmethod
    def _normalize_predictions(predictions: Any) -> list[str]:
        return normalize_predictions(predictions)

    @staticmethod
    def _ensure_batch_size(predictions: list[str], expected_size: int) -> None:
        ensure_batch_size(predictions, expected_size)

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
        return is_empty_prediction(prediction)

    def _empty_prediction_positions(self, predictions: list[str]) -> list[int]:
        return empty_prediction_positions(predictions)

    @staticmethod
    def _prepare_retry_batch(
        empty_positions: list[int],
        batch_prompts: list[str],
        batch_images: list[Image.Image],
    ) -> tuple[list[str], list[Image.Image]]:
        return prepare_retry_batch(empty_positions, batch_prompts, batch_images)

    @staticmethod
    def _finalize_retry_errors(
        predictions: list[str],
        errors: dict[int, str],
    ) -> dict[int, str]:
        return finalize_retry_errors(predictions, errors)

    @staticmethod
    def _apply_retry_results(
        current_predictions: list[str],
        empty_positions: list[int],
        retried_predictions: list[str],
        errors: dict[int, str],
    ) -> bool:
        return apply_retry_results(current_predictions, empty_positions, retried_predictions, errors)

    async def _retry_empty_predictions_async(
        self,
        predictions: list[str],
        batch_prompts: list[str],
        batch_images: list[Image.Image],
    ) -> tuple[list[str], dict[int, str]]:
        errors: dict[int, str] = {}
        current = list(predictions)

        for attempt in range(1, self._empty_prediction_max_retries + 1):
            empty_positions = self._empty_prediction_positions(current)
            if not empty_positions:
                break

            retry_prompts, retry_images = self._prepare_retry_batch(
                empty_positions,
                batch_prompts,
                batch_images,
            )

            try:
                if hasattr(self.model, "run_async"):
                    retried = await self.model.run_async(retry_prompts, retry_images)
                else:
                    retried = self.model.run(retry_prompts, retry_images)
                retried_list = self._normalize_predictions(retried)
            except Exception as exc:
                for position in empty_positions:
                    errors[position] = f"Empty prediction retry failed: {exc}"
                break

            if not self._apply_retry_results(current, empty_positions, retried_list, errors):
                break

            if attempt < self._empty_prediction_max_retries:
                await asyncio.sleep(
                    self._empty_prediction_retry_backoff_seconds * attempt
                )

        finalized_errors = self._finalize_retry_errors(current, errors)
        return current, finalized_errors

    def _retry_empty_predictions_sync(
        self,
        predictions: list[str],
        batch_prompts: list[str],
        batch_images: list[Image.Image],
    ) -> tuple[list[str], dict[int, str]]:
        errors: dict[int, str] = {}
        current = list(predictions)

        for attempt in range(1, self._empty_prediction_max_retries + 1):
            empty_positions = self._empty_prediction_positions(current)
            if not empty_positions:
                break

            retry_prompts, retry_images = self._prepare_retry_batch(
                empty_positions,
                batch_prompts,
                batch_images,
            )

            try:
                retried = self.model.run(retry_prompts, retry_images)
                retried_list = self._normalize_predictions(retried)
            except Exception as exc:
                for position in empty_positions:
                    errors[position] = f"Empty prediction retry failed: {exc}"
                break

            if not self._apply_retry_results(current, empty_positions, retried_list, errors):
                break

            if attempt < self._empty_prediction_max_retries:
                time.sleep(self._empty_prediction_retry_backoff_seconds * attempt)

        finalized_errors = self._finalize_retry_errors(current, errors)
        return current, finalized_errors

    def _checkpoint_context(self) -> dict[str, Any]:
        return build_checkpoint_context(self.config)

    def _save_checkpoint(self) -> None:
        save_checkpoint(self.state, self.checkpoint_path)

    def _merge_results(self, current_results: list[InferenceResult]) -> list[InferenceResult]:
        merged: dict[int, InferenceResult] = {
            int(row["index"]): InferenceResult(**row) for row in self.state.results
        }
        for result in current_results:
            merged[result.index] = result
        return sorted(merged.values(), key=lambda result: result.index)

    def _completed_results(self) -> list[InferenceResult]:
        return [InferenceResult(**row) for row in self.state.results]

    def _start_standard_run(
        self, total_samples: int
    ) -> tuple[list[int], Any] | tuple[None, list[InferenceResult]]:
        remaining = self._remaining_indices(total_samples)
        if not remaining:
            print("All samples already processed. Loading from checkpoint.")
            return None, self._completed_results()

        print(f"Processing {len(remaining)} remaining samples...")
        progress = tqdm(
            range(0, len(remaining), self.config.batch_size),
            desc="Evaluating",
            unit="batch",
        )
        return remaining, progress

    @staticmethod
    def _batch_inputs(
        indices: list[int],
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        ground_truths: Sequence[Any],
    ) -> tuple[list[Image.Image], list[str], list[Any]]:
        return (
            [images[i] for i in indices],
            [prompts[i] for i in indices],
            [ground_truths[i] for i in indices],
        )

    @staticmethod
    def _failure_results(
        batch_indices: list[int],
        ground_truths: Sequence[Any],
        error: Exception,
    ) -> list[InferenceResult]:
        return [
            InferenceResult(
                index=idx,
                prediction="",
                ground_truth=ground_truths[idx],
                latency_ms=0,
                error=str(error),
            )
            for idx in batch_indices
        ]

    @staticmethod
    def _successful_results(
        batch_indices: list[int],
        predictions: list[str],
        ground_truths: list[Any],
        latency: float,
        retry_errors: dict[int, str],
    ) -> list[InferenceResult]:
        return [
            InferenceResult(
                index=idx,
                prediction=prediction,
                ground_truth=ground_truth,
                latency_ms=latency,
                error=retry_errors.get(position),
            )
            for position, (idx, prediction, ground_truth) in enumerate(
                zip(batch_indices, predictions, ground_truths)
            )
        ]

    def _record_results(
        self,
        results: list[InferenceResult],
        progress: Any,
        latency: float | None = None,
    ) -> None:
        for result in results:
            self._append_checkpoint_result(result)
        if results:
            self._save_checkpoint()
        if latency is not None:
            progress.set_postfix({"latency": f"{latency:.1f}ms"})

    def _finalize_standard_batch(
        self,
        batch_indices: list[int],
        batch_gts: list[Any],
        predictions_list: list[str],
        retry_errors: dict[int, str],
        start_time: float,
    ) -> tuple[list[InferenceResult], float]:
        latency = (time.time() - start_time) * 1000 / len(batch_indices)
        return (
            self._successful_results(
                batch_indices,
                predictions_list,
                batch_gts,
                latency,
                retry_errors,
            ),
            latency,
        )

    async def run_async(
        self,
        images: Sequence[Image.Image],
        ground_truths: Sequence[Any],
        prompts: Sequence[str],
    ) -> list[InferenceResult]:
        if self.config.batch_api:
            return await self._run_batch_api(images, ground_truths, prompts)

        results: list[InferenceResult] = []
        remaining, progress = self._start_standard_run(len(images))
        if remaining is None:
            return progress

        for batch_start in progress:
            batch_indices = remaining[batch_start : batch_start + self.config.batch_size]
            batch_images, batch_prompts, batch_gts = self._batch_inputs(
                batch_indices,
                images,
                prompts,
                ground_truths,
            )

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
            except Exception as exc:
                print(f"Error processing batch: {exc}")
                results.extend(self._failure_results(batch_indices, ground_truths, exc))
                continue

            batch_results, latency = self._finalize_standard_batch(
                batch_indices,
                batch_gts,
                predictions_list,
                retry_errors,
                start_time,
            )
            results.extend(batch_results)
            self._record_results(batch_results, progress, latency)

        return self._merge_results(results)

    def run(
        self,
        images: Sequence[Image.Image],
        ground_truths: Sequence[Any],
        prompts: Sequence[str],
    ) -> list[InferenceResult]:
        if self.config.batch_api:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._run_batch_api(images, ground_truths, prompts))
            raise RuntimeError("Batch API sync run() called inside event loop; use run_async")

        results: list[InferenceResult] = []
        remaining, progress = self._start_standard_run(len(images))
        if remaining is None:
            return progress

        for batch_start in progress:
            batch_indices = remaining[batch_start : batch_start + self.config.batch_size]
            batch_images, batch_prompts, batch_gts = self._batch_inputs(
                batch_indices,
                images,
                prompts,
                ground_truths,
            )

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
            except Exception as exc:
                print(f"Error processing batch: {exc}")
                results.extend(self._failure_results(batch_indices, ground_truths, exc))
                continue

            batch_results, latency = self._finalize_standard_batch(
                batch_indices,
                batch_gts,
                predictions_list,
                retry_errors,
                start_time,
            )
            results.extend(batch_results)
            self._record_results(batch_results, progress, latency)

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
            return self._completed_results()

        if batch_info_path.exists() and hasattr(self.model, "resume_batch_async"):
            batch_info = load_batch_info(batch_info_path)
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
                error_map = load_batch_errors(batch_dir)
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
        error_map = load_batch_errors(batch_dir)
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
        results = build_batch_results(
            prediction_map,
            indices,
            ground_truths,
            latency_ms=latency_ms,
            error_map=error_map,
            is_empty_prediction=self._is_empty_prediction,
        )
        for result in results:
            self._append_checkpoint_result(result)

        self._save_checkpoint()
        all_results = self._completed_results()
        all_results.sort(key=lambda result: result.index)
        return all_results

    def _load_batch_errors(self, batch_dir: Path) -> dict[str, str]:
        return load_batch_errors(batch_dir)

    def clear_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self.state = RunnerState(context=self._checkpoint_context())
        self._completed_indices = set()
        print("Checkpoint cleared.")