"""Evaluation runner with checkpointing support."""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Union

from PIL import Image
from tqdm import tqdm

from evaluation.config import EvaluationConfig
from evaluation.types import InferenceResult, RunnerState
from models.base import Model, VLMModel


class EvaluationRunner:
    """
    Batch inference runner with checkpointing support.

    Handles batch processing, progress tracking, and resumable evaluation.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        model: Union[Model, VLMModel],
    ):
        self.config = config
        self.model = model

        # Setup output directory and checkpoint
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.json"

        # Load or create state
        self.state = self._load_or_create_state()

    def _load_or_create_state(self) -> RunnerState:
        """Load checkpoint if exists and resuming is enabled."""
        if self.config.resume_from_checkpoint and self.checkpoint_path.exists():
            try:
                state = RunnerState.load(self.checkpoint_path)
                if state.completed:
                    print(f"Resuming from checkpoint: {len(state.completed)} samples completed")
                return state
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting fresh.")

        return RunnerState()

    def _save_checkpoint(self) -> None:
        """Save current state to checkpoint."""
        self.state.save(self.checkpoint_path)

    async def run_async(
        self,
        images: List[Image.Image],
        ground_truths: List[Any],
        prompts: List[str],
    ) -> List[InferenceResult]:
        """
        Run async inference with batching and checkpointing.

        Args:
            images: List of PIL Images.
            ground_truths: List of ground truth values.
            prompts: List of prompts.

        Returns:
            List of InferenceResult objects.
        """
        results: List[InferenceResult] = []

        # Get remaining indices (not yet completed)
        remaining = [
            i for i in range(len(images)) if i not in self.state.completed
        ]

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

                # Run inference
                if hasattr(self.model, "run_async"):
                    predictions = await self.model.run_async(batch_prompts, batch_images)
                else:
                    predictions = self.model.run(batch_prompts, batch_images)

                latency = (time.time() - start_time) * 1000 / len(batch_indices)

                # Create results
                for idx, pred, gt in zip(batch_indices, predictions, batch_gts):
                    result = InferenceResult(
                        index=idx,
                        prediction=pred,
                        ground_truth=gt,
                        latency_ms=latency,
                    )
                    results.append(result)
                    self.state.completed.append(idx)
                    self.state.results.append(result.to_dict())

                # Save checkpoint
                self._save_checkpoint()

                # Update progress
                progress.set_postfix({"latency": f"{latency:.1f}ms"})

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

        # Combine with previously completed results
        all_results = [InferenceResult(**r) for r in self.state.results]

        # Sort by index
        all_results.sort(key=lambda r: r.index)

        return all_results

    def run(
        self,
        images: List[Image.Image],
        ground_truths: List[Any],
        prompts: List[str],
    ) -> List[InferenceResult]:
        """
        Run synchronous inference with batching and checkpointing.

        Args:
            images: List of PIL Images.
            ground_truths: List of ground truth values.
            prompts: List of prompts.

        Returns:
            List of InferenceResult objects.
        """
        results: List[InferenceResult] = []

        # Get remaining indices
        remaining = [
            i for i in range(len(images)) if i not in self.state.completed
        ]

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

                # Run inference
                predictions = self.model.run(batch_prompts, batch_images)

                latency = (time.time() - start_time) * 1000 / len(batch_indices)

                # Create results
                for idx, pred, gt in zip(batch_indices, predictions, batch_gts):
                    result = InferenceResult(
                        index=idx,
                        prediction=pred,
                        ground_truth=gt,
                        latency_ms=latency,
                    )
                    results.append(result)
                    self.state.completed.append(idx)
                    self.state.results.append(result.to_dict())

                # Save checkpoint
                self._save_checkpoint()

                # Update progress
                progress.set_postfix({"latency": f"{latency:.1f}ms"})

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

        # Combine with previously completed results
        all_results = [InferenceResult(**r) for r in self.state.results]

        # Sort by index
        all_results.sort(key=lambda r: r.index)

        return all_results

    def clear_checkpoint(self) -> None:
        """Clear the checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self.state = RunnerState()
        print("Checkpoint cleared.")
