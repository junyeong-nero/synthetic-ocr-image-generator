"""Evaluation pipeline for VLM/OCR models."""

from evaluation.config import (
    DEFAULT_PROMPTS,
    EvaluationConfig,
    FormatType,
    InferenceBackend,
    ModelConfig,
)
from evaluation.model_config import (
    ModelConfigLoader,
    ModelSpecificConfig,
    PromptConfig,
    SubsetConfig,
)
from evaluation.comparator import ModelComparator, compare_models
from evaluation.pipeline import EvaluationPipeline, evaluate_pipeline
from evaluation.report import ReportGenerator, generate_report
from evaluation.runner import EvaluationRunner
from evaluation.types import EvaluationOutput, InferenceResult, RunnerState

__all__ = [
    # Config
    "EvaluationConfig",
    "ModelConfig",
    "FormatType",
    "InferenceBackend",
    "DEFAULT_PROMPTS",
    # Model Config
    "ModelConfigLoader",
    "ModelSpecificConfig",
    "PromptConfig",
    "SubsetConfig",
    # Pipeline
    "EvaluationPipeline",
    "evaluate_pipeline",
    # Runner
    "EvaluationRunner",
    # Types
    "EvaluationOutput",
    "InferenceResult",
    "RunnerState",
    # Report
    "ReportGenerator",
    "generate_report",
    # Comparator
    "ModelComparator",
    "compare_models",
]
