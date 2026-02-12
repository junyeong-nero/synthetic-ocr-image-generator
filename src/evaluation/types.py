"""Evaluation result types."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
import json


@dataclass
class InferenceResult:
    """Single inference result."""

    index: int
    prediction: str
    ground_truth: Any
    latency_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RunnerState:
    """Runner checkpoint state for resumable evaluation."""

    completed: list[int] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save state to checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RunnerState":
        """Load state from checkpoint file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class EvaluationOutput:
    """Evaluation pipeline output."""

    config: dict[str, Any]
    metrics: dict[str, float]
    per_sample_results: list[dict[str, Any]]
    summary: dict[str, Any]
    metric_views: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config,
            "metrics": self.metrics,
            "metric_views": self.metric_views,
            "per_sample_results": self.per_sample_results,
            "summary": self.summary,
        }

    def save_json(self, path: Path) -> Path:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path
