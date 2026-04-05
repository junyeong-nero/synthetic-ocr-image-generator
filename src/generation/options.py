from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Optional


@dataclass(frozen=True)
class PublishOptions:
    repo_id: Optional[str] = None
    train_ratio: float = 0.9
    test_ratio: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PublishOptions":
        return cls(
            repo_id=data.get("repo_id"),
            train_ratio=float(data.get("train_ratio", 0.9)),
            test_ratio=float(data.get("test_ratio", 0.1)),
        )

    def with_overrides(
        self,
        *,
        repo_id: Optional[str] = None,
        train_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
    ) -> "PublishOptions":
        return replace(
            self,
            repo_id=repo_id if repo_id is not None else self.repo_id,
            train_ratio=train_ratio if train_ratio is not None else self.train_ratio,
            test_ratio=test_ratio if test_ratio is not None else self.test_ratio,
        )


@dataclass(frozen=True)
class GenerationOptions:
    template: Optional[str] = None
    template_family: Optional[str] = None
    min_template_complexity: Optional[int] = None
    max_template_complexity: Optional[int] = None
    template_config_dir: Optional[str] = None
    markdown_renderer: str = "playwright"
    style_profile: str = "balanced"
    coverage_targets: Any = None
    novelty_window: int = 80
    novelty_threshold: float = 0.95
    novelty_max_attempts: int = 4
    similar_char_ratio: float = 0.08
    similarity_db_path: Optional[str] = None
    formula_source_mode: str = "mixed"
    formula_dataset_path: Optional[str] = None
    formula_dataset_weight: float = 0.45
    formula_random_weight: float = 0.30
    formula_synthetic_weight: float = 0.25
    add_noise: Optional[bool] = None
    add_blur: Optional[bool] = None
    seed: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationOptions":
        return cls(
            template=data.get("template"),
            template_family=data.get("template_family"),
            min_template_complexity=data.get("min_template_complexity"),
            max_template_complexity=data.get("max_template_complexity"),
            template_config_dir=data.get("template_config_dir"),
            markdown_renderer=str(data.get("markdown_renderer", "playwright")),
            style_profile=str(data.get("style_profile", "balanced")),
            coverage_targets=data.get("coverage_targets"),
            novelty_window=int(data.get("novelty_window", 80)),
            novelty_threshold=float(data.get("novelty_threshold", 0.95)),
            novelty_max_attempts=int(data.get("novelty_max_attempts", 4)),
            similar_char_ratio=float(data.get("similar_char_ratio", 0.08)),
            similarity_db_path=data.get("similarity_db_path"),
            formula_source_mode=str(data.get("formula_source_mode", "mixed")),
            formula_dataset_path=data.get("formula_dataset_path"),
            formula_dataset_weight=float(data.get("formula_dataset_weight", 0.45)),
            formula_random_weight=float(data.get("formula_random_weight", 0.30)),
            formula_synthetic_weight=float(data.get("formula_synthetic_weight", 0.25)),
            add_noise=data.get("add_noise"),
            add_blur=data.get("add_blur"),
            seed=data.get("seed"),
        )

    def to_generator_kwargs(self, *, sample_start_index: int = 0) -> dict[str, Any]:
        generation_kwargs: dict[str, Any] = {
            "template": self.template,
            "template_family": self.template_family,
            "min_template_complexity": self.min_template_complexity,
            "max_template_complexity": self.max_template_complexity,
            "template_config_dir": self.template_config_dir,
            "markdown_renderer": self.markdown_renderer,
            "style_profile": self.style_profile,
            "coverage_targets": self.coverage_targets,
            "novelty_window": self.novelty_window,
            "novelty_threshold": self.novelty_threshold,
            "novelty_max_attempts": self.novelty_max_attempts,
            "similar_char_ratio": self.similar_char_ratio,
            "similarity_db_path": self.similarity_db_path,
            "formula_source_mode": self.formula_source_mode,
            "formula_dataset_path": self.formula_dataset_path,
            "formula_dataset_weight": self.formula_dataset_weight,
            "formula_random_weight": self.formula_random_weight,
            "formula_synthetic_weight": self.formula_synthetic_weight,
            "seed": self.seed,
            "sample_start_index": sample_start_index,
        }
        if self.add_noise is not None:
            generation_kwargs["add_noise"] = self.add_noise
        if self.add_blur is not None:
            generation_kwargs["add_blur"] = self.add_blur
        return generation_kwargs


@dataclass(frozen=True)
class GenerationTaskContext:
    lang: str
    size: int
    generation: GenerationOptions
    publish: PublishOptions

    def to_dict(self) -> dict[str, Any]:
        return {
            "lang": self.lang,
            "size": self.size,
            "generation": self.generation.to_dict(),
            "publish": self.publish.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationTaskContext":
        return cls(
            lang=str(data.get("lang", "ko")),
            size=int(data.get("size", 0)),
            generation=GenerationOptions.from_dict(data.get("generation") or {}),
            publish=PublishOptions.from_dict(data.get("publish") or {}),
        )

    @classmethod
    def from_manifest_data(cls, manifest_data: dict[str, Any]) -> "GenerationTaskContext":
        task_context = manifest_data.get("task_context")
        if isinstance(task_context, dict):
            return cls.from_dict(task_context)

        legacy_generation_config = dict(manifest_data.get("generation_config") or {})
        generation_payload = dict(legacy_generation_config)
        generation_payload["seed"] = manifest_data.get(
            "seed", legacy_generation_config.get("seed")
        )
        publish_payload = {
            "repo_id": manifest_data.get("repo_id"),
            "train_ratio": legacy_generation_config.get("train_ratio", 0.9),
            "test_ratio": legacy_generation_config.get("test_ratio", 0.1),
        }
        return cls(
            lang=str(manifest_data.get("lang", legacy_generation_config.get("lang", "ko"))),
            size=int(manifest_data.get("size", legacy_generation_config.get("size", 0))),
            generation=GenerationOptions.from_dict(generation_payload),
            publish=PublishOptions.from_dict(publish_payload),
        )

    def with_publish_overrides(
        self,
        *,
        repo_id: Optional[str] = None,
        train_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
    ) -> "GenerationTaskContext":
        return replace(
            self,
            publish=self.publish.with_overrides(
                repo_id=repo_id,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
            ),
        )
