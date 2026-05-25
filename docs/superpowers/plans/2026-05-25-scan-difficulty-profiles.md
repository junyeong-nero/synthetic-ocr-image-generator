# Scan Difficulty Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in scanned-paper visual difficulty buckets for benchmark-first synthetic OCR generation, with deterministic balanced sampling, metadata, and preview-sheet validation.

**Architecture:** Add a focused `visual_difficulty` generator module for profile parsing, bucket sampling, parameter sampling, and PIL post-processing. Wire it through `GenerationOptions`, the `generate` CLI, `Generator.generate_single()`, and a generation-level preview sheet helper that reads final metadata after shard aggregation.

**Tech Stack:** Python 3.11, dataclasses, argparse, PIL/Pillow, pytest, existing `uv run pytest` workflow.

---

## File Structure

- Create `src/generator/visual_difficulty.py`: owns bucket constants, distribution parsing, deterministic sampler, scan profile parameter sampling, and scanned-paper PIL effects.
- Create `src/generation/preview_sheet.py`: reads generated metadata and builds a labeled grid preview image from saved samples.
- Create `tests/generator/test_visual_difficulty.py`: unit tests for parser, sampler, profile ranges, and effect mutation.
- Create `tests/generation/test_preview_sheet.py`: unit test for preview sheet output from fixture images and metadata.
- Modify `src/generation/options.py`: add `visual_difficulty_mode`, `visual_difficulty_distribution`, and `preview_sheet` to generation context serialization.
- Modify `src/cli/generate.py`: add CLI flags and pass them into `GenerationOptions`.
- Modify `src/generator/generator.py`: configure visual difficulty and apply scan effects after rendering; record metadata.
- Modify `src/pipeline.py`: build preview sheet after aggregate metadata is rebuilt.
- Modify `tests/generator/test_dynamic_templates.py`: verify `Generator.generate_single()` metadata and image post-processing when enabled.
- Modify `tests/generation/test_phase2_generation_modules.py`: verify new `GenerationOptions` fields survive manifest/context round trips.
- Modify `docs/generation.md`: document the new CLI options and smoke command.

---

### Task 1: Visual Difficulty Core Module

**Files:**
- Create: `src/generator/visual_difficulty.py`
- Create: `tests/generator/test_visual_difficulty.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/generator/test_visual_difficulty.py`:

```python
import random

from PIL import Image, ImageChops

from generator.visual_difficulty import (
    BUCKETS,
    DifficultySampler,
    ScanDifficultyProfile,
    apply_scan_effects,
    default_scan_profiles,
    parse_visual_difficulty_distribution,
)


def test_parse_visual_difficulty_distribution_defaults_to_balanced() -> None:
    assert parse_visual_difficulty_distribution(None) == {
        "easy": 1 / 3,
        "medium": 1 / 3,
        "hard": 1 / 3,
    }


def test_parse_visual_difficulty_distribution_normalizes_valid_weights() -> None:
    parsed = parse_visual_difficulty_distribution("easy=2,medium=1,hard=1")

    assert parsed == {"easy": 0.5, "medium": 0.25, "hard": 0.25}


def test_parse_visual_difficulty_distribution_ignores_unknown_buckets() -> None:
    parsed = parse_visual_difficulty_distribution(["easy=1", "unknown=9", "hard=1"])

    assert parsed == {"easy": 0.5, "medium": 0.0, "hard": 0.5}


def test_parse_visual_difficulty_distribution_falls_back_when_empty() -> None:
    assert parse_visual_difficulty_distribution("unknown=1,hard=0") == {
        "easy": 1 / 3,
        "medium": 1 / 3,
        "hard": 1 / 3,
    }


def test_balanced_sampler_uses_round_robin_counts() -> None:
    sampler = DifficultySampler.balanced()

    counts = {bucket: 0 for bucket in BUCKETS}
    for sample_index in range(10):
        counts[sampler.select(sample_index)] += 1

    assert counts == {"easy": 4, "medium": 3, "hard": 3}


def test_weighted_sampler_is_deterministic_for_seed_and_sample_index() -> None:
    sampler = DifficultySampler.weighted({"easy": 0.1, "medium": 0.2, "hard": 0.7}, seed=123)

    first = [sampler.select(sample_index) for sample_index in range(20)]
    second = [sampler.select(sample_index) for sample_index in range(20)]

    assert first == second
    assert set(first).issubset(set(BUCKETS))


def test_default_scan_profiles_sample_parameters_inside_ranges() -> None:
    profiles = default_scan_profiles()

    for bucket, profile in profiles.items():
        params = profile.sample(random.Random(7))
        assert bucket in BUCKETS
        assert profile.paper_tint[0] <= params.paper_tint <= profile.paper_tint[1]
        assert profile.speckle_amount[0] <= params.speckle_amount <= profile.speckle_amount[1]
        assert profile.blur_radius[0] <= params.blur_radius <= profile.blur_radius[1]
        assert profile.skew_degrees[0] <= params.skew_degrees <= profile.skew_degrees[1]
        assert profile.contrast_factor[0] <= params.contrast_factor <= profile.contrast_factor[1]
        assert profile.scan_line_alpha[0] <= params.scan_line_alpha <= profile.scan_line_alpha[1]


def test_apply_scan_effects_changes_image_pixels() -> None:
    profile = ScanDifficultyProfile(
        name="hard",
        paper_tint=(235, 235),
        speckle_amount=(0.08, 0.08),
        blur_radius=(0.0, 0.0),
        skew_degrees=(0.0, 0.0),
        contrast_factor=(0.9, 0.9),
        scan_line_alpha=(0.15, 0.15),
    )
    params = profile.sample(random.Random(11))
    image = Image.new("RGB", (120, 80), color=(255, 255, 255))

    transformed = apply_scan_effects(image, params, rng=random.Random(11))

    assert transformed.mode == "RGB"
    assert transformed.size == image.size
    assert ImageChops.difference(image, transformed).getbbox() is not None
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
uv run pytest tests/generator/test_visual_difficulty.py -q
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'generator.visual_difficulty'`.

- [ ] **Step 3: Implement the core module**

Create `src/generator/visual_difficulty.py`:

```python
from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

BUCKETS: tuple[str, str, str] = ("easy", "medium", "hard")
BALANCED_SCAN_MODE = "balanced-scan"
VISUAL_EFFECT_FAMILY_SCANNED = "scanned_paper"


@dataclass(frozen=True)
class ScanEffectParams:
    bucket: str
    paper_tint: int
    speckle_amount: float
    blur_radius: float
    skew_degrees: float
    contrast_factor: float
    scan_line_alpha: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class ScanDifficultyProfile:
    name: str
    paper_tint: tuple[int, int]
    speckle_amount: tuple[float, float]
    blur_radius: tuple[float, float]
    skew_degrees: tuple[float, float]
    contrast_factor: tuple[float, float]
    scan_line_alpha: tuple[float, float]

    def sample(self, rng: random.Random) -> ScanEffectParams:
        return ScanEffectParams(
            bucket=self.name,
            paper_tint=rng.randint(*self.paper_tint),
            speckle_amount=rng.uniform(*self.speckle_amount),
            blur_radius=rng.uniform(*self.blur_radius),
            skew_degrees=rng.uniform(*self.skew_degrees),
            contrast_factor=rng.uniform(*self.contrast_factor),
            scan_line_alpha=rng.uniform(*self.scan_line_alpha),
        )


def _balanced_distribution() -> dict[str, float]:
    return {bucket: 1.0 / len(BUCKETS) for bucket in BUCKETS}


def _put_weight(parsed: dict[str, float], key: str, value: Any) -> None:
    bucket = key.strip().lower()
    if bucket not in BUCKETS:
        logger.warning("Ignoring unknown visual difficulty bucket '%s'.", key)
        return
    try:
        weight = float(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid visual difficulty weight for '%s': %r", key, value)
        return
    parsed[bucket] = max(0.0, weight)


def parse_visual_difficulty_distribution(raw: Any) -> dict[str, float]:
    if raw is None:
        return _balanced_distribution()

    parsed = {bucket: 0.0 for bucket in BUCKETS}

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            _put_weight(parsed, str(key), value)
    else:
        items: list[str] = []
        if isinstance(raw, str):
            items.extend(token for token in raw.split(",") if token.strip())
        elif isinstance(raw, (list, tuple, set)):
            for item in raw:
                if isinstance(item, str):
                    items.extend(token for token in item.split(",") if token.strip())

        for item in items:
            if "=" in item:
                key, value = item.split("=", 1)
            elif ":" in item:
                key, value = item.split(":", 1)
            else:
                logger.warning("Ignoring invalid visual difficulty distribution token '%s'.", item)
                continue
            _put_weight(parsed, key, value)

    total = sum(parsed.values())
    if total <= 0:
        logger.warning("Using balanced visual difficulty distribution because no positive weights were provided.")
        return _balanced_distribution()

    return {bucket: parsed[bucket] / total for bucket in BUCKETS}


class DifficultySampler:
    def __init__(
        self,
        distribution: dict[str, float],
        *,
        seed: int | None = None,
        round_robin: bool = False,
    ) -> None:
        self.distribution = {bucket: max(0.0, float(distribution.get(bucket, 0.0))) for bucket in BUCKETS}
        self.seed = seed
        self.round_robin = round_robin

    @classmethod
    def balanced(cls) -> "DifficultySampler":
        return cls(_balanced_distribution(), round_robin=True)

    @classmethod
    def weighted(cls, distribution: dict[str, float], seed: int | None = None) -> "DifficultySampler":
        return cls(distribution, seed=seed, round_robin=False)

    def select(self, sample_index: int) -> str:
        if self.round_robin:
            return BUCKETS[int(sample_index) % len(BUCKETS)]

        rng = random.Random((self.seed or 0) + int(sample_index) * 1_000_003)
        value = rng.random()
        cumulative = 0.0
        for bucket in BUCKETS:
            cumulative += self.distribution.get(bucket, 0.0)
            if value <= cumulative:
                return bucket
        return BUCKETS[-1]


def default_scan_profiles() -> dict[str, ScanDifficultyProfile]:
    return {
        "easy": ScanDifficultyProfile(
            name="easy",
            paper_tint=(246, 253),
            speckle_amount=(0.001, 0.006),
            blur_radius=(0.0, 0.25),
            skew_degrees=(-0.4, 0.4),
            contrast_factor=(0.98, 1.05),
            scan_line_alpha=(0.0, 0.025),
        ),
        "medium": ScanDifficultyProfile(
            name="medium",
            paper_tint=(238, 250),
            speckle_amount=(0.006, 0.02),
            blur_radius=(0.2, 0.55),
            skew_degrees=(-1.2, 1.2),
            contrast_factor=(0.9, 1.12),
            scan_line_alpha=(0.025, 0.08),
        ),
        "hard": ScanDifficultyProfile(
            name="hard",
            paper_tint=(224, 244),
            speckle_amount=(0.02, 0.055),
            blur_radius=(0.45, 0.95),
            skew_degrees=(-2.4, 2.4),
            contrast_factor=(0.78, 1.22),
            scan_line_alpha=(0.08, 0.18),
        ),
    }


def make_scan_sampler(raw_distribution: Any, *, seed: int | None) -> DifficultySampler:
    if raw_distribution is None:
        return DifficultySampler.balanced()
    return DifficultySampler.weighted(parse_visual_difficulty_distribution(raw_distribution), seed=seed)


def sample_scan_effect_params(bucket: str, *, seed: int | None, sample_index: int) -> ScanEffectParams:
    profiles = default_scan_profiles()
    profile = profiles.get(bucket)
    if profile is None:
        raise ValueError(f"Unknown scan difficulty bucket: {bucket}")
    bucket_offset = BUCKETS.index(bucket) * 7_919
    rng = random.Random((seed or 0) + int(sample_index) * 104_729 + bucket_offset)
    return profile.sample(rng)


def _apply_paper_tint(image: Image.Image, tint: int) -> Image.Image:
    tinted = Image.new("RGB", image.size, color=(int(tint), int(tint), max(0, int(tint) - 4)))
    return Image.blend(tinted, image.convert("RGB"), 0.82)


def _apply_speckle(image: Image.Image, amount: float, rng: random.Random) -> Image.Image:
    if amount <= 0:
        return image
    width, height = image.size
    count = max(1, int(width * height * amount))
    array = np.asarray(image.convert("RGB")).copy()
    xs = np.array([rng.randrange(width) for _ in range(count)])
    ys = np.array([rng.randrange(height) for _ in range(count)])
    values = np.array([rng.randrange(35, 230) for _ in range(count)], dtype=np.uint8)
    array[ys, xs] = np.stack([values, values, values], axis=1)
    return Image.fromarray(array, mode="RGB")


def _apply_scan_lines(image: Image.Image, alpha: float) -> Image.Image:
    if alpha <= 0:
        return image
    overlay = Image.new("RGB", image.size, color=(255, 255, 255))
    pixels = overlay.load()
    width, height = image.size
    for y in range(0, height, 6):
        shade = 210
        for x in range(width):
            pixels[x, y] = (shade, shade, shade)
    return Image.blend(image.convert("RGB"), overlay, max(0.0, min(1.0, alpha)))


def apply_scan_effects(
    image: Image.Image,
    params: ScanEffectParams,
    *,
    rng: random.Random | None = None,
) -> Image.Image:
    local_rng = rng or random.Random()
    result = _apply_paper_tint(image, params.paper_tint)
    result = ImageEnhance.Contrast(result).enhance(params.contrast_factor)
    result = _apply_speckle(result, params.speckle_amount, local_rng)
    result = _apply_scan_lines(result, params.scan_line_alpha)
    if params.blur_radius > 0:
        result = result.filter(ImageFilter.GaussianBlur(radius=params.blur_radius))
    if abs(params.skew_degrees) > 1e-6:
        result = result.rotate(
            params.skew_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(params.paper_tint, params.paper_tint, max(0, params.paper_tint - 4)),
        )
    return result.convert("RGB")
```

- [ ] **Step 4: Run visual difficulty tests**

Run:

```bash
uv run pytest tests/generator/test_visual_difficulty.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/generator/visual_difficulty.py tests/generator/test_visual_difficulty.py
git commit -m "Define scan difficulty primitives for benchmark diversity" -m "Constraint: Scan difficulty must be opt-in and deterministic by seed plus sample index.
Rejected: Renderer-specific scan effects | would duplicate behavior across PIL, html2image, and Playwright renderers
Confidence: high
Scope-risk: narrow
Directive: Keep camera and digital degradation out of this first effect family.
Tested: uv run pytest tests/generator/test_visual_difficulty.py -q
Not-tested: End-to-end markdown generation wiring."
```

---

### Task 2: CLI And Generation Options

**Files:**
- Modify: `src/generation/options.py`
- Modify: `src/cli/generate.py`
- Modify: `tests/generation/test_phase2_generation_modules.py`

- [ ] **Step 1: Add failing options round-trip tests**

Append this test to `tests/generation/test_phase2_generation_modules.py`:

```python
def test_generation_options_round_trip_visual_difficulty_fields() -> None:
    options = GenerationOptions(
        visual_difficulty_mode="balanced-scan",
        visual_difficulty_distribution="easy=1,medium=1,hard=1",
        preview_sheet="./preview.png",
    )

    restored = GenerationOptions.from_dict(options.to_dict())
    kwargs = restored.to_generator_kwargs(sample_start_index=5)

    assert restored.visual_difficulty_mode == "balanced-scan"
    assert restored.visual_difficulty_distribution == "easy=1,medium=1,hard=1"
    assert restored.preview_sheet == "./preview.png"
    assert kwargs["visual_difficulty_mode"] == "balanced-scan"
    assert kwargs["visual_difficulty_distribution"] == "easy=1,medium=1,hard=1"
    assert "preview_sheet" not in kwargs
```

Add this import near the other `generation.*` imports in `tests/generation/test_phase2_generation_modules.py`:

```python
from generation.options import GenerationOptions
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
uv run pytest tests/generation/test_phase2_generation_modules.py::test_generation_options_round_trip_visual_difficulty_fields -q
```

Expected: FAIL with `TypeError: GenerationOptions.__init__() got an unexpected keyword argument 'visual_difficulty_mode'`.

- [ ] **Step 3: Add fields to `GenerationOptions`**

Modify `src/generation/options.py`:

```python
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
    visual_difficulty_mode: Optional[str] = None
    visual_difficulty_distribution: Any = None
    preview_sheet: Optional[str] = None
    add_noise: Optional[bool] = None
    add_blur: Optional[bool] = None
    seed: Optional[int] = None
```

In `GenerationOptions.from_dict()`, add these constructor arguments before `add_noise`:

```python
visual_difficulty_mode=data.get("visual_difficulty_mode"),
visual_difficulty_distribution=data.get("visual_difficulty_distribution"),
preview_sheet=data.get("preview_sheet"),
```

In `GenerationOptions.to_generator_kwargs()`, add these keys before `seed`:

```python
"visual_difficulty_mode": self.visual_difficulty_mode,
"visual_difficulty_distribution": self.visual_difficulty_distribution,
```

Do not include `preview_sheet` in `to_generator_kwargs()`, because preview generation happens after shard aggregation in `pipeline.py`.

- [ ] **Step 4: Add CLI arguments and context wiring**

Modify `src/cli/generate.py` after the formula source arguments and before `--add-noise`:

```python
parser.add_argument(
    "--visual-difficulty-mode",
    type=str,
    default=None,
    choices=["balanced-scan"],
    help="Opt into visual difficulty effects. Use balanced-scan for scanned-paper easy/medium/hard buckets.",
)
parser.add_argument(
    "--visual-difficulty-distribution",
    action="append",
    default=None,
    help="Visual difficulty bucket weights, e.g. easy=1,medium=1,hard=1. Repeatable.",
)
parser.add_argument(
    "--preview-sheet",
    type=str,
    default=None,
    help="Optional path for a generated preview sheet after generation completes.",
)
```

Modify `build_context_from_args()` in `src/cli/generate.py` by adding these fields to `GenerationOptions(...)`:

```python
visual_difficulty_mode=args.visual_difficulty_mode,
visual_difficulty_distribution=args.visual_difficulty_distribution,
preview_sheet=args.preview_sheet,
```

- [ ] **Step 5: Run targeted options tests**

Run:

```bash
uv run pytest tests/generation/test_phase2_generation_modules.py::test_generation_options_round_trip_visual_difficulty_fields -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/generation/options.py src/cli/generate.py tests/generation/test_phase2_generation_modules.py
git commit -m "Expose scan difficulty generation options" -m "Constraint: Existing generation commands must keep their current behavior unless the new mode is selected.
Rejected: Preview sheet as generator kwargs | preview needs final aggregate metadata after shards are rebuilt
Confidence: high
Scope-risk: narrow
Directive: Keep --visual-difficulty-mode choices narrow until additional effect families exist.
Tested: uv run pytest tests/generation/test_phase2_generation_modules.py::test_generation_options_round_trip_visual_difficulty_fields -q
Not-tested: Full CLI invocation."
```

---

### Task 3: Generator Metadata And Image Post-Processing

**Files:**
- Modify: `src/generator/generator.py`
- Modify: `tests/generator/test_dynamic_templates.py`

- [ ] **Step 1: Write failing generator integration test**

Append this test to `tests/generator/test_dynamic_templates.py`:

```python
def test_generate_single_applies_scan_difficulty_metadata_and_effects(monkeypatch) -> None:
    generator = Generator.__new__(Generator)
    generator.template_specs = [
        TemplateSpec(
            template_id="scan-template",
            family="sections",
            mode="sections",
            complexity=1,
            source="test",
            weight=1.0,
            version="1",
            blueprint={},
        )
    ]
    generator.template_catalog = None
    generator.template_counts = Counter()
    generator.family_counts = Counter()
    generator.novelty_window = 8
    generator.novelty_threshold = 1.0
    generator.novelty_max_attempts = 1
    generator._recent_signatures = deque(maxlen=generator.novelty_window)
    generator.base_seed = 42
    generator.noise_ratio = 0.0
    generator.blur_ratio = 0.0
    generator.style_profile = "balanced"
    generator.markdown_renderer = "pil"
    generator.similar_char_ratio = 0.0
    generator.visual_difficulty_mode = "balanced-scan"
    generator.visual_difficulty_distribution = None
    generator._visual_difficulty_sampler = generator_module.DifficultySampler.balanced()
    generator._seed_for_sample = lambda _seed: None
    generator._derive_sample_seed = lambda _sample_index, _attempt: 123
    generator._select_template_spec = lambda: (generator.template_specs[0], 1.0)
    generator._mutate_text_generator_sections = lambda markdown, _ratio, merge_order: (markdown, 0)

    class _StubDataGenerator:
        @staticmethod
        def generate_markdown(template_id: str, template_spec: TemplateSpec) -> str:
            return "# Heading\n\nBody"

        @staticmethod
        def pop_merge_order() -> list[str]:
            return ["text"]

    generator.data_generator = _StubDataGenerator()

    monkeypatch.setattr(generator_module, "random_style", lambda _profile: generator_module.MarkdownStyle())
    monkeypatch.setattr(generator_module, "markdown_to_json_ast", lambda markdown_text: [{"raw": markdown_text}])

    class _StubRenderer:
        def __init__(self, _font_path, style):
            self.style = style

        def render(self, markdown_text: str):
            return Image.new("RGB", (120, 80), color=(255, 255, 255))

    monkeypatch.setattr(generator_module, "MarkdownRenderer", _StubRenderer)
    generator.font_paths = ["/tmp/dummy-font.ttf"]

    image, metadata = generator.generate_single(sample_index=2)

    assert metadata["visual_difficulty"] == "hard"
    assert metadata["visual_effect_family"] == "scanned_paper"
    assert metadata["scan_effect_params"]["bucket"] == "hard"
    assert image.size == (120, 80)
    assert image.getbbox() is not None
```

Also update the module-level imports in `tests/generator/test_dynamic_templates.py` after `parse_coverage_targets`:

```python
DifficultySampler = generator_module.DifficultySampler
```

- [ ] **Step 2: Run the failing generator test**

Run:

```bash
uv run pytest tests/generator/test_dynamic_templates.py::test_generate_single_applies_scan_difficulty_metadata_and_effects -q
```

Expected: FAIL with `AttributeError: module 'generator.generator' has no attribute 'DifficultySampler'` or missing metadata fields.

- [ ] **Step 3: Import visual difficulty helpers**

Modify imports in `src/generator/generator.py`:

```python
from src.generator.visual_difficulty import (
    BALANCED_SCAN_MODE,
    VISUAL_EFFECT_FAMILY_SCANNED,
    DifficultySampler,
    apply_scan_effects,
    make_scan_sampler,
    parse_visual_difficulty_distribution,
    sample_scan_effect_params,
)
```

Add `"DifficultySampler"` to `__all__` so tests can access it through `generator_module`:

```python
"DifficultySampler",
```

- [ ] **Step 4: Add generator configuration state**

In `Generator.__init__()` after `self.base_seed = None`, add:

```python
self.visual_difficulty_mode: Optional[str] = None
self.visual_difficulty_distribution: Any = None
self._visual_difficulty_sampler: Optional[DifficultySampler] = None
```

Add this method to `Generator` after `_configure_content_sources()`:

```python
def _configure_visual_difficulty(self, **kwargs) -> None:
    mode = kwargs.get("visual_difficulty_mode")
    self.visual_difficulty_mode = str(mode).strip().lower() if mode else None
    self.visual_difficulty_distribution = kwargs.get("visual_difficulty_distribution")

    if self.visual_difficulty_mode is None:
        self._visual_difficulty_sampler = None
        return

    if self.visual_difficulty_mode != BALANCED_SCAN_MODE:
        logger.warning(
            "Unknown visual difficulty mode '%s'. Visual difficulty effects disabled.",
            self.visual_difficulty_mode,
        )
        self.visual_difficulty_mode = None
        self._visual_difficulty_sampler = None
        return

    if self.visual_difficulty_distribution is None:
        self._visual_difficulty_sampler = DifficultySampler.balanced()
    else:
        distribution = parse_visual_difficulty_distribution(self.visual_difficulty_distribution)
        self._visual_difficulty_sampler = make_scan_sampler(distribution, seed=self.base_seed)
```

In `_configure_generation()`, call it after `_configure_content_sources(**kwargs)`:

```python
self._configure_visual_difficulty(**kwargs)
```

- [ ] **Step 5: Add image post-processing helper**

Add this method to `Generator` before `generate_single()`:

```python
def _apply_visual_difficulty(
    self,
    image: Image.Image,
    *,
    sample_index: int,
) -> tuple[Image.Image, dict[str, Any]]:
    if self.visual_difficulty_mode != BALANCED_SCAN_MODE or self._visual_difficulty_sampler is None:
        return image, {}

    bucket = self._visual_difficulty_sampler.select(sample_index)
    params = sample_scan_effect_params(
        bucket,
        seed=self.base_seed,
        sample_index=sample_index,
    )
    rng_seed = (self.base_seed or 0) + int(sample_index) * 104_729
    transformed = apply_scan_effects(image, params, rng=random.Random(rng_seed))
    return transformed, {
        "visual_difficulty": bucket,
        "visual_effect_family": VISUAL_EFFECT_FAMILY_SCANNED,
        "scan_effect_params": params.to_dict(),
    }
```

In `generate_single()`, after `image = renderer.render(markdown_text)`, add:

```python
image, visual_metadata = self._apply_visual_difficulty(image, sample_index=sample_index)
```

In the `metadata = { ... }` block after `"image_height": image.height,`, add:

```python
**visual_metadata,
```

- [ ] **Step 6: Run targeted generator tests**

Run:

```bash
uv run pytest tests/generator/test_dynamic_templates.py::test_generate_single_applies_scan_difficulty_metadata_and_effects tests/generator/test_dynamic_templates.py::test_generate_single_metadata_does_not_include_a4_clipping_flags -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/generator/generator.py tests/generator/test_dynamic_templates.py
git commit -m "Apply scan difficulty effects during markdown generation" -m "Constraint: Scan effects must run after rendering so all renderer backends share the same benchmark difficulty behavior.
Rejected: Adding scan logic to MarkdownRenderer | would make Playwright and PIL behavior diverge
Confidence: high
Scope-risk: moderate
Directive: Do not add visual metadata when visual difficulty mode is disabled.
Tested: uv run pytest tests/generator/test_dynamic_templates.py::test_generate_single_applies_scan_difficulty_metadata_and_effects tests/generator/test_dynamic_templates.py::test_generate_single_metadata_does_not_include_a4_clipping_flags -q
Not-tested: Preview sheet generation."
```

---

### Task 4: Preview Sheet Builder And Pipeline Hook

**Files:**
- Create: `src/generation/preview_sheet.py`
- Create: `tests/generation/test_preview_sheet.py`
- Modify: `src/pipeline.py`

- [ ] **Step 1: Write failing preview sheet test**

Create `tests/generation/test_preview_sheet.py`:

```python
import json
from pathlib import Path

from PIL import Image

from generation.preview_sheet import build_preview_sheet


def test_build_preview_sheet_groups_visual_difficulty_samples(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []

    for index, bucket in enumerate(["easy", "medium", "hard"]):
        image_path = image_dir / f"{bucket}.png"
        Image.new("RGB", (80, 120), color=(240 - index * 30, 240, 240)).save(image_path)
        rows.append(
            {
                "file_name": str(image_path),
                "sample_index": index,
                "visual_difficulty": bucket,
                "visual_effect_family": "scanned_paper",
            }
        )

    with open(metadata_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_path = tmp_path / "preview.png"

    written = build_preview_sheet(metadata_path=metadata_path, output_path=output_path, samples_per_bucket=2)

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    preview = Image.open(output_path)
    assert preview.width > preview.height
```

- [ ] **Step 2: Run the failing preview test**

Run:

```bash
uv run pytest tests/generation/test_preview_sheet.py -q
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'generation.preview_sheet'`.

- [ ] **Step 3: Implement preview sheet builder**

Create `src/generation/preview_sheet.py`:

```python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

BUCKET_ORDER = ("easy", "medium", "hard")


def _load_rows(metadata_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(metadata_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _fit_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = image.convert("RGB")
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color=(250, 250, 250))
    x = (size[0] - thumb.width) // 2
    y = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def build_preview_sheet(
    *,
    metadata_path: str | Path,
    output_path: str | Path,
    samples_per_bucket: int = 3,
    thumbnail_size: tuple[int, int] = (220, 300),
) -> Path:
    metadata_file = Path(metadata_path)
    destination = Path(output_path)
    rows = _load_rows(metadata_file)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = row.get("visual_difficulty")
        if bucket in BUCKET_ORDER:
            grouped[str(bucket)].append(row)

    selected: list[tuple[str, dict[str, Any]]] = []
    for bucket in BUCKET_ORDER:
        for row in grouped.get(bucket, [])[:samples_per_bucket]:
            selected.append((bucket, row))

    if not selected:
        raise ValueError(f"No visual difficulty samples found in '{metadata_file}'")

    label_height = 34
    padding = 16
    cols = max(1, samples_per_bucket)
    rows_count = len(BUCKET_ORDER)
    width = padding + cols * (thumbnail_size[0] + padding)
    height = padding + rows_count * (thumbnail_size[1] + label_height + padding)
    sheet = Image.new("RGB", (width, height), color=(235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    by_bucket = {bucket: grouped.get(bucket, [])[:samples_per_bucket] for bucket in BUCKET_ORDER}
    for row_index, bucket in enumerate(BUCKET_ORDER):
        y = padding + row_index * (thumbnail_size[1] + label_height + padding)
        draw.text((padding, y), bucket.upper(), fill=(20, 20, 20), font=font)
        for col_index, row in enumerate(by_bucket[bucket]):
            image_path = Path(str(row["file_name"]))
            with Image.open(image_path) as source:
                thumbnail = _fit_thumbnail(source, thumbnail_size)
            x = padding + col_index * (thumbnail_size[0] + padding)
            sheet.paste(thumbnail, (x, y + label_height))
            sample_label = f"sample {row.get('sample_index', '')}"
            draw.text((x, y + label_height + thumbnail_size[1] - 14), sample_label, fill=(20, 20, 20), font=font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination
```

- [ ] **Step 4: Hook preview sheet into pipeline**

Modify imports in `src/pipeline.py`:

```python
from src.generation.preview_sheet import build_preview_sheet
```

After `rebuild_aggregate_outputs(...)` and before `manifest.mark_finished()`, add:

```python
    if context.generation.preview_sheet:
        preview_path = build_preview_sheet(
            metadata_path=task_output_dir / "metadata.jsonl",
            output_path=Path(context.generation.preview_sheet),
        )
        logger.info("Preview sheet written to %s", preview_path)
```

- [ ] **Step 5: Run preview tests**

Run:

```bash
uv run pytest tests/generation/test_preview_sheet.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/generation/preview_sheet.py src/pipeline.py tests/generation/test_preview_sheet.py
git commit -m "Build preview sheets for scan difficulty validation" -m "Constraint: Preview sheets must use aggregate metadata so they can sample across shards.
Rejected: Per-shard preview generation | does not validate final dataset-level bucket coverage
Confidence: high
Scope-risk: narrow
Directive: Fail when --preview-sheet is requested and no visual difficulty samples exist.
Tested: uv run pytest tests/generation/test_preview_sheet.py -q
Not-tested: Full generation smoke command."
```

---

### Task 5: End-To-End Focused Tests

**Files:**
- Modify: `tests/generator/test_streaming_generation.py`
- Modify: `tests/generation/test_sharding.py`

- [ ] **Step 1: Add metadata streaming coverage with visual fields**

Append this test to `tests/generator/test_streaming_generation.py`:

```python
def test_markdown_dataset_generator_preserves_visual_difficulty_metadata(tmp_path: Path, monkeypatch) -> None:
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "dummy.ttf").write_bytes(b"font")

    dataset_generator = MarkdownDatasetGenerator(
        output_dir=str(tmp_path / "markdown_dataset"),
        font_dir=str(font_dir),
        lang="ko",
    )
    dataset_generator._markdown_generator = FakeMarkdownGenerator(dataset_generator.output_dir / "markdown")

    def fake_generate_single(sample_index: int = 0, **kwargs):
        image = Image.new("RGB", (100, 120), color="white")
        metadata = {
            "GT_markdown": f"# Sample {sample_index}",
            "GT_json": [{"type": "heading", "index": sample_index}],
            "format": "markdown",
            "sample_index": sample_index,
            "visual_difficulty": ["easy", "medium", "hard"][sample_index % 3],
            "visual_effect_family": "scanned_paper",
            "scan_effect_params": {"bucket": ["easy", "medium", "hard"][sample_index % 3]},
        }
        return image, metadata

    dataset_generator._markdown_generator.generate_single = fake_generate_single
    monkeypatch.setattr(
        "generation.markdown_dataset.attach_unified_ground_truth",
        lambda _fmt, meta: dict(meta, GT_json={"kind": "markdown"}),
    )

    result = dataset_generator.run(num_images=3, options=GenerationOptions(visual_difficulty_mode="balanced-scan"))

    assert result == str(dataset_generator.output_dir)
    rows = [
        json.loads(line)
        for line in (dataset_generator.output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["visual_difficulty"] for row in rows] == ["easy", "medium", "hard"]
    stats = json.loads((dataset_generator.output_dir / "realism_stats.json").read_text(encoding="utf-8"))
    assert stats["field_presence"]["visual_difficulty"] == 3
```

Add `GenerationOptions` to imports in `tests/generator/test_streaming_generation.py`:

```python
GenerationOptions = importlib.import_module("generation.options").GenerationOptions
```

- [ ] **Step 2: Add aggregate metadata coverage**

Append this assertion to `tests/generation/test_sharding.py::test_rebuild_aggregate_outputs_concatenates_completed_shards` inside the row dict construction:

```python
"visual_difficulty": ["easy", "medium"][offset % 2],
"visual_effect_family": "scanned_paper",
```

Append this assertion after reading `stats`:

```python
assert stats["field_presence"]["visual_difficulty"] == 4
```

- [ ] **Step 3: Run focused streaming and sharding tests**

Run:

```bash
uv run pytest tests/generator/test_streaming_generation.py::test_markdown_dataset_generator_preserves_visual_difficulty_metadata tests/generation/test_sharding.py::test_rebuild_aggregate_outputs_concatenates_completed_shards -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/generator/test_streaming_generation.py tests/generation/test_sharding.py
git commit -m "Cover visual difficulty metadata in generation outputs" -m "Constraint: Metadata must survive shard and aggregate output paths for benchmark analysis.
Rejected: Only testing Generator.generate_single | would miss dataset and aggregate metadata writers
Confidence: high
Scope-risk: narrow
Directive: Keep visual difficulty fields visible to realism_stats field_presence.
Tested: uv run pytest tests/generator/test_streaming_generation.py::test_markdown_dataset_generator_preserves_visual_difficulty_metadata tests/generation/test_sharding.py::test_rebuild_aggregate_outputs_concatenates_completed_shards -q
Not-tested: Manual preview inspection."
```

---

### Task 6: Documentation And Manual Smoke

**Files:**
- Modify: `docs/generation.md`

- [ ] **Step 1: Update generation docs**

In `docs/generation.md`, under `### Rendering and OCR Noise`, add:

````markdown
### Visual Difficulty Profiles

Scanned-paper difficulty can be enabled for benchmark-focused datasets:

```bash
uv run main.py generate \
  --lang "ko" \
  --size 9 \
  --seed 42 \
  --visual-difficulty-mode balanced-scan \
  --preview-sheet ./data/ko/images_markdown/scan-preview.png
```

`--visual-difficulty-mode balanced-scan` applies scanned-paper post-processing after markdown rendering. The default enabled distribution is balanced across `easy`, `medium`, and `hard`, using global sample index round-robin selection so completed run counts differ by at most one.

Use `--visual-difficulty-distribution easy=2,medium=1,hard=1` to override the default distribution. Custom weights are normalized and sampled deterministically from seed and global sample index.

When enabled, metadata includes:

- `visual_difficulty`
- `visual_effect_family`
- `scan_effect_params`

`--preview-sheet` writes a grid image after shard aggregation so you can visually inspect representative samples from each bucket.
````

- [ ] **Step 2: Run documentation-adjacent checks**

Run:

```bash
uv run pytest tests/generator/test_visual_difficulty.py tests/generator/test_dynamic_templates.py::test_generate_single_applies_scan_difficulty_metadata_and_effects tests/generation/test_preview_sheet.py -q
```

Expected: PASS.

- [ ] **Step 3: Run manual smoke generation**

Run:

```bash
uv run main.py generate \
  --lang ko \
  --size 9 \
  --seed 42 \
  --output-dir /tmp/synthetic-ocr-scan-smoke \
  --markdown-renderer pil \
  --visual-difficulty-mode balanced-scan \
  --preview-sheet /tmp/synthetic-ocr-scan-smoke/scan-preview.png
```

Expected:

- Command exits with status 0.
- `/tmp/synthetic-ocr-scan-smoke/ko/images_markdown/metadata.jsonl` exists.
- `/tmp/synthetic-ocr-scan-smoke/scan-preview.png` exists and is non-empty.

- [ ] **Step 4: Inspect generated metadata counts**

Run:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path('/tmp/synthetic-ocr-scan-smoke/ko/images_markdown/metadata.jsonl')
counts = Counter()
for line in path.read_text(encoding='utf-8').splitlines():
    row = json.loads(line)
    counts[row.get('visual_difficulty')] += 1
print(dict(counts))
assert counts == {'easy': 3, 'medium': 3, 'hard': 3}
PY
```

Expected: prints `{'easy': 3, 'medium': 3, 'hard': 3}` and exits with status 0.

- [ ] **Step 5: Commit docs**

```bash
git add docs/generation.md
git commit -m "Document scan difficulty generation workflow" -m "Constraint: Users need a concrete smoke command and metadata expectations for benchmark-focused generation.
Rejected: README-only documentation | generation details belong in docs/generation.md
Confidence: high
Scope-risk: narrow
Directive: Keep OCR recalibration documented as deferred until implemented.
Tested: uv run pytest tests/generator/test_visual_difficulty.py tests/generator/test_dynamic_templates.py::test_generate_single_applies_scan_difficulty_metadata_and_effects tests/generation/test_preview_sheet.py -q; uv run main.py generate --lang ko --size 9 --seed 42 --output-dir /tmp/synthetic-ocr-scan-smoke --markdown-renderer pil --visual-difficulty-mode balanced-scan --preview-sheet /tmp/synthetic-ocr-scan-smoke/scan-preview.png
Not-tested: Playwright smoke generation."
```

---

### Task 7: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run targeted test suite**

Run:

```bash
uv run pytest \
  tests/generator/test_visual_difficulty.py \
  tests/generator/test_dynamic_templates.py \
  tests/generator/test_streaming_generation.py \
  tests/generation/test_preview_sheet.py \
  tests/generation/test_sharding.py \
  tests/generation/test_phase2_generation_modules.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run CLI help smoke**

Run:

```bash
uv run main.py generate --help | rg "visual-difficulty|preview-sheet"
```

Expected: output includes `--visual-difficulty-mode`, `--visual-difficulty-distribution`, and `--preview-sheet`.

- [ ] **Step 3: Check git diff**

Run:

```bash
git diff --check
```

Expected: no output and exit status 0.

- [ ] **Step 4: Review final changed files**

Run:

```bash
git status --short
```

Expected: clean working tree after the Task 6 commit, or only intentionally uncommitted files if the implementation executor deferred a commit for review.

---

## Self-Review Notes

- Spec coverage: Task 1 covers profiles, parser, sampler, parameters, and scan effects. Task 2 covers CLI/options. Task 3 covers post-render effects and metadata. Task 4 covers preview sheet generation after aggregation. Task 5 covers metadata persistence. Task 6 covers documentation and manual smoke validation. Task 7 covers final verification.
- Scope check: camera-capture buckets, digital degradation buckets, OCR-score recalibration, and broader content/layout curriculum are not in this plan.
- Type consistency: the plan consistently uses `visual_difficulty_mode`, `visual_difficulty_distribution`, `preview_sheet`, `visual_difficulty`, `visual_effect_family`, and `scan_effect_params`.
