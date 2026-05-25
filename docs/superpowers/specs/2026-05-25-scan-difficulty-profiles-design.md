# Scan Difficulty Profiles Design

## Goal

Improve synthetic document quality for benchmark-first OCR/VLM datasets by adding explicit scanned-paper visual difficulty buckets. The first implementation should make generated samples more diverse and more discriminative while still remaining usable as training data.

The initial scope is scanned-paper visual difficulty only. Camera-capture artifacts, digital compression artifacts, and OCR-score-based bucket recalibration are intentionally deferred.

## Success Criteria

- Generation can opt into scanned-paper visual difficulty profiles without changing default behavior.
- Enabled generation samples `easy`, `medium`, and `hard` difficulty buckets with balanced default counts that differ by at most one sample per completed run.
- Each generated sample records the selected visual difficulty, effect family, and concrete effect parameters in metadata.
- Effects are deterministic for the same seed and sample index, including shard/resume runs.
- A preview sheet can be generated to visually inspect representative samples across buckets.
- Tests cover distribution parsing, deterministic bucket selection, parameter ranges, metadata, and preview sheet creation.

## Architecture

### ScanDifficultyProfile

Define named scanned-paper profiles for `easy`, `medium`, and `hard`. Each profile owns numeric ranges for visual effects such as:

- paper tint
- speckle or scanner noise
- blur radius
- skew angle
- contrast variation
- scan-line intensity

The profile should produce sampled effect parameters, not apply image effects directly. This keeps profile configuration separate from image processing.

### DifficultySampler

Select the visual difficulty bucket for each sample. The default enabled distribution is balanced across `easy`, `medium`, and `hard`.

For the balanced default, use global sample index round-robin selection so completed run counts differ by at most one sample. For custom distributions, use deterministic weighted selection from sample index and seed-derived RNG state. For example, a balanced dataset with `size=300` should produce `100/100/100`, and rerunning the same generation with the same seed should assign the same bucket to each sample.

Invalid distribution input should warn and fall back to balanced defaults. Non-normalized but valid weights should be normalized.

### ScanEffectPipeline

Apply scanned-paper effects after markdown rendering and before the image is saved. This keeps the current markdown content and renderer responsibilities stable.

The pipeline receives a PIL image and sampled scan effect parameters, then returns the transformed image. It should fail loudly if effect application fails while the mode is enabled, because silently saving the clean image would corrupt benchmark difficulty distribution.

### Metadata

When scan difficulty is enabled, sample metadata should include:

- `visual_difficulty`: `easy`, `medium`, or `hard`
- `visual_effect_family`: `scanned_paper`
- `scan_effect_params`: concrete sampled parameter values

Default generation without the new mode should preserve current behavior and avoid unexpected schema churn.

## CLI And Options

Add a minimal CLI surface:

- `--visual-difficulty-mode balanced-scan`
- `--visual-difficulty-distribution easy=0.34,medium=0.33,hard=0.33`
- `--preview-sheet PATH`

`balanced-scan` enables the scanned-paper profiles. If the distribution option is omitted, use balanced `easy/medium/hard` sampling. `--preview-sheet` writes a grid image containing representative generated samples grouped or labeled by bucket after generation completes.

The options should flow through `GenerationOptions` into the markdown generator. Existing generation commands should behave the same unless the new mode is explicitly selected.

## Data Flow

The enabled generation flow is:

```text
markdown content
  -> renderer image
  -> scan difficulty bucket selection
  -> scan effect parameter sampling
  -> scan effect application
  -> metadata attachment
  -> shard/root metadata and preview sheet
```

Shard and resume behavior must remain deterministic by using the global sample index rather than shard-local index alone.

## Error Handling

- Unknown visual difficulty mode: warn and disable visual difficulty effects.
- Invalid or empty distribution: warn and use balanced defaults.
- Distribution weights that do not sum to 1: normalize them.
- Unknown bucket names in distribution: ignore them and warn.
- Effect application failure while enabled: fail the sample generation instead of silently saving a clean image.
- Preview sheet failure: fail the command when `--preview-sheet` is explicitly requested, because the user asked for a validation artifact.

## Validation Plan

Unit tests:

- Parse valid and invalid visual difficulty distributions.
- Confirm balanced bucket selection covers `easy`, `medium`, and `hard` with counts differing by at most one.
- Confirm deterministic bucket selection for seed and sample index.
- Confirm sampled scan parameters stay inside profile ranges.
- Confirm metadata fields are present only when the feature is enabled.

Integration-style tests:

- Run a small markdown generation with scan difficulty enabled and assert all buckets appear.
- Assert metadata includes `visual_difficulty`, `visual_effect_family`, and `scan_effect_params`.
- Generate a preview sheet from a small set of images and assert the output file exists and is non-empty.

Manual smoke validation:

```bash
uv run main.py generate \
  --lang ko \
  --size 9 \
  --seed 42 \
  --visual-difficulty-mode balanced-scan \
  --preview-sheet ./data/ko/images_markdown/scan-preview.png
```

Review the preview sheet for visibly distinct scanned-paper difficulty levels.

## Deferred Work

- Camera-capture visual difficulty buckets.
- Digital degradation buckets for compression, downscale-upscale, and screenshot artifacts.
- OCR-score-based post-generation bucket recalibration.
- Broader content/layout difficulty curriculum.
