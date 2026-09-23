# Real-World Distribution Profiles

By default the generator samples styles, noise and document families from hand-tuned uniform ranges. A **distribution profile** replaces those ranges with distributions meant to match real OCR data, and records every sampled value in metadata. You can then measure and compare the output against real images.

## Quick Start

```bash
# Generate with the bundled general-purpose profile
uv run main.py generate --lang ko --size 1000 --seed 42 \
  --distribution-profile real_world_v1

# Scan-heavy Korean administrative documents
uv run main.py generate --lang ko --size 1000 --seed 42 \
  --distribution-profile ko_admin_scan_v1

# Publish; the dataset card gets a "Sample Distribution" section
uv run main.py publish --generated-path ./data/ko/images_markdown --repo-id you/your-dataset
```

`--distribution-profile` accepts a bundled name from `configs/generator/distributions/` or a path to your own YAML file. Without the flag, generation behaves exactly as before.

## What a Profile Controls

| Section | Effect |
|---|---|
| `family_mix` | Target share per template family. Used as coverage targets unless you pass `--coverage-target` explicitly |
| `block_weights` | Relative weights for filling non-required block slots (paragraph vs table vs list ...) |
| `typography.body_font_pt` | Physical body font size on a virtual A4 page; heading and code sizes scale with it |
| `typography.line_spacing` | Line spacing |
| `typography.colored_background` | Probability of keeping a tinted page background (otherwise white paper) |
| `capture_channels.<name>.weight` | Mix of `born_digital` / `scanned` / `photographed` pages |
| `capture_channels.<name>.dpi` | Target resolution. Playwright renders with a matching device scale factor, so a 300 dpi page is about 2480 px wide |
| `capture_channels.<name>.degradations` | Per-channel degradation parameters (see below) |

When a profile is active, the legacy `--add-noise` / `--add-blur` toggles are ignored and the profile's degradations apply instead.

### Degradations (`src/generator/degradation.py`)

| Key | Meaning |
|---|---|
| `paper_tint` | Warm or grey paper tint |
| `ink_fade` | Pull ink towards the paper colour (0 to 1) |
| `bleed_through` | Mirrored ghost of the reverse side (0 to 1) |
| `skew_deg` | In-plane rotation in degrees |
| `perspective` | Corner jitter as a fraction of page size; photographed pages also get a desk-coloured border |
| `illumination` | Linear lighting gradient strength |
| `shadow_strength` | Soft shadow over one side |
| `blur_sigma` | Gaussian blur in output pixels |
| `motion_blur_px` | Motion blur kernel length |
| `noise_sigma` | Additive gaussian sensor noise (0 to 255 scale) |
| `speckle_density` | Salt-and-pepper dust (fraction of pixels) |
| `grayscale` / `binarize` | Single-channel output / Otsu bitonal output |
| `jpeg_quality` | JPEG re-encode quality (`null` means none) |

### Distribution spec syntax

Any value can be a constant or a distribution:

```yaml
dpi: {choices: [150, 200, 300], weights: [0.2, 0.3, 0.5]}
skew_deg: {normal: [0.0, 0.5], clip: [-3, 3], round: 2}
blur_sigma: {lognormal: [-0.9, 0.45], clip: [0, 1.6]}
ink_fade: {beta: [2, 6], range: [0, 0.45]}
grayscale: {p: 0.5}
dpi: {histogram: {bins: [100, 150, 200, 300], weights: [10, 30, 60]}, round: 0}
```

## Added Metadata Columns

Profile runs add these per-sample columns, which are uploaded to the Hub and usable for filtering or per-bucket evaluation:

- `distribution_profile`
- `capture_channel`
- `target_dpi`
- `render_scale`
- `body_font_pt`
- `colored_background`
- `visual_difficulty` (`easy` / `medium` / `hard`, derived from sampled degradation strength)
- `degradation_params` (JSON)

## Where the Bundled Numbers Come From

Each YAML file marks values as `[sourced]` or `[prior]`.

- **DocLayNet** ([repo](https://github.com/DS4SD/DocLayNet)), 80,863 pages:
  - Category shares: Financial Reports 32%, Manuals 21%, Scientific 17%, Laws & Regulations 16%, Patents 8%, Tenders 6%.
  - Element counts: Text 510k, List-item 186k, Section-header 143k, Picture 46k, Table 35k, Formula 25k, ...
  - These determine `real_world_v1.family_mix` (blended with extra forms/operations mass that DocLayNet lacks) and `block_weights` (list items divided by the ~4 items per list block).
- **OmniDocBench** ([repo](https://github.com/opendatalab/OmniDocBench)): its page attributes (`fuzzy_scan`, `watermark`, `colorful_background`) and data-source types informed the capture-channel taxonomy. Per-attribute counts could not be retrieved, so channel weights are priors.
- **ADF scanner skew study** ([paper](https://www.researchgate.net/publication/224341611_Estimating_the_Skew_Angle_of_Scanned_Document_through_Background_Area_Information)): on 300 A4 sheets fed through a scanner, 3 sigma of skew fell within ±1.5°. This gives `scanned.skew_deg ~ N(0, 0.5)`.
- **AI Hub 공공행정문서 OCR** ([page](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=88)): older administrative records with poor scan and photo quality. This informed the direction of `ko_admin_scan_v1`. All of its ratios are priors.

Treat every `[prior]` as a starting point. The next section shows how to replace priors with measurements.

## Calibrating Against Real Data

1. **Measure a reference set** of real images you are allowed to use. The source can be a directory, a generated `metadata.jsonl`, or a streamed HF dataset:

   ```bash
   uv run main.py distribution measure --hf-dataset opendatalab/OmniDocBench --hf-split train \
     --max-images 500 --output stats/real.json --suggest-yaml stats/real_suggest.yaml
   # or: --images /path/to/scans
   ```

   `real_suggest.yaml` contains the specs that can be read directly from images: `dpi` and `skew_deg` histograms, and `grayscale` / `binarize` / `colored_background` rates. Paste them into your profile.

2. **Generate a pilot set and measure it the same way:**

   ```bash
   uv run main.py generate --lang ko --size 300 --seed 1 --distribution-profile my_profile --output-dir ./pilot
   uv run main.py distribution measure --metadata ./pilot/ko/images_markdown/metadata.jsonl --output stats/pilot.json
   ```

3. **Compare:**

   ```bash
   uv run main.py distribution compare --reference stats/real.json --candidate stats/pilot.json --output stats/gap.md
   ```

   The report ranks metrics by normalised Wasserstein distance: under 0.1 is close, 0.1 to 0.3 is noticeable, above 0.3 is different. It also tells you whether the synthetic set is higher or lower on each metric.
   - Tune `blur_sigma` using `laplacian_var`.
   - Tune `noise_sigma` / `speckle_density` using `noise_sigma`.
   - Tune `ink_fade` / `paper_tint` using `contrast` / `background_luma`.
   - Tune channel weights and `perspective` using `aspect_ratio` / `abs_skew_deg`.
   - Repeat until the gaps are small.

Measured metrics: size, `est_dpi_a4` (width / 8.27 in, which assumes full A4 portrait pages), skew (projection profile), Laplacian variance, Immerkaer noise sigma, background/ink luminance and contrast, ink ratio, Hasler–Süsstrunk colourfulness, and grayscale / binary / coloured-background flags. Blur and noise are measured at a fixed 1000 px width, so they are comparable across resolutions.

**Limitation:** the skew estimator is reliable for scans (within about 0.1° of the true value). On photographed pages, perspective and desk borders dominate and the estimate is not meaningful.

## Beyond Visual Realism

Profiles fix *how pages look* and *which structures appear*. The text itself still comes from the corpus. For realistic content, generate a language corpus first (`uv run main.py corpus generate ...`) so pages contain real-language sentences instead of placeholder text. To confirm the benchmark is realistic, check that model rankings on the synthetic set correlate (Spearman) with rankings on a real benchmark.
