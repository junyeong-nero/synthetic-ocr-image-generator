# Real-World Distribution Profiles

By default the generator samples styles, noise and document families from hand-tuned uniform ranges. A **distribution profile** replaces those ranges with distributions meant to match real OCR data, and records every sampled value in metadata. You can then measure and compare the output against real images.

## Quick Start

```bash
# 1) Real-language text: import Korean Wikipedia (CC BY-SA 3.0) as corpus
uv run main.py corpus import-wikitext --kowikitext-split dev --kowikitext-split test --lang ko

# 2) Generate with the calibrated general-purpose profile
uv run main.py generate --lang ko --size 1000 --seed 42 \
  --distribution-profile real_world_v2

# Scan-heavy Korean administrative documents
uv run main.py generate --lang ko --size 1000 --seed 42 \
  --distribution-profile ko_admin_scan_v1

# 3) Preview the dataset card locally, then publish
uv run main.py publish --generated-path ./data/ko/images_markdown --repo-id you/your-dataset \
  --license cc-by-sa-3.0 --text-source "Korean WikiText (https://github.com/lovit/kowikitext), CC BY-SA 3.0" \
  --dry-run                                   # writes DATASET_CARD.md, no upload
uv run main.py publish --generated-path ./data/ko/images_markdown --repo-id you/your-dataset \
  --license cc-by-sa-3.0 --text-source "Korean WikiText (https://github.com/lovit/kowikitext), CC BY-SA 3.0"
```

`--distribution-profile` accepts a bundled name from `configs/generator/distributions/` or a path to your own YAML file. Without the flag, generation behaves exactly as before.

A profile needs a real text corpus (`data/corpus/<lang>/paragraphs.txt`, see [Real-Language Corpus](#real-language-corpus)). Without one, paragraphs would fall back to Faker placeholder text, which is Latin lorem ipsum even for `ko`/`ja`, so generation stops with an error instead.

## What a Profile Controls

| Section | Effect |
|---|---|
| `family_mix` | Share of pages per template family. Each sample draws its family from this mix, so the shares hold for any run or shard size. Passing `--coverage-target` explicitly switches back to the legacy balancing |
| `template_weights` | Multiplier on a template's catalog weight (unlisted: 1.0, `0` excludes it). Within a family, templates are drawn in proportion to catalog weight × multiplier. An explicit `--template` is kept even if the profile excludes it |
| `block_weights` | Relative weights for filling non-required block slots (paragraph vs table vs list ...) |
| `typography.body_font_pt` | Physical body font size on a virtual A4 page; heading and code sizes scale with it |
| `typography.line_spacing` | Line spacing |
| `typography.colored_background` | Probability of keeping a tinted page background (otherwise white paper) |
| `typography.spacing_scale` | Multiplier for CSS block/heading margins (real pages are tighter than web CSS) |
| `typography.ink_gray` | Body text grey level (0 = black). Table borders and horizontal rules always use the text ink colour under a profile, so they stay solid after binarization |
| `typography.colored_headings` | Probability that headings keep an accent colour instead of ink colour |
| `typography.text_align` | `left` or `justify` (CJK-friendly justification) for body text. HTML renderers only; the PIL renderer ignores it |
| `typography.word_break` | `normal` or `keep-all` (breaks fall between words, not mid-syllable) for body text. HTML renderers only; the PIL renderer ignores it |
| `typography.table_style` | `web` (legacy: light/ink borders, header shading, zebra rows) \| `grid` (solid ink borders, no zebra, bold centred header) \| `header_shaded` (no vertical rules, shaded header row) \| `booktabs` (thick top/bottom table rules, a header rule, no vertical lines) \| `borderless` (header underline only). Unset keeps `web`. A numeric column's markdown right-alignment (`---:`, see Realistic Table Content below) is an inline style and always wins over these class rules. HTML renderers only |
| `page.margins_mm` | `{top, bottom, left, right}` page margins in millimetres on A4 (210mm wide), converted to CSS padding as `mm / 210 × page width`. Unset sides keep the sampled base style's margin |
| `page.columns` | `1` or `2` CSS columns for the whole page. May be a plain distribution spec, or `{default: <spec>, by_family: {<family>: <spec>, ...}}` to give some template families (e.g. `academic`) a different two-column share than the rest. The `h1` title spans every column; tables and figures are never split across columns (`break-inside: avoid`). Content still flows in document order, so `GT_markdown` (column-major reading order) is unchanged. HTML renderers only |
| `page.column_gap_mm` | Gap between columns in millimetres, used only when `page.columns` is `2`. The gap narrows the columns; it does not widen the page |
| `page.header`, `page.footer` | Probabilities (`{p: ...}` or bare floats), default off. Draw a short running title/organisation/date and footer in the page margins. HTML renderers only; PIL records empty furniture |
| `page.page_number` | Distribution over `none` (default), `dashed` (`- 3 -`), `plain` (`3`), `of_total` (`3 / 12`), `korean` (`3쪽`; plain outside Korean), `english` (`Page 3`). With furniture, a sheet container uses `page.aspect_ratio` to anchor the footer at the bottom; without an aspect ratio it follows the content height. Overflow grows the container and triggers the existing block trimming, preserving GT |
| `content.continuation_page` | Probability, default off. Drop the title and randomly selected leading sections. Record `continuation_page`; recompute `merge_order`, block counts and section count. Legacy compositions without a block-level merge order are left intact |
| `content.start_mid_paragraph` | Probability conditional on a continuation page, default off. Start with a paragraph tail cut at a sentence boundary, when available; record whether applied as `start_mid_paragraph` |
| `fonts.exclude` | File-name substrings of fonts never used as body font (e.g. hairline weights) |
| `fonts.body` / `fonts.heading` | Weighted groups of file-name substrings (e.g. `{Myeongjo: 3, batang: 2, Gothic: 2}`): a group is picked by weight, then a file uniformly among its matches. `fonts.exclude` still narrows the candidates. Falls back to the profile's default (unweighted, exclude-filtered) choice when unset or when no group matches any file. HTML renderers emit a separate `@font-face` for the heading font (h1-h3, table headers); the PIL renderer ignores it |
| `fonts.code` | Same weighted-group syntax (or a bare list, e.g. `[D2Coding]`) for a monospace code/pre font. Not narrowed by `fonts.exclude`, so a face excluded from body text (e.g. the D2Coding monospace font) can still be used here. HTML renderers only |
| `content.section_count_scale` / `extra_blocks_per_section` | Scale sections / add blocks per section for page density |
| `content.paragraph_max_chars` / `paragraph_parts` | Paragraph length (number of corpus paragraphs joined, clip length) |
| `content.similar_char_ratio` | Share of prose characters swapped for look-alikes in both image and GT. The bundled `real_world_v2` and `ko_admin_scan_v1` set `0.0`, since real pages have almost no such typos. `--similar-char-ratio` overrides it; without either, the default is `0.08` |
| `content.table_schemas` | Mapping of table schema name -> selection weight (`financial`, `budget`, `schedule`, `roster`, `order`, `statistics`; see `src/generator/table_schemas.py`). Each schema has semantic, internally-consistent columns (e.g. `order`'s 금액 = 수량 x 단가, with a 합계 row) and ko/en/ja headers, instead of the legacy unrelated columns. Without this key (or without a profile), table generation is unchanged |
| `content.heading_numbering` | `none` (default) / `arabic` (`1.` or `1.1`) / `korean_admin` (`Ⅰ.` / `1.` / `가.`, one pool sampled per document) / `roman` (`I.`, `II.`, ...). Numbers every `## ` section heading in document order. `korean_admin` only applies when `lang` is `ko`; other languages fall back to `arabic`. See `src/generator/document_conventions.py` |
| `content.list_style` | `markdown` (default: unchanged `- item` / `1. item`) or `korean_admin`: bullet lists render as `□` / `○` / `·` lines (one marker per block) and numbered lists as escaped `1\)`, `가)` or `①` lines (never plain `1)`, since python-markdown and mistune disagree about an unescaped digit + `)`), with an occasional trailing `※ ...` note line. Applies to any document shape. `korean_admin` only applies when `lang` is `ko`; other languages fall back to `markdown` |
| `content.law_articles` | Probability (bare float or `{p: ...}`) that a `policy_document`-shaped page's `numbered_list` blocks are rewritten as `제N조(제목) ...` articles with `①②...` clauses instead of following `list_style`. Article numbers increment across the whole document. Korean-only (`lang == "ko"`); recorded per sample as `law_articles_used` |
| `content.genre_corpus` | Probability (bare float or `{p: ...}`) that a page whose `document_shape` has a genre (see [Genre-Specific Corpus](#genre-specific-corpus)) draws its `paragraph` blocks from that genre's corpus file instead of the general `paragraphs` corpus. `0` (default) or a missing genre file both keep the general corpus. Recorded per sample as `content_genre` (the genre name, or `null`) |
| `content.merged_table_ratio` | Share (0-1) of schema tables written into `GT_markdown` as a raw HTML `<table>` block with merged cells instead of a pipe table (see `src/generator/html_table.py`): two-level headers (`statistics` puts 상반기 \| 하반기 under each year, `financial` 당기 \| 전기 under 금액(백만원), `budget` and `schedule` group their amount / date-time columns), row groups (a `schedule` date or `roster` 소속 spanning several rows) and `order`'s 합계 label spanning up to 금액. Numeric cells keep an inline right alignment like `---:`; span headers are centred by the table-style CSS. The block has no blank lines, so it stays one block for page trimming, text mutation (never mutated) and `GT_json` (one `block_html` token); the markdown block metric scores it with TEDS. Needs `content.table_schemas`; `0` or unset keeps pipe tables. HTML renderers only: with `--markdown-renderer pil` the key is ignored, since PIL would draw the tags as text |
| `page.aspect_ratio` | Sheet shape. Short content is padded to a full sheet; content longer than one sheet is cut at a block boundary and re-rendered, like the first page of a multi-page file (`page_trimmed` in metadata). `GT_markdown` always matches the image |
| `capture_channels.<name>.weight` | Mix of `born_digital` / `scanned` / `photographed` pages |
| `capture_channels.<name>.dpi` | Target resolution. Playwright renders with a matching device scale factor, so a 300 dpi page is about 2480 px wide |
| `capture_channels.<name>.degradations` | Per-channel degradation parameters (see below). A channel with `scenarios` samples these first as the base, then a scenario overrides a subset |
| `capture_channels.<name>.scenarios.<scenario>.weight` | Selection weight for this scenario within its channel (`rng.choices` picks one scenario per sample; weights don't need to sum to 1) |
| `capture_channels.<name>.scenarios.<scenario>.dpi` | Overrides the channel's `dpi` for this scenario. Omit to inherit the channel's `dpi` |
| `capture_channels.<name>.scenarios.<scenario>.degradations` | Overrides the channel's `degradations` by key for this scenario; keys it doesn't mention keep the channel-level spec. A channel without `scenarios` behaves exactly as before |

When a profile is active, the legacy `--add-noise` / `--add-blur` toggles are ignored and the profile's degradations apply instead.

### Degradations (`src/generator/degradation.py`, effects in `src/generator/capture_artifacts.py`)

Applied in this order: paper colour (tint, ink fade, bleed-through) → marks
that need the clean, axis-aligned page and are not GT content (highlighter,
stamp, fold lines, punch holes) — `edge_crop`'s real-text ink bounding box is
snapshotted as a mask before these are drawn, so a stamp or punch hole is
never mistaken for protected ink → page-shape geometry (page curl,
perspective, skew), which carries that ink mask through the same warps →
`edge_crop` crops the final, post-geometry image against the warped mask
(never past it) → scan/photo lighting (scanner border, illumination, shadow)
→ optics/sensor (blur, motion blur, toner streaks, noise, speckle) →
grayscale/binarize + JPEG.

| Key | Meaning |
|---|---|
| `paper_tint` | Warm or grey paper tint |
| `ink_fade` | Pull ink towards the paper colour (0 to 1) |
| `bleed_through` | Ghost of the reverse side (0 to 1): the page mirrored, cut into horizontal bands that are shuffled and shifted, then blurred more strongly, so ghost text does not line up with the page's own lines |
| `highlighter` | Translucent yellow bar over 1-3 text lines, found by a horizontal ink projection profile |
| `stamp` | Red organisation seal (circle or rounded square, double ring, short Korean text), slightly rotated, uneven ink, multiply-blended near the lower right or over the last text lines |
| `fold_lines` | Paper crease count (0-2): dark line, light edge, slight brightness step on one side |
| `punch_holes` | 2-3 dark round holes confined to the left margin, left of the page's ink bounding box (never over ink) |
| `edge_crop` | Crop part of the page border (0 to ~0.1), never past the ink bounding box, so GT stays complete; for photographed pages |
| `page_curl` | Smooth vertical warp (0 to ~0.05, `cv2.remap`) simulating book curvature; for photographed pages |
| `skew_deg` | In-plane rotation in degrees |
| `perspective` | Corner jitter as a fraction of page size; photographed pages also get a desk-coloured border |
| `scanner_border` | Dark shadow band along one or two page edges (0 to 1) |
| `illumination` | Linear lighting gradient strength |
| `shadow_strength` | Soft shadow over one side |
| `blur_sigma` | Gaussian blur in output pixels |
| `motion_blur_px` | Motion blur kernel length |
| `toner_streaks` | Count of faint vertical streaks |
| `noise_sigma` | Additive gaussian sensor noise (0 to 255 scale) |
| `speckle_density` | Salt-and-pepper dust (fraction of pixels) |
| `grayscale` / `binarize` | Single-channel output / bitonal output |
| `binarize_method` | Thresholding used when `binarize` is true: `otsu` (default, global), `adaptive` (local mean), or `sauvola` |
| `jpeg_quality` | JPEG re-encode quality (`null` means none) |

### Capture scenarios

Degradation parameters are correlated in reality (an old archive scan is
tinted *and* faded *and* bleeds through *and* is low-dpi; a fax is bitonal
*and* low-dpi *and* speckled), not sampled independently. A channel can add
`scenarios`: named, weighted sub-distributions that override a subset of the
channel's `degradations` (and optionally its `dpi`) together.

```yaml
capture_channels:
  scanned:
    weight: 0.40
    dpi: {choices: [200, 300], weights: [0.3, 0.7]}
    degradations:
      skew_deg: {normal: [-0.1, 0.3], clip: [-1.5, 1.5], round: 2}
      noise_sigma: {lognormal: [0.8, 0.4], clip: [0.0, 8.0], round: 2}
      # ... rest of the base range, sampled by every scanned page
    scenarios:
      office_adf:
        weight: 0.55            # picked ~55% of the time this channel is sampled
        degradations:
          stamp: {p: 0.06}      # added on top of the base range
      fax:
        weight: 0.10
        dpi: {choices: [100, 150], weights: [0.4, 0.6]}   # overrides the channel dpi
        degradations:
          binarize: {p: 1.0}    # overrides the channel's own `binarize` spec
          noise_sigma: {lognormal: [1.0, 0.3], clip: [0.0, 10.0], round: 2}
```

Sampling picks a channel (as before), then, if it has `scenarios`, picks one
by weight and merges its `degradations` over the channel's own by key before
sampling every value. A channel without `scenarios` behaves exactly as
before. `real_world_v3` and `ko_admin_scan_v1` split their `scanned` channel
into `office_adf` / `archive` / `photocopy` / `fax` and their `photographed`
channel into `phone_flat` / `phone_book` / `low_light`; the sampled name is
recorded in metadata as `capture_scenario` (`""` for a channel with no
scenarios).

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

`page.columns` (and any future per-family key) may also be `{default: <spec>, by_family: {<family>: <spec>, ...}}`, where `<family>` matches a `family_mix` key (e.g. `academic`, `technical`). The selected template's family picks its `by_family` spec; a family absent from `by_family`, or no family mix at all, falls back to `default`:

```yaml
page:
  columns:
    default: {choices: [1, 2], weights: [0.95, 0.05]}
    by_family:
      academic: {choices: [1, 2], weights: [0.6, 0.4]}
```

## Added Metadata Columns

Profile runs add these per-sample columns, which are uploaded to the Hub and usable for filtering or per-bucket evaluation:

- `distribution_profile`
- `capture_channel`
- `capture_scenario` (name of the sampled scenario, or `""` for a channel with no `scenarios`)
- `target_dpi`
- `render_scale`
- `body_font_pt`
- `colored_background`
- `visual_difficulty` (`easy` / `medium` / `hard`, derived from sampled degradation strength)
- `degradation_params` (JSON)
- `font_name`, `page_aspect_ratio`, `page_trimmed`
- `heading_font_name`, `code_font_name` (only when `fonts.heading` / `fonts.code` matched a file)
- `table_style` (only when `typography.table_style` is set)
- `page_columns` (only when `page.columns` samples to `2`)
- `page_furniture`: `{header, footer, page_number}` strings, empty strings for absent parts; emitted only when furniture keys are configured. Separate header slots are newline-separated. Furniture is drawn outside the body and excluded from `GT_markdown` and `GT_json`. Evaluation removes whole predicted lines matching these annotations after Unicode/whitespace normalization; strings of at least 10 characters allow fuzzy similarity ≥ 0.9, shorter strings require exact matches. Raw predictions remain available. Samples without this metadata retain legacy scoring.
- `continuation_page`, `start_mid_paragraph`: actual composition outcomes, emitted only when the corresponding content controls are configured

## Where the Bundled Numbers Come From

Each YAML file marks values as `[sourced]` or `[prior]`.

- **DocLayNet** ([repo](https://github.com/DS4SD/DocLayNet)), 80,863 pages:
  - Category shares: Financial Reports 32%, Manuals 21%, Scientific 17%, Laws & Regulations 16%, Patents 8%, Tenders 6%.
  - Element counts: Text 510k, List-item 186k, Section-header 143k, Picture 46k, Table 35k, Formula 25k, ...
  - These determine `real_world_v1.family_mix` (blended with extra forms/operations mass that DocLayNet lacks) and `block_weights` (list items divided by the ~4 items per list block).
- **OmniDocBench** ([repo](https://github.com/opendatalab/OmniDocBench)): its page attributes (`fuzzy_scan`, `watermark`, `colorful_background`) and data-source types informed the capture-channel taxonomy. Per-attribute counts could not be retrieved, so channel weights are priors.
- **ADF scanner skew study** ([paper](https://www.researchgate.net/publication/224341611_Estimating_the_Skew_Angle_of_Scanned_Document_through_Background_Area_Information)): on 300 A4 sheets fed through a scanner, 3 sigma of skew fell within ±1.5°. This gives `scanned.skew_deg ~ N(0, 0.5)`.
- **AI Hub 공공행정문서 OCR** ([page](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=88)): older administrative records with poor scan and photo quality. This informed the direction of `ko_admin_scan_v1`. All of its ratios are priors.
- **Korean HWP / Word page-setup defaults**: `page.margins_mm` mixes the 한글(HWP) default page margins (top 20mm + a 15mm header zone = 35mm effective top margin, bottom 15mm, left/right 30mm) with Word's 1-inch (25.4mm) "Normal" margins on all sides. The split between the two conventions is a prior.
- **SciTSR** ([paper](https://arxiv.org/abs/1908.04729)): 2,885 of 12,000 train and 716 of 3,000 test tables (about 24%) have at least one spanning cell. This sets `real_world_v3.content.merged_table_ratio` to `0.25`, a floor, since SciTSR tables come from scientific papers; `ko_admin_scan_v1` uses a `0.5` prior for HWP administrative tables.
- **Table style and page columns**: no source gives a census of table border/shading styles or two-column page share, so `typography.table_style` and `page.columns` are engineering priors: `web`/`grid`/`header_shaded` cover common office-suite/print looks, `booktabs` fits academic/technical writing (the LaTeX `booktabs` package default), and `page.columns.by_family` gives `academic` (conference/journal layout) and `technical` families a higher two-column share than the page-wide default.

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

Measured metrics: size, `est_dpi_a4` (width / 8.27 in, which assumes full A4 portrait pages), skew (projection profile), Laplacian variance, Immerkaer noise sigma, background/ink luminance and contrast, ink ratio, Hasler–Süsstrunk colourfulness, grayscale / binary / coloured-background flags, and the layout metrics below. Blur, noise and layout are measured at a fixed 1000 px width, so they are comparable across resolutions.

**Limitation:** the skew estimator is reliable for scans (within about 0.1° of the true value). On photographed pages, perspective and desk borders dominate and the estimate is not meaningful.

## Layout and Text Statistics

A pixel-texture match (blur, noise, contrast) can still hide a page whose *structure* and *content* are unrealistic — a synthetic page can look "close" on the 12 metrics above while having lorem-ipsum text and web-CSS-wide margins. Two more measurement layers close that gap.

### Layout metrics (`src/realism/layout_stats.py`)

`distribution measure` / `distribution compare` automatically include these keys on every row, merged with the pixel-texture metrics above (no extra flag needed):

| Metric | Meaning |
|---|---|
| `text_line_count` | Number of detected text-line bands (row-projection profile over an Otsu ink/no-ink binarisation) |
| `text_line_height_frac` | Median text-line height as a fraction of page **width** |
| `text_line_height_pt` | Median text-line height in points, assuming the page width is A4 (8.27 in). This is the *ink* height of a line band, not the font's em size — it runs smaller than the configured body font size and is not offered as a `body_font_pt` suggestion (see below) |
| `margin_top_frac` / `margin_bottom_frac` / `margin_left_frac` / `margin_right_frac` | Ink-bounding-box margins as a fraction of page height / width |
| `column_count` | 1 plus the number of persistent column gutters. Evaluated per text-line band: an "internal gap" is a white run bounded by ink on both sides of that band (so empty space after a short line is never one) and wider than ~2.5% of page width (well above an inter-word gap). A gutter is an x-range covered by such a gap in ≥60% of bands, centred in the middle 25-75% of the text area — so a full-width running title or centred page number crossing the gutter, or a wide-but-left-hugging bullet/list indent, do not defeat or fake detection |
| `text_area_frac` | Ink-bounding-box area as a fraction of total page area |
| `rule_count` | Long horizontal rules (table borders, `<hr>`s) found by morphological opening with a wide horizontal kernel, thin enough not to be a text line |

**Limitations:** like the skew estimator, these are unreliable on photographed pages — perspective distortion and desk background break the row/column projection profiles. A table with solid vertical borders (a ruled column separator running the table's full height) can put ink in every row of the table, merging the whole table into a single `text_line_count` band and inflating its measured height. A page whose bands are *dominated* by a well-aligned multi-column table (most lines are table rows, e.g. several stacked tables with little prose) can likewise read as multiple `column_count` — the table's bounded, consistently positioned cell gaps look exactly like a document column gutter from this metric's point of view; this is a known limitation, not corrected for.

When layout stats are present, `distribution measure --suggest-yaml` also proposes `margins_mm` (paste under `page.margins_mm.{top,bottom,left,right}`, millimetres on an assumed A4 sheet), in addition to the existing `dpi` / `skew_deg` / `grayscale` / `binarize` / `colored_background` specs. There is no `body_font_pt` suggestion: `text_line_height_pt` is ink-band height, not the font's em size (Hangul ink is roughly 0.8-0.9 em, Latin depends on ascenders/descenders, headings and merged bands shift it further), so it would be a systematically biased proposal. Use `text_line_height_pt` in `distribution compare` to check line height directly instead of deriving a font-size suggestion from it.

### Text metrics (`src/realism/text_stats.py`, `distribution text-stats`)

Character-class shares and symbol usage from plain text, so lorem-ipsum or Latin-only content shows up as a gap even when the image looks right:

```bash
# Real reference text, one file per page
uv run main.py distribution text-stats --text-dir /path/to/real_txt --output stats/real_text.json

# Or a JSONL file with one text field per line
uv run main.py distribution text-stats --texts real.jsonl --field text --output stats/real_text.json

# Synthetic: reads GT_markdown and strips markdown syntax (#, *, |, list markers, links, HTML tags) first
uv run main.py distribution text-stats --metadata ./pilot/ko/images_markdown/metadata.jsonl --output stats/pilot_text.json

# Same `compare` command works on text-stat summaries
uv run main.py distribution compare --reference stats/real_text.json --candidate stats/pilot_text.json
```

Per-page metrics: `chars_per_page`, `line_count`, `mean_line_length`, and a `_share` (of `len(text)`, including whitespace) for each of `hangul`, `hanja`, `kana`, `latin`, `digit`, `punctuation`, `symbol` (reference/list marks such as `□○※①`), `whitespace`. Shares need not sum to 1.0: characters outside these scripts (Cyrillic, Arabic, control characters, ...) are left uncounted. The summary also carries a `top_symbols` list: the most frequent punctuation/symbol characters across the whole corpus, useful for spotting missing 개조식 markers (`□`, `○`, `※`) in synthetic Korean text.

**Limitation:** `mean_line_length` is not comparable across sources with different line conventions. `GT_markdown` paragraphs are one long logical line each (breaks only at explicit markdown boundaries), while OCR transcripts and most real reference `.txt` files keep the document's visual line breaks. `distribution compare` will show a large gap on this metric that reflects the convention mismatch, not real content — reformat one side to match, or ignore `mean_line_length` in the comparison.

## Real-vs-Synthetic Discriminator

`distribution compare` reports per-metric gaps one statistic at a time. A discriminator instead asks the sharper question: taking *all* the measured statistics together, can a classifier tell a real page from a synthetic one? This is the standard "classifier two-sample test" (C2ST) trick, implemented with a small numpy-only L2-regularised logistic regression in `src/realism/discriminator.py` (no scikit-learn/torch, per this project's dependency-group rules) — no new dependency was needed since numpy is already a core dependency.

1. **Measure both sides with `--save-rows`**, so the per-image rows (not just aggregated quantiles) are kept, each tagged with its source image path:

   ```bash
   uv run main.py distribution measure --images /path/to/real --save-rows --output stats/real.json
   uv run main.py distribution measure --metadata ./pilot/ko/images_markdown/metadata.jsonl --save-rows --output stats/pilot.json
   ```

2. **Run the discriminator:**

   ```bash
   uv run main.py distribution discriminate --reference stats/real.json --candidate stats/pilot.json \
     --top 20 --output stats/discriminate.md
   ```

Class 0 is the reference (real) set, class 1 the candidate (synthetic) set. Classes are balanced by subsampling the larger side down to the smaller side's count (seeded, so the report is reproducible for the same inputs and `--seed`), then evaluated with stratified k-fold cross-validation (`--folds`, default 5, clamped down to the smaller class size when the sets are tiny).

**How to read the report:**

- **Mean AUC (with fold spread).** AUC ≈ 0.5 means the classifier cannot beat a coin flip — the two sets are statistically indistinguishable on these statistics, which is the goal. AUC → 1.0 means the sets are easy to tell apart; a wide spread across folds (large `auc_std` relative to `auc_mean`) means the estimate itself is noisy, usually because the measured sets are small.
- **Per-feature coefficients**, ranked by absolute value, from one final model fit on the whole balanced set: which statistics the classifier leans on most, i.e. the biggest gaps to close next in the profile.
- **Top-N candidate images** (`--top`, default 20): the candidate (synthetic) rows the model is most confident about, by predicted probability, with their image paths and a `score_source`. Every probability is a genuine held-out prediction, never a score from a model fit on that same row: a candidate row drawn into the balanced training subsample is scored with its cross-validation **out-of-fold** probability (`cv_out_of_fold`, the prediction from the one fold where it was held out), and a candidate row never drawn into the subsample is scored by the final model, for which it is a true holdout row (`final_model_holdout`). Without this split, ranking by the final model's in-sample score for training rows would systematically bias the top-N toward whichever rows happened to be drawn into the subsample — worse the larger the candidate set is relative to the reference set. Open these images directly instead of only reading aggregate numbers.

**Caveat:** like the skew and layout estimators it consumes, the discriminator still runs on photographed pages, but a high AUC driven mostly by `margin_*_frac`, `column_count` or `text_line_*` on a photographed-heavy set may reflect that those specific metrics are unreliable on photographs (perspective distortion, desk background — see [Layout metrics](#layout-metrics-srcrealismlayout_statspy) above) rather than a genuine distribution gap. Check which features dominate the coefficients table before concluding the generator itself needs tuning.

If `--reference` or `--candidate` was measured without `--save-rows`, `distribution discriminate` fails with a message pointing you back to `--save-rows`.

## Fit a capture channel from print–scan pairs

Keep the exact clean render, print it at a known physical width, then scan or
photograph the same page. Collect several pages per device/settings combination,
including text and tables. Keep the full page visible and use a flat sheet;
curled pages, shadows, stamps and printer halftones are not modeled by this fitter.

Create a CSV (paths are relative to the CSV):

```csv
clean,captured
clean/page-01.png,scans/page-01.jpg
clean/page-02.png,scans/page-02.jpg
```

```bash
uv run main.py distribution fit-capture --pairs pairs.csv --output stats/capture-fit \
  --channel scanned --clean-dpi 150
# Or match exact filenames in two directories:
uv run main.py distribution fit-capture --clean-dir clean --captured-dir scans \
  --output stats/capture-fit --channel photographed
```

Inspect `pair-0001.json`, etc. for fitted parameters, clean-to-capture homography,
correlation, RANSAC inliers, whether ECC refinement converged, paper RGB and
estimated black-ink level. Failed pairs have an `error` instead; the command
returns nonzero if any pair fails. `profile.yaml` contains empirical `choices`
specs from successful pairs under `capture_channels.<channel>`, with `dpi` beside
`degradations`. Merge the reviewed snippet into your own profile, retaining its
channel weight. Existing scenarios override channel defaults, so update the
intended scenario instead when appropriate. No shipped profile is modified.

The estimates use the existing degradation keys:

| Estimate | Interpretation |
|---|---|
| `dpi` | Similarity scale × clean DPI; default clean DPI assumes a 210 mm A4 width. Supply `--clean-dpi` for other sizes. Perspective foreshortening also affects scale. |
| `skew_deg` | Rotation in the generator/OpenCV sign convention. |
| `perspective` | Moment-matched magnitude for the generator's inward corner jitter; approximate across a collection, not a unique inverse for one page. |
| `ink_fade`, `paper_tint` | Fitted ink offset relative to paper, and a warm/dark-paper flag; tint is a boolean, not an exact recovered colour. |
| `blur_sigma` | Gaussian sigma in captured pixels, grid searched from 0 to 3 in steps of 0.1 with fitted intensity gain/offset. |
| `noise_sigma` | Flat-paper high-pass residual matched to seeded Gaussian noise, including clipping and JPEG suppression (0–25 in steps of 0.25). |
| `jpeg_quality` | Nearest standard IJG/Pillow quality from file quantization tables; custom tables are approximate, lossless files yield null. |
| `grayscale`, `binarize` | Pixel-based flags, including tolerance for JPEG ringing on bitonal pages. |

Thresholding makes blur/noise/fade unidentifiable, so these are null for bitonal
captures and omitted from aggregation. Other null estimates are also omitted.
The snippet describes independent marginal distributions; it does not recover
correlations or scenario weights. Group captures by device/settings first.
Blur and noise at a grid boundary deserve review. Registration error, resizing,
lighting gradients and scanner denoising can bias optical estimates. JPEG noise
recovery assumes the fitted Gaussian model; it cannot undo arbitrary processing.

Generate a pilot with the reviewed profile and compare it with held-out captures
using `distribution measure` / `compare`. Repeat the print → capture → fit loop
when changing printer, paper, scanner settings or camera conditions. The seed
controls RANSAC and noise calibration; identical inputs and seed are repeatable
within the same OpenCV environment. Generation without a profile is unchanged.

## Calibration Record: `real_world_v2`

`real_world_v2` was fitted with the loop above against **118 real pages** that are reachable from GitHub:

- XFUND zh+ja validation: 100 scanned forms ([repo](https://github.com/doc-analysis/XFUND))
- the 18 OmniDocBench demo pages ([repo](https://github.com/opendatalab/OmniDocBench/tree/main/demo_data)).

Channels are compared to their matching source: synthetic `scanned` pages against XFUND, and `born_digital` pages against OmniDocBench (`--where capture_channel=...`).

What the calibration changed relative to v1:

| Finding (v1 pilot vs real) | Change in v2 |
|---|---|
| Page aspect 1.07 vs 1.41: pages were only as tall as their content | `page.aspect_ratio` (A4/Letter), padding, and fit-to-sheet trimming |
| Ink coverage 2.5% vs 5–11%, contrast 57 vs 151–175 | Denser `content`, `spacing_scale` 0.45–0.8, near-black `ink_gray` |
| Scan skew σ≈0.3° (XFUND) vs prior 0.5° | `scanned.skew_deg ~ N(-0.1, 0.3)` |
| XFUND scans: 100% grayscale, 3% bitonal, white-balanced paper | `grayscale` 0.9, `binarize` 0.05, `paper_tint` 0.12 |
| Born-digital median ~194 dpi, JPEG sources, 11% coloured background | DPI mix, JPEG quality mix, `colored_background` 0.11 |

Results are in [`calibration/real_world_v2.md`](calibration/real_world_v2.md). The reference set is small and forms-heavy, so for your own target domain, re-run `measure` on your data (see above).

## Real-Language Corpus

`uv run main.py corpus import-wikitext` converts WikiText dumps into `data/corpus/<lang>/paragraphs.txt` and `titles.txt`. These are picked up automatically, and titles, list items and table cells are derived from them.

- `--kowikitext-split dev|test|train` downloads [Korean WikiText](https://github.com/lovit/kowikitext) from GitHub releases:
  - dev + test: about 18k clean paragraphs, enough for most runs
  - train: about 1.7 GB
- `--input FILE` imports any local WikiText-format file.
- The cleaner strips markup debris such as empty `(, )` pairs, wiki list markers (`# item`, `: quote`), namespace titles (`분류:…`), missing spaces after sentence ends, and whole paragraphs containing table markup, URLs or talk-page signatures. Leading markdown markers are also neutralised when paragraphs are built, so corpus text can never turn into a heading.
- **License:** Wikipedia text is CC BY-SA 3.0. Publish with `--license cc-by-sa-3.0` and credit the source with `--text-source` (written into the card's attribution section).

## Genre-Specific Corpus

A single Wikipedia-derived `paragraphs.txt` makes every template family read like encyclopedia prose -- a `policy_document` page and a `meeting_minutes` page end up with the same generic sentences. `uv run main.py corpus generate --category <name>` (see [`corpus generate`](cli.md#corpus-generate)) also writes these genre-specific categories to `data/corpus/<lang>/<name>.txt`:

| Category | Style | Matched `document_shape` |
|---|---|---|
| `report_lines` | One 개조식 line per item, ending in `~함`/`~임`/`~됨` | `release_note` |
| `meeting_notes` | 2-3 sentence discussion/decision entries | `meeting_minutes` |
| `notice_paragraphs` | 2-3 sentence official notice/announcement paragraphs | `policy_document` |
| `contract_clauses` | 2-3 sentence contract/agreement clause paragraphs | `form_like` |
| `academic_abstracts` | 2-3 sentence paper-abstract paragraphs | `academic_note` |
| `financial_commentary` | 2-3 sentence financial performance commentary | `business_report` |

`DocumentComposer` (`src/generator/document_blocks.py`, `GENRE_BY_DOCUMENT_SHAPE`) derives the genre from the selected template's `document_shape` and feeds it to `DataProvider.paragraph(genre=...)` (`src/generator/data_provider.py`), which reads `data/corpus/<lang>/<genre>.txt` and falls back to the general `paragraphs` corpus when that file is absent -- so generating only some categories, or none, is always safe. Only `document_shape`s whose blueprint allows `paragraph` blocks at all are mapped (`table_heavy` and `formula_heavy` never emit one); other shapes are unaffected and keep reading the general corpus. Whether the genre corpus is actually used (when both it exists and the shape has a mapping) is controlled by `content.genre_corpus`, off by default -- see the table above.

## Beyond Visual Realism

Profiles fix *how pages look* and *which structures appear*; the words come from the corpus (see above, or `corpus generate` for LLM-written domain text). To confirm the benchmark is realistic, check that model rankings on the synthetic set correlate (Spearman) with rankings on a real benchmark.
