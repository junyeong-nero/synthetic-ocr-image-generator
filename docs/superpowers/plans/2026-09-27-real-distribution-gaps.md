# Real-Distribution Gaps: Implementation Plan

Closes the gaps found in the 2026-09-27 analysis of how far synthetic pages are
from real OCR pages (sections 2-6 of that analysis). PR #2 already fixed the
profile-intent bugs (family mix, template weights, corpus requirement,
look-alike typos, table rule colour).

Each task below ships as **its own branch and pull request**, implemented by one
subagent and reviewed and merged by the controller session.

## Global Constraints (bind every task)

1. **Branch and PR.** Branch from the latest `origin/main`, named as the task
   says (`git branch -m <name>` if the worktree created another name). Commit
   subjects use the repo prefixes `[feat]`, `[fix]`, `[add]`, `[docs]`,
   `[test]`, `[chore]`. Push and open a PR against `main` with `gh pr create`
   (English body: Summary, Changes, Testing, Notes). **Never merge.** The
   controller reviews; you fix findings on the same branch. When asked to
   rebase: `git fetch origin && git rebase origin/main`, resolve, re-run tests,
   `git push --force-with-lease`.
2. **Behaviour without a profile is unchanged.** Every new behaviour is gated
   by a distribution-profile key whose code default is "off" / legacy. A run
   without `--distribution-profile`, or with a profile that does not set the
   new key, must follow the old code path. Existing tests must keep passing.
3. **Where knobs are switched on.** Enable new keys in
   `configs/generator/distributions/real_world_v3.yaml` (and in
   `ko_admin_scan_v1.yaml` when the feature fits scanned Korean admin
   documents). Mark each value `[sourced]` (with the source) or `[prior]` in a
   comment. **Never edit `real_world_v1.yaml` or `real_world_v2.yaml`.**
4. **Docs.** Document every new profile key in `docs/distribution.md` ("What a
   Profile Controls" table or "Degradations" table) and every new CLI flag or
   subcommand in `docs/cli.md`. Keep README changes to one line if any.
5. **GT always matches the image.** `GT_markdown` must describe exactly what
   is rendered (except page furniture, see Task 12). Anything that can cut or
   hide content (cropping, trimming) must keep GT consistent.
6. **Determinism.** All sampling uses the per-sample seeded RNG (`random`
   seeded in `Generator._seed_for_sample`, or the `rng` passed in
   `profile_application` / `degradation`). Same seed → same image and GT.
7. **TDD.** Write the failing test first, watch it fail for the right reason,
   then implement. Tests live under `tests/` and import with the `src.` prefix
   (see `tests/generator/test_profile_generation_controls.py` for a generator
   fixture that injects a temp corpus and font dir). Unit tests must not depend
   on real font files or network.
8. **Test command.** Full `uv sync` installs torch, so use the lightweight env:

   ```bash
   PYTHONPATH=. uv run --no-project --python 3.11 --with pytest --with faker \
     --with pillow --with opencv-python-headless --with numpy --with markdown \
     --with "playwright==1.55.0" --with pyyaml --with tqdm --with mistune \
     --with scikit-image --with datasets --with huggingface-hub --with pydantic \
     pytest -q --ignore=tests/test-gemini.py tests
   ```

   Add `--with <pkg>` for any package your task adds. `main` currently has
   exactly **12 known failing tests** (stale tests calling removed APIs). Do not
   fix them and do not add new failures:
   - `tests/generation/test_phase2_generation_modules.py::test_build_dataset_readme_contains_expected_sections`
   - `tests/generation/test_phase2_generation_modules.py::test_publish_pipeline_uses_manifest_context`
   - `tests/generator/test_streaming_generation.py::test_markdown_dataset_generator_run_streams_top_level_metadata_and_stats`
   - 9 tests in `tests/test_wave2_cli_pipeline_helpers.py`
9. **Visual check.** For any task that changes rendering or degradation,
   render a few pages with Playwright (chromium is installed; the
   `playwright==1.55.0` wheel above matches it) and look at them. Font files
   (`fonts/**/*.ttf`) are git-ignored: if your worktree has none, point
   `font_dir` at `/Users/junyeong-nero/workspace/synthetic-ocr-image-generator/fonts/ko`.
   Use a tiny Korean corpus in a temp dir injected through
   `DataProvider(lang="ko", corpus_dir=...)`. Save the images under your
   report directory and list them in the report.
10. **Code organisation.** Follow `AGENTS.md`: keep files focused; put a new
    responsibility in a new module rather than growing `generator.py`,
    `document_blocks.py`, `markdown_renderers.py` or `degradation.py`.
    Match the surrounding style (comment density, naming).
11. **Dependencies.** No new dependency groups. A new package may be added to
    an existing group (and to `[project].dependencies` if generation needs it)
    only when the task says so.
12. **No subagents.** Implementers do all work themselves.

## Rulings Made Before Execution

- **Page furniture GT (Task 12):** headers, footers and page numbers are drawn
  but excluded from `GT_markdown` and recorded in metadata `page_furniture`;
  evaluation drops predicted lines that match them (OmniDocBench convention).
- **Merged cells (Task 10):** tables with row/col spans are written into
  `GT_markdown` as an HTML `<table>` block; the metrics already parse HTML
  tables (`src/metrics/markdown_block_metrics.py`).
- **Fonts:** no font files are added to the repo (licensing, git-ignored).
  Profiles weight fonts by file-name pattern so user-installed fonts such as
  함초롬바탕 or 맑은 고딕 are picked up when present.
- **Discriminator (Task 8):** numpy logistic regression on per-image
  statistics, no scikit-learn/torch.
- **Out of scope:** acquiring a Korean real reference set (licence/login bound,
  the user's job); rendering HWP/HWPX templates (no headless HWP renderer);
  collecting print-scan pairs (physical). Tooling for the last one is Task 14.

## Waves

Tasks in one wave touch disjoint files and may run in parallel. A later wave
starts after the tasks it depends on are merged.

| Wave | Tasks |
|---|---|
| 1 | 1 tables · 2 capture artifacts · 3 layout/text stats · 4 rank correlation · 5 page typography |
| 2 | 6 Korean conventions · 7 capture scenarios (after 2) · 8 discriminator (after 3) · 9 table styles + columns (after 5) · 10 merged cells (after 1) |
| 3 | 11 genre corpus (after 6) · 12 page furniture + continuation (after 9) · 14 capture fitting (after 8) |
| 4 | 13 figures · 15 formulas (after 12) |
| 5 | 16 forms (after 11, 13) · 17 layout re-rendering (after 12) |
| 6 | 18 LLM HTML pages (after 17) |

---

### Task 1: Realistic table content

**Branch:** `feat/realistic-tables` · **Model tier:** standard

**Problem.** `src/generator/table_generator.py` always emits column 0 = product
name, 1 = quantity, 2 = price, rest = feature text, under headers drawn from
four unrelated hard-coded sets (e.g. "시간 | 월 | 화 | 수" above "김밥 | 8 |
24,000원"). Numbers are unrelated to each other, nothing is right-aligned.

**Build.**
- New module `src/generator/table_schemas.py`: coherent table schemas whose
  columns have semantic types (label, date, time, integer, amount, percent,
  change, person, department, position, phone, status, free text) and
  localized header labels for `ko`, `en`, `ja` (other languages use `en`).
- Schemas (at least): `financial` (항목 | 당기 | 전기 | 증감 | 증감률, amounts
  in 백만원 with the unit in the header, e.g. "당기(백만원)"), `budget`
  (구분 | 예산액 | 집행액 | 집행률), `schedule` (일자 | 시간 | 내용 | 장소 |
  담당), `roster` (성명 | 소속 | 직위 | 연락처), `order` (품목 | 규격 | 수량 |
  단가 | 금액, with a final 합계 row), `statistics` (구분 | year columns).
- Values must be internally consistent: 금액 = 수량 × 단가, 합계 = column sum,
  증감 = 당기 − 전기, 증감률 = 증감 / 전기, 집행률 = 집행액 / 예산액. Thousand
  separators; negatives as `△1,234` (ko financial), `(1,234)` or `-1,234`
  depending on schema/locale; percentages with one decimal.
- Numeric columns use markdown right alignment (`---:`) in the separator row.
- Respect the blueprint's `table.rows` / `table.columns` ranges where the
  schema allows (drop optional columns to fit; pick schemas compatible with
  the column range).
- Profile key `content.table_schemas`: mapping schema → weight. When the key is
  absent (or there is no profile) `TableGenerator` behaves exactly as today.
  Plumb it from `DocumentComposer` content specs into `DocumentBlockBuilder` /
  `TableGenerator`.
- Row labels and free-text cells come from the `DataProvider` (corpus-derived
  where possible, curated lists otherwise); add curated `ko`/`en`/`ja` label
  lists (account items, budget categories, place names...) in the new module,
  not in `language_data.py`.
- Enable in `real_world_v3.yaml` (financial-heavy mix) and
  `ko_admin_scan_v1.yaml` (budget/schedule/roster-heavy mix).

**Tests.** Consistency rules above on many seeds; numeric columns right-aligned
and text columns not; localized headers for ko and en; column range honoured;
legacy output unchanged without the key; determinism.

---

### Task 2: Capture artifacts

**Branch:** `feat/capture-artifacts` · **Model tier:** standard

**Problem.** `src/generator/degradation.py` lacks common real-scan/photo
artifacts, and its bleed-through mirrors the same page (text ghosts line up
with the page's own text).

**Build** in a new module `src/generator/capture_artifacts.py`, dispatched from
`apply_capture_degradation` (keep `degradation.py` a thin pipeline). New
optional degradation keys, all default off:
- `stamp` (bool): red seal (circle or rounded square, double ring, short
  Korean text such as "직인", "(인)", an organisation-like name), slightly
  rotated, uneven ink, multiply-blended near the lower right or over the last
  text lines. Applied before grayscale/binarize.
- `highlighter` (bool): translucent yellow bar over 1-3 text lines found by a
  horizontal projection profile.
- `fold_lines` (int 0-2): crease across the page (dark line with a light
  edge, slight brightness step on one side).
- `punch_holes` (bool): 2 or 3 dark round holes in the left margin only
  (never over ink).
- `scanner_border` (0..1): dark shadow band along one or two page edges.
- `toner_streaks` (int): faint vertical streaks.
- `page_curl` (0..~0.05): photographed pages, smooth vertical warp via
  `cv2.remap` (book curvature).
- `edge_crop` (0..~0.1): photographed pages, crop part of the page border
  **only outside the ink bounding box** (GT must stay complete).
- `binarize_method` (`otsu` default | `adaptive` | `sauvola`): used when
  `binarize` is true.
- `bleed_through`: keep key and strength meaning, but build the reverse side
  from horizontally mirrored content cut into horizontal bands that are
  shuffled and shifted, with stronger blur, so ghosts do not align with the
  page's own lines.
Document the application order in the module docstring. Update
`CaptureSample.difficulty()` only if a new effect clearly makes pages harder
(e.g. `page_curl`, `edge_crop`).

Enable with `[prior]` values in `real_world_v3.yaml` (e.g. stamps and punch
holes on a small share of scanned pages, curl on photographed) and
`ko_admin_scan_v1.yaml` (stamps more common). Update the Degradations table
in `docs/distribution.md`.

**Tests.** Each effect changes pixels only where it should (punch holes in the
left margin, highlighter over a text line, stamp red-dominant before
grayscale); `edge_crop` never removes ink; new bleed-through ghosts are not
row-aligned with the page's lines; determinism; empty params stay identity.

---

### Task 3: Layout and text statistics

**Branch:** `feat/layout-text-stats` · **Model tier:** standard

**Problem.** Calibration (`src/realism/`, `main.py distribution measure|compare`)
only measures 12 global pixel statistics, so it cannot see layout or text
differences (a lorem-ipsum page can look "close").

**Build.**
- `src/realism/layout_stats.py`: image-only layout metrics computed the same
  way for real and synthetic pages at the existing analysis width: text line
  count, median text line height (as a fraction of page width and as pt
  assuming A4 width), ink-bbox margins (top/bottom/left/right fractions),
  estimated column count (persistent vertical white gap across the text
  area), text-area fraction, count of long horizontal rules (morphology).
  Merge these keys into the per-image rows so `measure` / `compare` pick them
  up. State in the docstring that they are unreliable on photographed pages.
- `src/realism/text_stats.py`: per-page text statistics from strings:
  character-class shares (hangul, hanja, kana, latin, digit, punctuation,
  symbol such as □○※①, whitespace), characters per page, mean line length,
  top-k non-alphanumeric symbols. For synthetic data strip markdown syntax
  from `GT_markdown` first.
- CLI: `distribution text-stats` reading `--metadata metadata.jsonl`
  (`GT_markdown`), `--texts file.jsonl --field text`, or `--text-dir *.txt`,
  writing a summary JSON with the same structure `distribution compare`
  consumes, so text summaries can be compared with the existing command.
- Optionally extend `suggest_profile_specs` with `page.margins_mm` and
  `typography.body_font_pt` histograms (keys introduced by Task 5; emit them
  as suggestions only).
- Docs: extend the calibration section of `docs/distribution.md`.

**Tests.** Synthetic drawn pages with known line count, line height, margins
and a two-column page; text stats on known strings; `compare` works on
text-stat summaries.

---

### Task 4: Synthetic-vs-real rank correlation

**Branch:** `feat/rank-correlation` · **Model tier:** standard

**Problem.** The docs say to validate the benchmark by correlating model
rankings on synthetic data with rankings on a real benchmark, but no tool does
it.

**Build.**
- `src/evaluation/rank_correlation.py`: load synthetic scores from this repo's
  evaluation outputs (`evaluation_result/**/report.json` and/or
  `leaderboard.json`; read the existing formats) for a chosen metric
  (default `avg_markdown_overall_score`), and real scores from a CSV or JSON
  mapping model → score. Match models by normalized id plus an optional alias
  file. Compute Spearman rho and Kendall tau (scipy is available through
  scikit-image), a bootstrap 95% CI over models, n, and list unmatched models.
- CLI `main.py rank-correlation --synthetic ... --real ... [--metric]
  [--aliases] [--output report.md]` in a new `src/cli/rank_correlation.py`,
  registered in `src/cli/app.py`. Markdown report with a per-model rank table.
- Docs in `docs/evaluation.md` and `docs/cli.md`.

**Tests.** Known rankings → known rho/tau; ties; alias matching; unmatched
models reported; report formatting.

---

### Task 5: Page typography

**Branch:** `feat/page-typography` · **Model tier:** standard

**Problem.** Margins are ~4-7% of the page (real A4 documents: ~8-14%), text is
always ragged-right, and one font file styles the whole page (headings, body,
tables and code alike; code falls back to the same font). After the profile
font filter, 20 of 25 Korean fonts are Nanum and 8 are NanumSquare display
faces.

**Build.**
- `page.margins_mm`: `{top, bottom, left, right}`, each a distribution spec in
  millimetres on A4 (210 mm wide) → CSS padding (`mm / 210 × page width`).
- `typography.text_align`: `left` | `justify` (CJK-friendly justification).
- `typography.word_break`: `normal` | `keep-all`.
- `fonts.body`, `fonts.heading`: weighted groups of file-name substrings (e.g.
  `body: {Myeongjo: 3, batang: 2, Gothic: 2}`); pick a group by weight, then a
  file uniformly inside it; `fonts.exclude` still applies; fall back to the
  current choice when nothing matches. `fonts.code`: substrings for a
  monospace face (e.g. `D2Coding`).
- Renderer CSS gets separate `@font-face` rules for heading (h1-h3, th) and
  code (code, pre) fonts. Metadata adds `heading_font_name`, `code_font_name`.
- PIL renderer may ignore the new keys (document it).
- Enable in `real_world_v3.yaml` and `ko_admin_scan_v1.yaml` with `[prior]`
  values (Korean HWP default margins: top 20 mm + header 15 mm, left/right
  30 mm; Word: 25.4 mm).

**Tests.** Margin px derived from mm; justify/keep-all CSS; heading/code
font-faces emitted; group weights honoured statistically; legacy CSS and font
choice unchanged without the keys.

---

### Task 6: Korean document conventions

**Branch:** `feat/korean-doc-conventions` · **Model tier:** standard

**Problem.** Every page is web-style markdown (`# title`, `## section`,
`- item`). Korean documents number headings (1., 가., Ⅰ.), use 개조식 markers
(□, ○, -, ※) and law-style articles (제1조(목적), ①).

**Build** in a new module `src/generator/document_conventions.py`, applied by
`DocumentComposer` (`src/generator/document_blocks.py`):
- `content.heading_numbering`: `none` | `arabic` (1., 1.1) | `korean_admin`
  (Ⅰ./1./가.) | `roman`.
- `content.list_style`: `markdown` | `korean_admin` (bullet lists as □/○/·
  lines, numbered lists as 1)/가)/① lines, optional ※ note line).
- `content.law_articles` (probability) for `policy_document` shapes: numbered
  lists become `제N조(제목) ...` paragraphs with ①② clauses.
- `korean_admin` and law styles apply only when `lang` is `ko`; other languages
  fall back to `arabic` / `markdown`.
- Marker lines must not be re-interpreted as markdown lists or headings in a
  way that makes the rendered image, `GT_markdown` and `GT_json` disagree
  (python-markdown renders the image, mistune builds `GT_json`); prefer
  markers that are plain text in both, and test it.
- Metadata: record the chosen styles.
- Make sure `text_mutation` still behaves (the profile sets
  `similar_char_ratio: 0`, but the CLI can override it).

**Tests.** Numbering formats; ko-only gating; image/GT agreement on marker
lines (parse with both markdown libraries); legacy unchanged.

---

### Task 7: Capture scenarios (correlated degradations)

**Branch:** `feat/capture-scenarios` · **Model tier:** standard · **After:** Task 2

**Problem.** Degradation parameters are sampled independently, but in reality
they move together (old archive scans: tint + fade + bleed + low dpi; fax:
bitonal + low dpi + speckle).

**Build.**
- `capture_channels.<ch>.scenarios`: named sub-scenarios with `weight`,
  optional `dpi`, and `degradations` that override the channel-level
  degradations. Sampling picks a scenario by weight, then samples its merged
  specs. Channels without `scenarios` behave exactly as today.
- `CaptureSample.scenario`; metadata `capture_scenario`.
- Rewrite `real_world_v3.yaml` scanned/photographed channels into scenarios
  (e.g. scanned: office_adf, archive, photocopy, fax; photographed:
  phone_flat, phone_book, low_light) using Task 2's artifacts, keeping channel
  weights and the calibrated v2 ranges as the base. Same for
  `ko_admin_scan_v1.yaml`.
- Docs.

**Tests.** Scenario weights honoured; override merge; metadata; legacy profiles
unchanged.

---

### Task 8: Real-vs-synthetic discriminator

**Branch:** `feat/realism-discriminator` · **Model tier:** standard · **After:** Task 3

**Build.**
- `src/realism/discriminator.py`: classifier two-sample test on per-image
  statistic rows (image + layout stats). Standardize features, numpy logistic
  regression with L2, stratified k-fold, balanced classes by subsampling.
  Report AUC and accuracy with fold spread, per-feature coefficients (which
  statistics separate the sets), and the top-N candidate images the model is
  most confident are synthetic (paths + probabilities).
- If `distribution measure` output does not keep per-image rows and paths, add
  that (e.g. `--save-rows`).
- CLI `distribution discriminate --reference real.json --candidate syn.json
  [--top 20] [--output report.md]`.
- Docs: how to read AUC (≈0.5 indistinguishable) and use the top-N list.

**Tests.** Separable synthetic feature sets → AUC near 1 and the separating
feature ranked first; identical distributions → AUC near 0.5; determinism.

---

### Task 9: Table styles and multi-column pages

**Branch:** `feat/table-styles-columns` · **Model tier:** standard · **After:** Task 5

**Build.**
- `typography.table_style`: `web` (legacy look) | `grid` (solid ink borders,
  no zebra, bold centred header) | `header_shaded` | `booktabs` (top/bottom
  thick rules and a header rule, no vertical lines) | `borderless` (header
  underline only). Printed-look padding. Numeric alignment comes from the
  markdown separator row (Task 1).
- `page.columns`: 1 | 2 (CSS columns, the `h1` title spanning all columns,
  tables and figures not split across columns, gap in mm). `GT_markdown`
  order is unchanged and equals the column-major reading order; verify the
  fit-to-sheet trimming still works with columns.
- Enable in `real_world_v3.yaml` (academic/technical pages more likely 2
  columns if you can condition on family; otherwise a small global share).

**Tests.** CSS per style; columns CSS; legacy unchanged; a rendered two-column
page reads in GT order (visual check).

---

### Task 10: Merged-cell tables

**Branch:** `feat/merged-cell-tables` · **Model tier:** most capable · **After:** Task 1

**Build.**
- New `src/generator/html_table.py`: build tables with multi-level headers
  (colspan group headers such as "2024년" over "상반기 | 하반기") and row
  groups (rowspan category labels), from Task 1 schemas.
- `content.merged_table_ratio`: share of tables emitted as an HTML `<table>`
  block (one chunk, no blank lines) in `GT_markdown`; rendered as-is through
  python-markdown.
- Make sure `fit_markdown_to_sheet`, composition metadata, `text_mutation` and
  `GT_json` handle the HTML block.
- Check `evaluate_markdown_blocks` scores an identical HTML table 1.0 and
  penalizes flattened spans.

**Tests.** Well-formed HTML; span grid consistency; metric behaviour; not
mutated; legacy unchanged.

---

### Task 11: Genre-specific corpus

**Branch:** `feat/genre-corpus` · **Model tier:** standard · **After:** Task 6

**Problem.** One Wikipedia corpus fills every genre, so reports, minutes and
notices all read like encyclopedia prose.

**Build.**
- New LLM corpus categories in `src/corpus_llm/constants.py` with `ko`/`en`/`ja`
  prompts: `report_lines` (개조식 lines ending in ~함/~임), `meeting_notes`,
  `notice_paragraphs`, `contract_clauses`, `academic_abstracts`,
  `financial_commentary`. Reuse the existing `corpus generate` flow.
- `DataProvider`: genre-aware lookups (`paragraph(genre=...)` or similar) with
  fallback to general `paragraphs.txt`.
- `DocumentComposer` passes a genre derived from the template family/shape;
  profile key `content.genre_corpus` (probability of using a genre corpus when
  one exists).
- Docs: corpus section of `docs/distribution.md`.

**Tests.** Fixture corpus dirs: genre text used when present, fallback when
absent, legacy unchanged.

---

### Task 12: Page furniture and continuation pages

**Branch:** `feat/page-furniture` · **Model tier:** most capable · **After:** Task 9

**Build.**
- New `src/generator/page_furniture.py`: running header (short doc title,
  organisation name, date; optional rule under it), footer text, page number
  styles (`- 3 -`, `3`, `3 / 12`, `3쪽`, `Page 3`).
- Renderer: a page container of the sheet height (width × aspect) with the
  header at the top and footer/page number at the bottom; content between.
  Adapt `sheet_overflow_ratio` / fit-to-sheet trimming to the container.
- Profile keys: `page.header` (p), `page.footer` (p), `page.page_number`
  (choices incl. `none`).
- GT: furniture is **not** in `GT_markdown`; metadata `page_furniture`
  (`{header, footer, page_number}` strings or null).
- Evaluation: where predictions are scored against `GT_markdown`, remove
  predicted lines that match a furniture string (normalized, fuzzy) before
  `evaluate_markdown_blocks`, only for samples that carry `page_furniture`.
- New `src/generator/page_composition.py`: `content.continuation_page` (p):
  drop the `#` title and a random number of leading sections so the page
  starts at a later section; `content.start_mid_paragraph` (p): the page
  starts with the tail of a paragraph (cut at a sentence boundary). Keep
  `merge_order` and composition metadata consistent.
- Enable in `real_world_v3.yaml` and `ko_admin_scan_v1.yaml`.

**Tests.** Furniture rendered but absent from GT; metadata; evaluation strips
matching predicted lines and nothing else; continuation pages consistent;
legacy unchanged.

---

### Task 13: Figures with real content

**Branch:** `feat/figure-assets` · **Model tier:** standard · **After:** Tasks 6, 12

**Problem.** Every figure is a grey box with the word "Figure"; no charts,
diagrams or photos.

**Build.**
- New `src/generator/figure_assets.py`: bar, line and pie charts (matplotlib,
  random but plausible data, localized labels drawn with the page's body font
  file) and simple box-and-arrow diagrams (PIL). Deterministic per seed.
- `content.figure_kinds`: mapping kind → weight (`placeholder` = legacy).
- Composer emits `![그림 N. caption](figure://<id>)` with a localized caption;
  the generator builds the image assets and passes them to
  `renderer.render(..., image_assets=...)` (including fit-to-sheet
  re-renders).
- Add `matplotlib` to the `generate` dependency group and
  `[project].dependencies` in `pyproject.toml`.
- Chart text is figure content and stays out of GT (only the image markdown
  with its caption is in GT).

**Tests.** Asset generation per kind; deterministic; assets wired into render
calls; legacy placeholder unchanged.

---

### Task 14: Capture parameter fitting from print-scan pairs

**Branch:** `feat/capture-fit` · **Model tier:** most capable · **After:** Task 8

**Build.**
- `src/realism/capture_fit.py`: given a clean synthetic render and a real
  capture of the printed page (scan or photo), align them (ORB + RANSAC
  homography, then ECC refinement) and estimate rotation/skew, scale
  (effective dpi), perspective magnitude, paper/ink levels (→ `ink_fade`,
  `paper_tint`), blur sigma (grid search on the aligned clean image), noise
  sigma (residual), JPEG quality (from the file's quantization tables when it
  is a JPEG), grayscale/bitonal flags.
- CLI `distribution fit-capture --pairs pairs.csv` (clean,captured) or
  `--clean-dir/--captured-dir` matched by file name; per-pair JSON and an
  aggregated YAML snippet of degradation specs for a profile channel.
- Docs: the print → scan/photograph → fit loop.

**Tests.** Degrade a clean page with known parameters and recover them within
tolerance; alignment on a rotated/perspective-warped copy.

---

### Task 15: Formula realism

**Branch:** `feat/formula-realism` · **Model tier:** most capable · **After:** Task 12

**Build.**
- `typography.formula_style`: `boxed` (legacy) | `plain` (centred display math
  without border/background).
- `content.inline_math_ratio`: share of paragraphs in academic/technical shapes
  that get 1-3 inline `$...$` spans (short expressions from the formula
  pools). The HTML renderer turns inline `$...$` into inline formula images
  sized and baseline-aligned to the body text. Only spans the generator
  inserted may be converted (no false matches on ordinary text).
- `text_mutation` must never alter math spans.
- Check the metric counts inline formulas as formula blocks as intended.

**Tests.** Plain vs boxed CSS; inline spans in GT and converted in HTML;
protected from mutation; legacy unchanged.

---

### Task 16: Realistic forms

**Branch:** `feat/realistic-forms` · **Model tier:** standard · **After:** Tasks 11, 13

**Problem.** `form_like` is a markdown table plus a checklist; real forms
(e.g. XFUND, the calibration reference) are label/value grids with checkbox
options, date and signature lines, and stamps.

**Build.**
- New `src/generator/form_blocks.py`: `kv_table` (2- or 4-column label/value
  grid, labels from localized field lists, typed values from the
  `DataProvider`), `checkbox_group` (`구분: ☑ 신규 ☐ 변경 ☐ 해지`),
  `signature_line` (date line plus `신청인: 홍길동 (서명 또는 인)`).
- New template(s) (e.g. `application_form`, family `forms`) using them. The
  template catalog gains a `profile_only: true` flag so legacy runs never
  select these templates; honour it in template resolution.
- Label-cell shading without polluting GT: a document-level class on the
  page (e.g. `.markdown-body.form td:nth-child(odd)`), not markup in GT.
- Enable in `real_world_v3.yaml` and `ko_admin_scan_v1.yaml` through
  `template_weights`.

**Tests.** Block formats; profile-only gating; GT/image agreement; legacy
catalog selection unchanged.

---

### Task 17: Layout-conditioned re-rendering

**Branch:** `feat/layout-rerender` · **Model tier:** most capable · **After:** Task 12

**Build.**
- `src/generator/layout_source.py`: load page layouts from DocLayNet-style
  COCO JSON (categories Caption, Footnote, Formula, List-item, Page-footer,
  Page-header, Picture, Section-header, Table, Text, Title) or a simple JSON
  list of `{bbox, category}` per page.
- `src/generator/layout_renderer.py`: HTML with absolutely positioned boxes
  scaled to the sheet; fill each box with corpus text sized to fit, tables
  from the table generator, figures from Task 13 assets or placeholders,
  formulas from the formula pools; page header/footer boxes become
  `page_furniture` (Task 12 convention).
- `GT_markdown` in reading order (column clustering, then top-to-bottom).
- Generation option `--layout-source PATH` (CLI, `GenerationOptions`,
  manifest); when set, each sample uses a layout instead of the composer;
  profile capture degradations still apply.

**Tests.** Fixture layouts: boxes filled, GT order, furniture excluded,
determinism; option round-trip.

---

### Task 18: LLM-generated HTML pages

**Branch:** `feat/llm-html-pages` · **Model tier:** most capable · **After:** Task 17

**Build.**
- `corpus generate-html`: prompt the existing LLM providers for
  self-contained HTML document pages per genre (공문, 보도자료, 계약서,
  재무제표, 영수증, 2단 논문...) using a restricted tag/CSS subset
  (headings, p, lists, tables with rowspan/colspan, header/footer and column
  containers). Sanitize and store under `data/html_pages/<lang>/<genre>/`.
- `src/generator/html_page_source.py`: load a stored page, inject fonts and
  profile typography, render with Playwright, derive `GT_markdown` from the
  DOM (headings, paragraphs, lists → markdown; tables → markdown or HTML when
  spans exist; header/footer → `page_furniture`).
- Generation option `--html-source DIR`, reusing the page-source integration
  point Task 17 introduced.
- API keys are needed to create pages; tests use fixture HTML.

**Tests.** Sanitizer; DOM → GT conversion incl. merged tables and furniture;
render path with a fixture page; option round-trip.
