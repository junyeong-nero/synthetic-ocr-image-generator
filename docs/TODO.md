# Real-Distribution Work: Status and TODO

Last updated: 2026-09-29.

Goal: make synthetic pages closer to real OCR data (content, layout, capture) and measure how close they are. The full task list, global constraints and rulings are in [`docs/superpowers/plans/2026-09-27-real-distribution-gaps.md`](superpowers/plans/2026-09-27-real-distribution-gaps.md). New knobs are switched on in `configs/generator/distributions/real_world_v3.yaml` (and `ko_admin_scan_v1.yaml` where they fit); `real_world_v2` stays the calibrated record. Runs without a profile behave as before.

## Done (merged to `main`)

| PR | Plan task | What it adds |
|---|---|---|
| #2 | — | Profile intent fixes: `family_mix` is honoured, `template_weights` (drop formula-only pages), a text corpus is required under a profile, `content.similar_char_ratio`, table/`hr` rules in ink colour |
| #3 | — | Plan document and the `real_world_v3` profile |
| #4 | 4 | `main.py rank-correlation`: Spearman/Kendall between synthetic and real model rankings, bootstrap CI |
| #5 | 5 | Page typography: `page.margins_mm` (page width kept constant), `typography.text_align` / `word_break`, `fonts.body` / `heading` / `code` |
| #6 | 2 | Capture artifacts: `stamp`, `highlighter`, `fold_lines`, `punch_holes`, `scanner_border`, `toner_streaks`, `page_curl`, `edge_crop` (crops the final photo, never cuts ink), `binarize_method`, band-shuffled `bleed_through` |
| #7 | 3 | Layout stats (line count/height, margins, columns, rules) in `distribution measure/compare`; `distribution text-stats` |
| #8 | 1 | Realistic tables: `content.table_schemas` (financial, budget, schedule, roster, order, statistics) with consistent numbers and right-aligned numeric columns |
| #9 | 9 | `typography.table_style` (web, grid, header_shaded, booktabs, borderless); `page.columns` (per family) and `column_gap_mm` |
| #10 | 6 | Korean conventions: `content.heading_numbering`, `list_style` (□ ○ · / 1) 가) ①), `law_articles` (제N조) |
| #11 | 8 | `distribution measure --save-rows` and `distribution discriminate` (classifier two-sample test, out-of-fold ranking of the most synthetic-looking pages) |
| #12 | 7 | Capture scenarios: `capture_channels.<ch>.scenarios` (scanned: office_adf, archive, photocopy, fax; photographed: phone_flat, phone_book, low_light) |
| (direct) | 10 | Merged-cell tables: HTML `<table>` with rowspan/colspan in GT (`content.merged_table_ratio`), centred span cells |
| (direct) | 11 | Genre-specific corpus: LLM categories + genre-aware `DataProvider` (`content.genre_corpus`) |
| #13 | 12 | Page furniture (header, footer, page number; excluded from GT, stripped in evaluation) + continuation pages (`page.header/footer/page_number`, `content.continuation_page/start_mid_paragraph`) |
| #14 | 14 | Capture parameter fitting from print-scan pairs (`distribution fit-capture`: ORB+RANSAC+ECC alignment, blur/noise/JPEG/dpi estimates, profile YAML snippet) |

The full suite passes on `main` except the 12 known stale tests listed below.

## In progress (not merged)

None. All worktree branches are merged and their worktrees/branches (local and remote) are deleted. One unreviewed stash remains: `post-PR local JPEG tweak` on top of `feat/capture-fit` (forward-simulates JPEG compression inside the blur search in `src/realism/capture_fit.py`); promote it to a follow-up PR or drop it with `git stash drop`.

## Not started

| Plan task | Depends on |
|---|---|
| 13. Figures with real content (charts, diagrams; adds matplotlib) | 6 (done), 12 |
| 15. Formula realism (plain display math, inline `$...$` in paragraphs) | 12 |
| 16. Realistic forms (label/value grids, checkbox groups, signature lines; profile-only templates) | 11, 13 |
| 17. Layout-conditioned re-rendering from DocLayNet-style layouts (`--layout-source`) | 12 |
| 18. LLM-generated HTML pages (`corpus generate-html`, `--html-source`) | 17 |

## Needs a person (out of scope for code)

- **Korean real reference set.** Get licensed pages (e.g. AI Hub 공공행정문서 OCR or the target domain), then run `distribution measure/compare/text-stats/discriminate` against `real_world_v3` output and re-fit the `[prior]` values. The calibration report covers `real_world_v2` only.
- **Real benchmark scores** for `rank-correlation`.
- **LLM API keys** to generate the genre corpora (Task 11) and HTML pages (Task 18).
- **Print-scan pairs** for Task 14: print synthetic pages, scan or photograph them, then fit.
- **HWP/HWPX templates:** skipped, no headless HWP renderer.

## Known issues and follow-ups

- 12 stale tests fail on `main` because they call removed APIs: `tests/generation/test_phase2_generation_modules.py` (2), `tests/generator/test_streaming_generation.py` (1), `tests/test_wave2_cli_pipeline_helpers.py` (9).
- The highlighter paints one rectangle over 1-3 lines, so it can overhang short lines and clip the top of the next line.
- `real_world_v3` scanned pages spend 45% on archive/photocopy/fax scenarios, so its scan statistics drift from the XFUND calibration by design. Re-measure once a reference set exists.
- `column_count` is unreliable on table-dominated pages and on photographed pages (documented in `src/realism/layout_stats.py`).
- Runs without a profile now also carry `heading_numbering`, `list_style` and `law_articles_used` metadata columns (legacy values).

## How to continue

- One branch and PR per task, reviewed and then merged. Follow the plan's global constraints: gate every feature behind a profile key, switch it on in `real_world_v3.yaml`, keep GT matching the image, test first, and commit + push WIP often.
- Full `uv sync` pulls torch, so tests run in a lightweight env:

  ```bash
  PYTHONPATH=. uv run --no-project --python 3.11 --with pytest --with faker \
    --with pillow --with opencv-python-headless --with numpy --with markdown \
    --with "playwright==1.55.0" --with pyyaml --with tqdm --with mistune \
    --with scikit-image --with datasets --with huggingface-hub --with pydantic \
    pytest -q --ignore=tests/test-gemini.py tests
  ```
