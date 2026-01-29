# Generation

Generation is orchestrated by `src/pipeline.py` and format-specific generators in `src/generator/`.

## Common Flow

1. Ensure corpus and character similarity DB (sentence only).
2. Instantiate the generator for the selected format.
3. Generate images and metadata.
4. Write `metadata.jsonl` and `realism_stats.json`, then upload to Hugging Face.

Output directory layout:

```
data/<lang>/
  corpus_<lang>.txt
  char_similarity_db_<lang>.json
  images_sentence/
  images_table/
  images_document/
  images_markdown/
  images_kie/
  images_mixed/
  realism_stats.json
```

`realism_stats.json` includes:

- `total_samples` and optional `format`/`format_counts`
- `field_presence` counts per metadata key
- length stats for text, list, and dict fields
- numeric field summary stats (min/max/mean)

## Sentence

Generator: `SentenceGenerator`

Metadata fields:

- `typo_text` (target for evaluation)
- `original_text`
- rendering params (font path, background, blur, etc.)

Notes:

- Uses Wikipedia corpus and a character similarity DB.
- `typo_ratio` controls typo injection.
- Use `--seed` for reproducible generation.

## Table

Generator: `TableGenerator`

Metadata fields:

- `html` (table HTML)
- `json` (table structure and cells)
- `template`, `num_rows`, `num_cols`, `font_size`

Template choices: `invoice`, `schedule`, `product`, `contact`.

## Document

Generator: `DocumentGenerator`

Metadata fields:

- `ground_truth` (elements with `type`, `text`, `bounding_box`, `reading_order`)
- `template`, `elements_count`, `font_size`
- `add_noise`, `add_blur`

Template choices: `invoice`, `receipt`, `form`, `letter`, `report`.

## Markdown

Generator: `MarkdownGenerator`

Metadata fields:

- `markdown` (raw markdown)
- `template`, `add_noise`, `add_blur`

Template choices: `readme`, `technical_doc`, `blog_post`, `api_doc`, `tutorial`.

## KIE

Generator: `KIEGenerator`

Metadata fields:

- `document_type`
- `ground_truth` (entities, line_items, raw_text)

Document types: `receipt`, `invoice`, `form`, `business_card`.

## Mixed

`--mixed` uses `MixedGenerator` to combine formats. It writes a single `metadata.jsonl` containing a `format` field, then uploads each format as its own dataset subset.
