# Document Structure Diversity Design

## Goal

Increase generated document structure and document-type diversity for synthetic OCR datasets while preserving the current generation pipeline, CLI shape, metadata flow, and reproducibility model.

The current markdown generator mostly samples counts for text, table, and formula sections, then shuffles those sections. That gives useful local variation, but document-level structure saturates quickly because the default catalog has one template and a small block vocabulary.

This design expands diversity through richer block composition and more template families, not through scan noise or post-render visual effects.

## Success Criteria

- Generation can produce multiple document families such as reports, manuals, meeting notes, academic notes, forms, API docs, release notes, policy documents, table-heavy documents, and formula-heavy documents.
- Templates can request varied block types, including paragraphs, tables, formulas, bullet lists, numbered lists, checklists, quotes, code blocks, command blocks, image placeholders, and horizontal rules.
- Existing generation commands continue to work with the current defaults.
- Generated metadata records enough structure information to audit diversity after a run.
- Novelty checks continue to use structural signatures, and the new block types contribute to those signatures.
- Tests cover template parsing, block generation, metadata fields, reproducibility, and small end-to-end generation.

## Non-Goals

- Do not implement scanned-paper visual difficulty in this work. That remains covered by the scan difficulty profile design.
- Do not add new model dependency groups.
- Do not require external downloads or network calls.
- Do not replace the current renderer stack.
- Do not commit generated datasets or evaluation artifacts.

## Recommended Approach

Use an incremental block-based expansion of the existing section generator.

This keeps the current `Generator`, `TemplateCatalog`, renderer selection, metadata writing, sharding, and publish flow intact. The main change is that template blueprints can describe a richer block mix, and `MarkdownDataGenerator` can compose those blocks into document-like structures.

This is preferable to immediately building separate hard-coded archetype generators because it gives most of the diversity benefit with less duplication and a smaller behavioral surface. If later templates need very specific flows, archetype-specific helpers can be added on top of the same block primitives.

## Architecture

### Template Catalog

Keep YAML templates under `configs/generator/templates/*.yaml`.

Each template should remain a `TemplateSpec` with:

- `id`
- `family`
- `complexity`
- `weight`
- `mode`
- `aliases`
- `tags`
- `blueprint`

The first implementation should support `mode: sections` and treat `mode: blueprint` as an alias for the same block-composition path only if that can be done without breaking existing behavior. The current implementation coerces unsupported modes to `sections`, so the implementation should either keep using `sections` consistently or explicitly normalize supported aliases.

### Blueprint Shape

Extend the existing section blueprint with optional block-composition keys:

```yaml
blueprint:
  document_shape: report
  title_prefixes: [Weekly Report, Research Note]
  section_count: [3, 6]
  blocks_per_section: [1, 3]
  allowed_blocks:
    - paragraph
    - bullet_list
    - numbered_list
    - checklist
    - table
    - formula
    - quote
    - code
    - command
    - image
    - rule
  required_blocks:
    - table
  text:
    max_line_chars: 72
  table:
    rows: [2, 6]
    columns: [3, 5]
  formula:
    section_count: [0, 2]
```

Backward-compatible keys such as `text.section_count`, `table.section_count`, and `formula.section_count` should keep working.

### Block Generators

Add focused block-generation helpers that return markdown snippets and a block type label. They can start as functions or small classes, depending on the surrounding code shape during implementation.

Required block types:

- `paragraph`: heading plus wrapped paragraph text, using existing `TextGenerator` behavior.
- `bullet_list`: unordered list of features, requirements, or paragraph fragments.
- `numbered_list`: ordered procedure or step list.
- `checklist`: task-style checklist with checked and unchecked items.
- `table`: existing table generator output.
- `formula`: existing formula generator output.
- `quote`: markdown blockquote using generated sentence text.
- `code`: fenced code/config block using install commands, usage commands, config lines, or API endpoints.
- `command`: shell-style fenced command block or inline command section.
- `image`: markdown placeholder image line that existing renderers can draw as a stable placeholder.
- `rule`: markdown horizontal rule.

Keep each block self-contained so the orchestrator can shuffle or place it without needing to inspect internals.

### Document Composer

Replace the current fixed `text_sections + table_sections + formula_sections` merge path with a composer that can:

- Choose a document title.
- Choose section count from the blueprint.
- Satisfy `required_blocks`.
- Fill remaining slots from `allowed_blocks` using weights or uniform sampling.
- Keep simple document-level shape rules, such as title first and optional rule separators.
- Return markdown plus structural metadata.

The existing simple path should be preserved when a template only provides legacy `text`, `table`, and `formula` section counts.

### Metadata

Add metadata fields that support diversity audits:

- `document_family`: same as `template_family`.
- `document_shape`: blueprint `document_shape` or fallback template family.
- `block_types`: ordered list of block types used in the sample.
- `block_type_counts`: count map for block types.
- `section_count`: number of top-level generated sections.

Existing fields such as `template_id`, `template_family`, `template_complexity`, `merge_order`, `structure_signature`, `image_width`, and `image_height` remain.

### Novelty Signature

Extend `_structure_signature()` only where needed so new markdown patterns are represented distinctly:

- fenced code blocks as `code`
- image placeholders as `image`
- blockquotes as `quote`
- checklist items as `check`
- horizontal rules as `rule`
- ordered and unordered lists as separate tokens

Most of these tokens are already present, so the main requirement is to ensure generated blocks actually use those markdown forms consistently.

## Template Families

Seed the catalog with several concrete templates:

- `business_report`: paragraphs, bullets, tables, quotes.
- `meeting_minutes`: checklist, numbered list, table, short paragraphs.
- `technical_manual`: numbered list, command block, code block, table.
- `api_reference`: code block, command block, table, bullets.
- `academic_note`: formula, paragraph, quote, image placeholder.
- `release_note`: bullets, checklist, code block, rule.
- `policy_document`: numbered list, quote, paragraphs, checklist.
- `form_like`: table-heavy, checklist, short fields.
- `table_heavy`: multiple tables with short text.
- `formula_heavy`: multiple formulas with explanatory paragraphs.

Families should be useful with `--coverage-target`, for example:

- `business`
- `technical`
- `academic`
- `operations`
- `forms`

## Data Flow

```text
template selection
  -> blueprint resolution
  -> document composer
  -> block generator calls
  -> markdown text
  -> similar-character mutation
  -> structure signature and novelty check
  -> style sampling
  -> markdown rendering
  -> metadata attachment
  -> shard/root metadata
```

## Error Handling

- Unknown block types should warn and be ignored.
- If all requested block types are invalid, fall back to paragraph/table/formula defaults.
- If `required_blocks` cannot be satisfied because a type is unknown, warn and continue with valid requirements.
- Invalid count ranges should fall back to safe defaults.
- Empty generated blocks should be skipped.
- If all blocks are skipped, generate at least one paragraph block.

## Testing Plan

Unit tests:

- Template catalog loads multiple templates and preserves family, complexity, tags, and blueprint fields.
- Blueprint parsing accepts legacy section-count templates and richer block templates.
- Block generation returns non-empty markdown for each supported block type.
- Required blocks appear in generated metadata when possible.
- Invalid block names fall back without failing generation.
- Seeded generation remains deterministic for the same sample index and seed.

Integration-style tests:

- Generate a small dataset with the expanded catalog and assert multiple `document_family` and `block_types` values appear.
- Generate with `--coverage-target` and assert underrepresented families are sampled.
- Run a small CLI generation smoke test with `--style-profile aggressive` and low novelty threshold.

Manual smoke command:

```bash
uv run main.py generate \
  --lang ko \
  --size 20 \
  --seed 42 \
  --style-profile aggressive \
  --novelty-threshold 0.92 \
  --novelty-max-attempts 6
```

After generation, inspect `metadata.jsonl` for `document_family`, `document_shape`, `block_types`, and `structure_signature` distribution.

## Rollout Plan

1. Add tests for the new block composition behavior.
2. Implement block generation helpers and metadata capture.
3. Expand the template catalog with multiple document families.
4. Update generation docs with diversity recipes.
5. Run focused unit tests and a small generation smoke test.

## Deferred Work

- Dedicated archetype classes for highly structured documents such as invoices, contracts, worksheets, and academic papers.
- Fine-grained block weights per template.
- Multi-page document generation.
- Diversity scoring reports built into the CLI.
- Corpus expansion for domain-specific document language.
