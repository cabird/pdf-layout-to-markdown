# Table Extraction Design

## Problem

Currently all tables are treated as `image` regions — cropped as PNGs. This works but means tables aren't searchable text. We want to extract simple tables as markdown when possible, while keeping complex tables as images.

## Design Decisions

Based on discussion with GPT 5.2 and project constraints:

### 1. Classification: LLM detects, pdfplumber decides

- **Detection LLM** classifies `image` regions as `table` or `figure` (new `image_kind` field). This is a visual decision the LLM is good at.
- **pdfplumber** attempts extraction and **scores** the result. This is deterministic and cheap.
- The LLM does NOT decide simple vs complex — that's based on extraction quality.

### 2. Keep 3 region types, add `image_kind`

Rather than adding new types (which would cascade through validation, colors, assembly), we add an optional `image_kind` field to `image` regions:

```json
{"id": "img_1", "type": "image", "image_kind": "table", "bbox": [...]}
{"id": "img_2", "type": "image", "image_kind": "figure", "bbox": [...]}
```

### 3. Extraction approach: programmatic-first

1. Always crop the image (for provenance/fallback)
2. Attempt pdfplumber table extraction on the bbox
3. Score the extraction quality
4. If good enough → use markdown table in assembly
5. If not → keep as image (current behavior)

No extra LLM calls for table extraction. Token-cheap.

### 4. Quality scoring for extracted tables

Accept markdown if ALL of:
- pdfplumber found a table in the bbox
- No ragged rows (all rows have same column count)
- `empty_ratio` (empty cells / total cells) < 0.35
- At least 2 rows and 2 columns
- Table has content (not all empty)

Reject (keep as image) if ANY of:
- No table found by pdfplumber
- Ragged rows (inconsistent column count)
- Too many empty cells
- Single row or single column

### 5. Assembly integration

When a table is successfully extracted as markdown:
- The assembly LLM receives it as a text block containing a markdown table
- Assembly prompt updated to say "preserve markdown tables as-is"
- The cropped image is still saved for reference but not included in final markdown

When extraction fails:
- Falls back to current image behavior — no change

### 6. Flag: `--tables`

- Off by default (current behavior preserved)
- When enabled:
  - Detection prompt asks for `image_kind` on image regions
  - Table extraction attempted on `image_kind: "table"` regions
  - Quality-scored and either converted to markdown or kept as image

### 7. What we're NOT doing (yet)

- LLM-based table repair (sending cropped image to LLM to fix bad extraction)
- Footnote handling (separate feature)
- Multi-level header flattening
- Table notes/captions association

## Implementation Changes

1. `validate_layout()` — allow optional `image_kind` field on image regions
2. `DETECTION_SYSTEM_PROMPT` — conditionally add instruction for `image_kind`
3. New function `extract_table_as_markdown()` — pdfplumber extraction + scoring
4. `process_page()` — when `--tables`, attempt table extraction before falling back to image
5. Assembly prompt — tell model to preserve markdown tables
6. `main()` — add `--tables` argument
