# PDF Extraction Pipeline

## Goal

Turn academic papers (PDFs) into model-friendly Markdown + images. Extract text as Markdown, tables and figures as images, preserving reading order and inline placement of callout boxes.

## Pipeline Overview

```
PDF Page
  │
  ├─► render (page → high-res PNG)
  │
  ├─► analyze (page → raw text block positions from PDF)
  │
  ▼
Detection Model (image + analyze blocks → layout JSON)
  │
  ▼
draw (layout JSON → annotated PNG for visual verification)
  │
  ├─► human or model reviews → iterate if boxes are wrong
  │
  ▼
Assembly Model (boxed page + extracted content → Markdown)
```

## Region Types

Only three:

| Type | Meaning | Extraction |
|------|---------|------------|
| `text` | Text content to extract | Extract via pdfplumber using bounding box |
| `image` | Visual content (figures, tables with charts, diagrams) | Crop as high-res PNG |
| `skip` | Headers, footers, page numbers, line numbers | Ignore during assembly |

Tables are `image` for now. The assembly model can still read them visually when producing markdown.

## Layout JSON Format

```json
{
  "page": 3,
  "regions": [
    {"id": "skip_hdr",  "type": "skip",  "bbox": [0.07, 0.055, 0.93, 0.09]},
    {"id": "txt_L1",    "type": "text",  "bbox": [0.07, 0.093, 0.49, 0.42]},
    {"id": "img_1",     "type": "image", "bbox": [0.077, 0.414, 0.474, 0.639]},
    {"id": "txt_R1",    "type": "text",  "bbox": [0.51, 0.093, 0.93, 0.137]}
  ]
}
```

- `bbox` is normalized coordinates `[x0, y0, x1, y1]` where `(0,0)` is top-left and `(1,1)` is bottom-right
- IDs are arbitrary but unique per page
- A small margin (~1%) around each region is good — overlaps between adjacent regions are OK

## Detection Step

**Input to model:**
1. Rendered page image (PNG)
2. Output of `analyze` command — raw text block positions with snippets

**Model's job:**
- Group blocks into logical regions
- Classify each region as `text`, `image`, or `skip`
- The model does NOT guess coordinates — it uses the real block positions from `analyze` and combines/groups them
- Output: layout JSON

**Key insight from experiments:** Vision models struggle to estimate precise normalized coordinates from images alone. Combining the visual (to understand what things are) with programmatic block positions (to know where things are) works much better.

## Assembly Step

**Input to model:**
1. Annotated page image (with bounding boxes drawn)
2. Extracted content per region:
   - `text` regions: extracted text from pdfplumber
   - `image` regions: cropped PNG files
3. The layout JSON

**Model's job:**
- Decide reading order across all regions
- Decide markdown formatting for each text region (paragraph, blockquote for callouts, heading level, etc.)
- Place image references inline at the correct position
- A callout/takeaway box is just a text region — the assembly model decides it should be a blockquote based on visual context
- Figures and tables get two images: one with caption, one without (crop using iteration)

## Script: experiment_01.py

General-purpose tool with no layout assumptions baked in:

```
uv run experiment_01.py render [PAGE_NUMS]      # PDF pages → PNG
uv run experiment_01.py grid [PAGE_NUMS]         # PNG with coordinate grid overlay
uv run experiment_01.py analyze [PAGE_NUMS]      # Dump raw text block positions
uv run experiment_01.py draw LAYOUT_JSON_FILE    # Draw bounding boxes → annotated PNG
```

`PAGE_NUMS` defaults to `3,6,7`. Comma-separated, 1-indexed.

## Tech Stack

- **PyMuPDF (fitz)**: PDF rendering, annotation (drawing boxes), text block extraction
- **pdfplumber**: Text extraction from specific bounding box regions (for the extraction phase)
- **Python + uv**: Runtime with inline script dependencies
- **Claude**: Vision model for detection and assembly steps

## Key Learnings from Experiments

1. **Vision models can't precisely estimate coordinates from page images.** Normalized coordinate guesses were consistently off, sometimes by 10-15% of page height. The grid overlay helps but adds complexity.

2. **Combining vision + programmatic data works well.** The `analyze` command gives precise block positions from the PDF itself. The model's job becomes grouping and classifying blocks, not guessing where they are.

3. **Simple region types are sufficient.** We started with 10+ types (text_block, table, figure, callout, heading, caption, header, footer, page_number, line_numbers) and simplified to 3: `text`, `image`, `skip`. Classification and formatting decisions belong in the assembly step, not detection.

4. **Tables with embedded visualizations should be images.** Complex tables (color gradients, mini bar charts, merged cells) are not reliably extractable as text/markdown. Extract as high-res images — models can read them visually when needed.

5. **Callouts and takeaway boxes are just text regions.** They get placed inline in reading order. The assembly model decides formatting (e.g., blockquote) based on visual context.

6. **Iterate with visual verification on every page.** Always produce an annotated image with bounding boxes drawn. Don't iterate (send back for adjustment) on every page — but always produce the debug overlay for review.

7. **Margins and overlaps are fine.** A small margin around each region prevents clipping. Adjacent regions can overlap slightly — that's better than gaps that miss content.

8. **Version your outputs.** Layout JSONs and annotated PNGs should be versioned (v1, v2, v3...) so you have provenance as you iterate.

9. **Paper-specific artifacts (line numbers, column counts) should not be hardcoded.** The script must be general-purpose. Layout interpretation is the model's job, not the script's.

## Test Pages

Using `AI-Where-It_Matters.pdf`:

- **Page 3**: Two-column text with two simple tables (one per column)
- **Page 6**: Full-width complex table with embedded bar charts spanning top half, two text columns below, takeaway callout box
- **Page 7**: Scatter plot figure on left, text with inline takeaway callout, continuous right column text

## Next Steps

- [ ] Build the detection prompt: given image + analyze output, produce layout JSON
- [ ] Build text extraction using pdfplumber with bounding boxes from layout JSON
- [ ] Build image cropping for `image` regions
- [ ] Build the assembly prompt: given boxed page + extracted content, produce Markdown
- [ ] Figure/table caption splitting: extract whole region as image, then use model to crop caption-only vs content-only versions
- [ ] Test on more diverse papers (single column, three column, different publishers)
