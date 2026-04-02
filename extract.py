#!/usr/bin/env python3
# /// script
# dependencies = ["pymupdf", "pdfplumber", "openai", "python-dotenv", "rich"]
# requires-python = ">=3.11"
# ///
"""
PDF page extraction pipeline: PDF pages → Markdown + images via LLM-guided layout detection.

Usage:
    uv run extract.py paper.pdf [--pages 3,6,7] [-o output_dir/]

For each page:
  1. Render page to PNG + extract text blocks (PyMuPDF)
  2. LLM call 1 — detect layout regions from image + block positions
  3. Extract text (pdfplumber) and crop images for each region
  4. LLM call 2 — assemble final markdown from extracted content
  5. Write outputs: layout JSON, annotated PNG, cropped images, markdown
"""

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.text import Text

DPI = 300
MODEL = "gpt-5-mini"

# Pricing per million tokens (gpt-5-mini)
COST_PER_M = {"input": 0.25, "output": 2.00, "thinking": 2.00}

# Accumulated token usage across all LLM calls
usage_totals = {"input": 0, "output": 0, "thinking": 0, "calls": 0}
_lock = threading.Lock()
console = Console()
_error_log_path = None


def log(msg):
    """Thread-safe print via Rich console."""
    console.print(msg, highlight=False)


def log_error(label, exc):
    """Append a timestamped error with full traceback to the error log file."""
    if _error_log_path is None:
        return
    ts = time.strftime("%H:%M:%S")
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    with _lock:
        with open(_error_log_path, "a") as f:
            f.write(f"[{ts}] {label}\n")
            f.writelines(tb)
            f.write("\n")


def track_usage(resp, label):
    """Accumulate token usage from an OpenAI response."""
    u = resp.usage
    input_tok = u.prompt_tokens
    output_tok = u.completion_tokens
    thinking_tok = getattr(u, "completion_tokens_details", None)
    thinking_tok = getattr(thinking_tok, "reasoning_tokens", 0) if thinking_tok else 0
    text_tok = output_tok - thinking_tok

    with _lock:
        usage_totals["input"] += input_tok
        usage_totals["output"] += text_tok
        usage_totals["thinking"] += thinking_tok
        usage_totals["calls"] += 1


def print_usage_summary():
    """Print accumulated usage totals and cost."""
    t = usage_totals
    total_cost = (
        t["input"] * COST_PER_M["input"] / 1_000_000
        + t["output"] * COST_PER_M["output"] / 1_000_000
        + t["thinking"] * COST_PER_M["thinking"] / 1_000_000
    )
    log(f"Token usage ({t['calls']} LLM calls):")
    log(f"  Input:    {t['input']:>10,} tokens  (${t['input'] * COST_PER_M['input'] / 1_000_000:.4f})")
    log(f"  Output:   {t['output']:>10,} tokens  (${t['output'] * COST_PER_M['output'] / 1_000_000:.4f})")
    if t["thinking"]:
        log(f"  Thinking: {t['thinking']:>10,} tokens  (${t['thinking'] * COST_PER_M['thinking'] / 1_000_000:.4f})")
    log(f"  Total cost: ${total_cost:.4f}")


def slugify(name):
    """Convert a filename to a URL-friendly slug."""
    stem = Path(name).stem
    slug = re.sub(r'[^a-z0-9]+', '-', stem.lower()).strip('-')
    return slug


def find_output_dir(base):
    """Find an available output directory, appending _v2, _v3, etc. if needed."""
    if not base.exists():
        return base
    v = 2
    while True:
        candidate = base.parent / f"{base.name}_v{v}"
        if not candidate.exists():
            return candidate
        v += 1


class StatusLine:
    """Mutable Rich renderable for a live status line."""

    def __init__(self):
        self.text = ""
        self._lock = threading.Lock()

    def update(self, text):
        with self._lock:
            self.text = text

    def __rich__(self):
        if self.text:
            return Text(f"  {self.text}", style="dim italic")
        return Text("")


class CostLine:
    """Mutable Rich renderable showing live accumulated cost."""

    def __rich__(self):
        t = usage_totals
        cost = (
            t["input"] * COST_PER_M["input"] / 1_000_000
            + t["output"] * COST_PER_M["output"] / 1_000_000
            + t["thinking"] * COST_PER_M["thinking"] / 1_000_000
        )
        if t["calls"] == 0:
            return Text("")
        return Text(f"  ${cost:.4f} ({t['calls']} LLM calls)", style="dim")


MAX_RETRIES = 5
RETRY_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError)


def llm_call(client, label, **kwargs):
    """Make an OpenAI chat completion with exponential backoff on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            track_usage(resp, label)
            return resp
        except RETRY_EXCEPTIONS as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = 2 ** attempt
            log(f"    [{label}] {type(e).__name__}, retrying in {delay}s...")
            time.sleep(delay)


def validate_layout(data):
    """Validate detection JSON structure. Returns list of error strings (empty = valid)."""
    errors = []
    if not isinstance(data, dict):
        return [f"Expected a JSON object, got {type(data).__name__}"]
    if "regions" not in data:
        errors.append("Missing 'regions' key")
    elif not isinstance(data["regions"], list):
        errors.append(f"'regions' should be a list, got {type(data['regions']).__name__}")
    else:
        for i, r in enumerate(data["regions"]):
            prefix = f"regions[{i}]"
            if not isinstance(r, dict):
                errors.append(f"{prefix}: expected object, got {type(r).__name__}")
                continue
            for key in ("id", "type", "bbox"):
                if key not in r:
                    errors.append(f"{prefix}: missing '{key}'")
            if "type" in r and r["type"] not in ("text", "image", "skip"):
                errors.append(f"{prefix}: type must be 'text', 'image', or 'skip', got '{r['type']}'")
            if "image_kind" in r and r["image_kind"] not in ("table", "figure"):
                errors.append(f"{prefix}: image_kind must be 'table' or 'figure', got '{r['image_kind']}'")
            if "bbox" in r:
                bbox = r["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"{prefix}: bbox must be a list of 4 numbers")
                elif not all(isinstance(v, (int, float)) for v in bbox):
                    errors.append(f"{prefix}: bbox values must be numbers")
    if "reading_order" not in data:
        errors.append("Missing 'reading_order' key")
    elif not isinstance(data["reading_order"], list):
        errors.append(f"'reading_order' should be a list, got {type(data['reading_order']).__name__}")
    return errors


def sanitize_bbox(bbox, label=""):
    """Clamp bbox values to 0-1 and validate x0 < x1, y0 < y1."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, y0 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, y0))
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"Degenerate bbox {label}: [{x0}, {y0}, {x1}, {y1}]")
    return [x0, y0, x1, y1]


LAYOUT_FIX_FORMAT = """\
I expected JSON in this exact format:
{{
  "regions": [
    {{"id": "txt_1", "type": "text", "bbox": [x0, y0, x1, y1]}},
    {{"id": "img_1", "type": "image", "bbox": [x0, y0, x1, y1]}}
  ],
  "reading_order": ["txt_1", "img_1"]
}}

Rules:
- "type" must be one of: "text", "image", "skip"
- "bbox" must be a list of 4 numbers (normalized 0-1)
- "reading_order" must list IDs of non-skip regions

But I got this output instead:
{raw_output}

Validation errors:
{errors}

Please fix and return valid JSON only."""


# Colors for region types (R, G, B) in 0-1 range
REGION_COLORS = {
    "text": (0, 0, 0.9),
    "image": (1, 0.5, 0),
    "skip": (0.6, 0.6, 0.6),
}

# ---------------------------------------------------------------------------
# Rendering & analysis
# ---------------------------------------------------------------------------


def render_page(doc, page_idx, dpi=DPI):
    """Render a page to PNG bytes. Returns (png_bytes, width, height)."""
    page = doc[page_idx]
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return pix.tobytes("png"), pix.width, pix.height


def analyze_page(doc, page_idx):
    """Extract text blocks with normalized bboxes. Returns list of dicts."""
    page = doc[page_idx]
    r = page.rect
    blocks = page.get_text("blocks")

    result = []
    for b in blocks:
        x0, y0, x1, y1, text, bno, btype = b
        result.append({
            "bbox": [
                round(x0 / r.width, 3),
                round(y0 / r.height, 3),
                round(x1 / r.width, 3),
                round(y1 / r.height, 3),
            ],
            "text": text.strip().replace("\n", " ")[:200],
            "type": "img" if btype == 1 else "txt",
        })

    result.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return result


# ---------------------------------------------------------------------------
# LLM Call 1: Detection
# ---------------------------------------------------------------------------

DETECTION_SYSTEM_PROMPT = """\
You are a document layout analyzer. You receive:
1. A page image from a PDF
2. A list of text blocks extracted from that page, each with a normalized \
bounding box [x0, y0, x1, y1] where coordinates range from 0 to 1, and a \
text snippet.

Your task:
- Identify the logical regions on the page and classify each as:
  - "text" — body text, headings, captions (will be extracted as text)
  - "image" — figures, charts, diagrams, photos, complex tables (will be cropped as an image)
  - "skip" — headers, footers, page numbers, margin artifacts (will be ignored)
- Group adjacent blocks into the same region ONLY when they are part of the same \
semantic section with no visual boundary between them. Start a NEW region when you see:
  - A background color change (e.g. a tinted/shaded callout box or sidebar)
  - A visible border, rule, or divider line
  - A shift in content type (e.g. body text → reference list, → author bios, \
→ acknowledgments, → a boxed sidebar or callout)
  - A distinct typographic section even within the same column
  The goal is semantic regions, not geometric columns. A single column often contains \
multiple distinct regions.
- Figure/table captions MUST be separate "text" regions, not merged into the "image" \
region. The image region should tightly cover only the visual content. Place the caption \
region immediately before or after the image in the reading order so the assembly step \
can associate them.
- Construct a bounding box for each region in normalized coordinates [x0, y0, x1, y1].
- Add approximately 1% margin around each region. Slight overlaps between regions are OK.
- Assign each region a descriptive ID like "txt_1", "img_1", "skip_hdr", "txt_caption_1", \
"txt_sidebar_1", "txt_refs_1", "txt_bio_1".
- Suggest a reading order listing the IDs of non-skip regions.

Return JSON only, no commentary. Format:
{
  "regions": [
    {"id": "txt_1", "type": "text", "bbox": [x0, y0, x1, y1]},
    {"id": "img_1", "type": "image", "bbox": [x0, y0, x1, y1]}
  ],
  "reading_order": ["txt_1", "img_1", "txt_2"]
}"""

DETECTION_TABLE_ADDENDUM = """

Additional instruction: For each "image" region, add an "image_kind" field:
- "table" — if the region contains a data table (rows and columns of text/numbers)
- "figure" — if the region contains a chart, diagram, photo, or other visual content

Example: {"id": "img_1", "type": "image", "image_kind": "table", "bbox": [...]}"""


def detect_layout(client, png_bytes, blocks, page_num, tables=False):
    """LLM call 1: detect layout regions from page image + text blocks."""
    png_b64 = base64.b64encode(png_bytes).decode()

    blocks_text = "\n".join(
        f"  {i:3d} {b['type']} [{b['bbox'][0]:.3f}, {b['bbox'][1]:.3f}, "
        f"{b['bbox'][2]:.3f}, {b['bbox'][3]:.3f}]  {b['text'][:90]}"
        for i, b in enumerate(blocks)
    )

    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{png_b64}", "detail": "high"},
        },
        {
            "type": "text",
            "text": (
                f"Page {page_num}. Here are the extracted text blocks "
                f"(normalized coordinates):\n\n{blocks_text}"
            ),
        },
    ]

    system_prompt = DETECTION_SYSTEM_PROMPT
    if tables:
        system_prompt += DETECTION_TABLE_ADDENDUM

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    max_fix_attempts = 2
    for attempt in range(1 + max_fix_attempts):
        label = f"detect p{page_num}" if attempt == 0 else f"detect p{page_num} fix#{attempt}"
        resp = llm_call(client, label, model=MODEL, messages=messages,
                        response_format={"type": "json_object"})
        raw = resp.choices[0].message.content

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt == max_fix_attempts:
                raise ValueError(f"Page {page_num}: detection JSON never parsed after {max_fix_attempts} fixes: {e}")
            log(f"    [detect p{page_num}] invalid JSON, asking for fix...")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": LAYOUT_FIX_FORMAT.format(
                raw_output=raw[:500], errors=str(e))})
            continue

        errors = validate_layout(data)
        if not errors:
            return data

        if attempt == max_fix_attempts:
            raise ValueError(f"Page {page_num}: detection JSON invalid after {max_fix_attempts} fixes: {errors}")
        log(f"    [detect p{page_num}] schema errors: {errors}, asking for fix...")
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": LAYOUT_FIX_FORMAT.format(
            raw_output=raw[:500], errors="\n".join(errors))})


# ---------------------------------------------------------------------------
# Drawing boxes (for annotated review image)
# ---------------------------------------------------------------------------


def norm_to_pdf(bbox, page_rect):
    """Convert normalized [x0, y0, x1, y1] (0-1) to a fitz.Rect in PDF points."""
    x0, y0, x1, y1 = bbox
    return fitz.Rect(
        page_rect.x0 + x0 * page_rect.width,
        page_rect.y0 + y0 * page_rect.height,
        page_rect.x0 + x1 * page_rect.width,
        page_rect.y0 + y1 * page_rect.height,
    )


def draw_labeled_rect(page, pdf_rect, label, color, line_width=1.5):
    """Draw a colored rectangle with a label on a PDF page."""
    shape = page.new_shape()
    shape.draw_rect(pdf_rect)
    shape.finish(color=color, width=line_width, fill=color, fill_opacity=0.08)
    shape.commit()

    fontsize = 7
    text_width = fitz.get_text_length(label, fontsize=fontsize)
    bg_rect = fitz.Rect(
        pdf_rect.x0, pdf_rect.y0 - 11,
        pdf_rect.x0 + text_width + 4, pdf_rect.y0 - 1,
    )
    page.draw_rect(bg_rect, color=color, fill=(1, 1, 1), fill_opacity=0.9)
    page.insert_text(
        fitz.Point(pdf_rect.x0 + 2, pdf_rect.y0 - 3),
        label, fontsize=fontsize, color=color,
    )


def draw_boxes(pdf_path, page_idx, layout):
    """Draw bounding boxes from layout onto the page. Returns annotated PNG bytes."""
    tmp_doc = fitz.open(str(pdf_path))
    page = tmp_doc[page_idx]
    page_rect = page.rect
    reading_order = layout.get("reading_order", [])

    for region in layout["regions"]:
        if region.get("image_kind") == "table":
            color = (0, 0.7, 0)  # Green for tables
        else:
            color = REGION_COLORS.get(region["type"], (0.5, 0.5, 0.5))
        pdf_rect = norm_to_pdf(region["bbox"], page_rect)

        order_pos = ""
        if reading_order and region["id"] in reading_order:
            idx = reading_order.index(region["id"]) + 1
            order_pos = f" #{idx}"

        draw_labeled_rect(page, pdf_rect, f"{region['id']}{order_pos}", color)

    scale = DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    png_bytes = pix.tobytes("png")
    tmp_doc.close()
    return png_bytes


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------


def extract_text_regions(pdf_path, page_num, regions):
    """Extract text from multiple bbox regions using pdfplumber (single file open).

    Args:
        regions: list of (region_id, norm_bbox) tuples
    Returns:
        dict of region_id -> extracted text
    """
    results = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_num - 1]
        w, h = page.width, page.height
        for rid, norm_bbox in regions:
            x0, y0, x1, y1 = norm_bbox
            crop_box = (x0 * w, y0 * h, x1 * w, y1 * h)
            cropped = page.crop(crop_box)
            results[rid] = cropped.extract_text() or ""
    return results


def crop_image_region(page, norm_bbox):
    """Crop a region from a fitz page at full DPI. Returns cropped PNG bytes."""
    r = page.rect
    x0, y0, x1, y1 = norm_bbox
    clip = fitz.Rect(x0 * r.width, y0 * r.height, x1 * r.width, y1 * r.height)
    scale = DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
    return pix.tobytes("png")


def extract_table_as_markdown(pdf_path, page_num, norm_bbox):
    """Try to extract a table from a bbox region using pdfplumber.

    Returns (markdown_str, score_info) if successful, or (None, score_info) if
    the extraction quality is too low.
    """
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_num - 1]
        w, h = page.width, page.height
        x0, y0, x1, y1 = norm_bbox
        crop_box = (x0 * w, y0 * h, x1 * w, y1 * h)
        cropped = page.crop(crop_box)

        tables = cropped.find_tables()
        if not tables:
            return None, {"reason": "no table found by pdfplumber"}

        # Use the largest table found in the region
        table = max(tables, key=lambda t: len(t.extract()))
        rows = table.extract()

        if not rows or len(rows) < 2:
            return None, {"reason": f"too few rows: {len(rows) if rows else 0}"}

        # Score the extraction quality
        n_cols = len(rows[0])
        if n_cols < 2:
            return None, {"reason": f"too few columns: {n_cols}"}

        ragged = [i for i, row in enumerate(rows) if len(row) != n_cols]
        if ragged:
            return None, {"reason": f"ragged rows (inconsistent column count): rows {ragged}"}

        total_cells = len(rows) * n_cols
        empty_cells = sum(1 for row in rows for cell in row if not cell or not cell.strip())
        empty_ratio = empty_cells / total_cells if total_cells > 0 else 1.0

        if empty_ratio > 0.35:
            return None, {"reason": f"too many empty cells: {empty_ratio:.0%}"}

        score_info = {
            "rows": len(rows),
            "cols": n_cols,
            "empty_ratio": round(empty_ratio, 2),
            "accepted": True,
        }

        # Build markdown table
        def clean_cell(cell):
            if cell is None:
                return ""
            return cell.strip().replace("|", "\\|").replace("\n", " ")

        header = rows[0]
        md_lines = []
        md_lines.append("| " + " | ".join(clean_cell(c) for c in header) + " |")
        md_lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in rows[1:]:
            md_lines.append("| " + " | ".join(clean_cell(c) for c in row) + " |")

        return "\n".join(md_lines), score_info


TABLE_LLM_SYSTEM_PROMPT = """\
You are a table extraction specialist. You receive a cropped image of a table \
from an academic paper.

Return JSON with three fields:
1. "title": a short title (max 20 words) summarizing what the table shows.
2. "description": 1-3 sentences describing the table's content for a reader \
who cannot see the image.
3. "markdown": the table converted to markdown using | column | separators | \
and a --- header separator row.

Rules for the markdown field:
- Reproduce the table content exactly as shown — do not invent or omit values.
- For cells you cannot read clearly, use "?" rather than guessing.
- If the table has multi-level headers (spanning columns), flatten them by \
repeating or concatenating parent headers into each column header.
- If the table is too complex to represent in markdown (heavily merged cells, \
embedded charts, color-coded heatmaps), set markdown to null.
- Keep cell content concise. Replace newlines within cells with spaces.

Return JSON only, no commentary."""


def extract_table_via_llm(client, cropped_png_bytes, region_id, page_num, raw_text=""):
    """LLM fallback: ask the model to convert a cropped table image to markdown.

    Returns (markdown_str, info) or (None, info) if the table is too complex.
    The info dict includes title and description for the image_descriptions output.
    """
    img_b64 = base64.b64encode(cropped_png_bytes).decode()

    text_hint = f"Convert this table to markdown ({region_id}, page {page_num}):"
    if raw_text.strip():
        text_hint += (
            f"\n\nExtracted text from this region (may have formatting issues, "
            f"use as a hint for cell values):\n{raw_text.strip()}"
        )

    resp = llm_call(client, f"table p{page_num} {region_id}", model=MODEL, messages=[
        {"role": "system", "content": TABLE_LLM_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": text_hint},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{img_b64}", "detail": "high",
            }},
        ]},
    ], response_format={"type": "json_object"})

    raw = resp.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, {"method": "llm", "reason": "invalid JSON response"}

    title = data.get("title", "")
    description = data.get("description", "")
    md = data.get("markdown")

    if not md or not isinstance(md, str) or not md.strip().startswith("|"):
        return None, {
            "method": "llm", "reason": "model reported COMPLEX or no markdown",
            "title": title, "description": description,
        }

    # Basic sanity check: must have at least a header + separator + one data row
    lines = [l for l in md.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return None, {
            "method": "llm", "reason": f"too few lines: {len(lines)}",
            "title": title, "description": description,
        }

    return md.strip(), {
        "method": "llm", "accepted": True, "lines": len(lines),
        "title": title, "description": description,
    }


# ---------------------------------------------------------------------------
# LLM Call 2: Assembly
# ---------------------------------------------------------------------------

ASSEMBLY_SYSTEM_PROMPT = """\
You are a document-to-markdown converter. You receive:
1. An annotated page image with labeled bounding boxes showing detected regions
2. The extracted text for each text region, and the filename for each image region
3. A suggested reading order from the detection step

Your task:
- Review and confirm (or adjust) the reading order based on the page layout.
- For each text region, decide the appropriate markdown formatting:
  - Normal paragraphs for body text
  - Headings (##, ###) for section titles
  - Blockquotes (>) for callout boxes or sidebars
  - Bullet lists if the text contains list items
- For each image region, place an image reference at the correct position: \
![description](filename)
  Use a brief alt text based on what you can see in the annotated page image.
- IMPORTANT: The extracted text may have missing spaces, garbled words, or OCR artifacts \
from PDF extraction. Use the page image as ground truth to fix these — restore proper \
word spacing, correct mangled words, and fix punctuation. The image shows what the text \
actually says.
- IMPORTANT: Some text regions may already contain a markdown table (lines starting \
with |). Preserve these tables exactly as provided — do not reformat, reorder columns, \
or change cell content. Just place them at the correct position in reading order.
- IMPORTANT: Only include text that actually appears on the page. Do NOT invent or \
add headings, labels, or section titles that are not present in the extracted text. \
If the page continues a section from a previous page, just continue the text without \
adding a heading.
- Output ONLY the final markdown for this page. No commentary or explanation."""


def assemble_markdown(client, annotated_png_bytes, regions_with_content, reading_order, page_num):
    """LLM call 2: assemble final markdown from extracted content."""
    annotated_b64 = base64.b64encode(annotated_png_bytes).decode()

    # Build content blocks: annotated image first, then region details
    user_parts = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{annotated_b64}", "detail": "high"},
        },
        {
            "type": "text",
            "text": f"Page {page_num}. Suggested reading order: {reading_order}\n\nRegion contents:",
        },
    ]

    for region in regions_with_content:
        rid = region["id"]
        rtype = region["type"]

        if rtype == "text":
            user_parts.append({
                "type": "text",
                "text": f"\n--- Region {rid} (text) ---\n{region['content']}",
            })
        elif rtype == "image":
            user_parts.append({
                "type": "text",
                "text": f"\n--- Region {rid} (image, file: {region['filename']}) ---",
            })

    resp = llm_call(client, f"assemble p{page_num}", model=MODEL, messages=[
        {"role": "system", "content": ASSEMBLY_SYSTEM_PROMPT},
        {"role": "user", "content": user_parts},
    ])

    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# LLM Call 3: Image description (post-pass)
# ---------------------------------------------------------------------------

DESCRIBE_SYSTEM_PROMPT = """\
You are given the full text of an academic paper and one image extracted from it. \
Provide two things:

1. A short title (max 20 words) summarizing what the image shows.
2. A longer description so a reader without access to the image can understand \
what it conveys. Use the paper text as context to interpret the image accurately \
— reference specific data, labels, and findings shown.

Keep the description factual and compact: 2-5 sentences for simple figures, \
up to a short paragraph for complex charts or tables. Do not editorialize or \
repeat the caption verbatim — add interpretive value beyond what the caption says.

Return JSON only:
{"title": "Short title here", "description": "Longer description here"}"""


def describe_image(client, image_path, paper_text):
    """Generate a title and description for one image using the full paper as context."""
    img_bytes = image_path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode()

    resp = llm_call(client, f"describe {image_path.name}", model=MODEL, messages=[
        {"role": "system", "content": DESCRIBE_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Paper text for context:\n\n{paper_text}"},
            {"type": "text", "text": f"Describe this image ({image_path.name}):"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{img_b64}", "detail": "high",
            }},
        ]},
    ], response_format={"type": "json_object"})

    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
        return {"title": data.get("title", ""), "description": data.get("description", raw)}
    except json.JSONDecodeError:
        return {"title": "", "description": raw}


def run_describe(client, output_dir, workers):
    """Post-pass: generate text descriptions for all extracted images."""
    md_files = sorted(output_dir.glob("page*.md"))
    if not md_files:
        log("No page markdown files found -- run extraction first.")
        return

    paper_text = "\n\n".join(f.read_text(encoding="utf-8") for f in md_files)

    image_files = sorted(
        p for p in output_dir.glob("page*_*.png")
        if "_boxes" not in p.name and "_grid" not in p.name
        and p.name != p.stem
    )
    image_files = [p for p in image_files if "img" in p.name or "table" in p.name]

    # Skip images that were successfully converted to markdown tables
    def was_converted_to_markdown(img_path):
        table_json = img_path.with_name(img_path.stem + "_table.json")
        if not table_json.exists():
            return False
        info = json.loads(table_json.read_text(encoding="utf-8"))
        return info.get("accepted") or info.get("llm_fallback", {}).get("accepted")

    skipped = [p for p in image_files if was_converted_to_markdown(p)]
    image_files = [p for p in image_files if not was_converted_to_markdown(p)]

    # Pre-populate descriptions from table extraction LLM calls
    descriptions = {}
    for img_path in skipped:
        table_json = img_path.with_name(img_path.stem + "_table.json")
        if table_json.exists():
            info = json.loads(table_json.read_text(encoding="utf-8"))
            # LLM fallback stores title/description alongside markdown
            llm_info = info.get("llm_fallback", {})
            title = llm_info.get("title", "")
            desc = llm_info.get("description", "")
            if title or desc:
                descriptions[img_path.name] = {"title": title, "description": desc}

    if skipped:
        log(f"Skipping {len(skipped)} table(s) already converted to markdown")

    if not image_files:
        if descriptions:
            desc_file = output_dir / "image_descriptions.json"
            desc_file.write_text(json.dumps(descriptions, indent=2), encoding="utf-8")
            log(f"Descriptions (from table extraction) -> {desc_file}")
        else:
            log("No extracted images found to describe.")
        return

    log(f"Describing {len(image_files)} images with paper context ({len(paper_text):,} chars)")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=25),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}", style="dim"),
        console=console,
    )
    status_line = StatusLine()
    task = progress.add_task("Describing images", total=len(image_files), status="")
    cost_line = CostLine()

    with Live(Group(cost_line, progress, status_line), console=console, refresh_per_second=10):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(describe_image, client, img, paper_text): img
                for img in image_files
            }
            for future in as_completed(futures):
                img = futures[future]
                try:
                    result = future.result()
                    descriptions[img.name] = result
                    progress.update(task, advance=1, status=img.name)
                    status_line.update(f"{img.name}: {result['title']}")
                except Exception as e:
                    progress.update(task, advance=1, status=f"ERROR: {img.name}")
                    status_line.update(f"{img.name}: ERROR -- {e}")
                    log_error(f"describe {img.name}", e)

    # Write descriptions file
    desc_file = output_dir / "image_descriptions.json"
    desc_file.write_text(json.dumps(descriptions, indent=2), encoding="utf-8")
    log(f"Descriptions -> {desc_file}")

    # Generate variant markdowns from combined.md
    combined_file = output_dir / "combined.md"
    if combined_file.exists():
        combined = combined_file.read_text(encoding="utf-8")

        text_only = combined
        accessible = combined

        for img_name, info in descriptions.items():
            desc = info["description"]
            pattern = rf"(!\[[^\]]*\]\({re.escape(img_name)}\))"
            text_only = re.sub(pattern, f"[Image: {img_name}] {desc}", text_only)
            accessible = re.sub(pattern, rf"\1\n\n> **Image description:** {desc}", accessible)

        text_only_file = output_dir / "combined_text_only.md"
        text_only_file.write_text(text_only, encoding="utf-8")
        log(f"Text-only markdown -> {text_only_file}")

        accessible_file = output_dir / "combined_accessible.md"
        accessible_file.write_text(accessible, encoding="utf-8")
        log(f"Accessible markdown -> {accessible_file}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_page(client, pdf_path, page_num, output_dir, tracker=None, tables=False):
    """Run the full pipeline for a single page."""
    page_idx = page_num - 1

    def status(msg, advance=False):
        if tracker:
            tracker["progress"].update(
                tracker["tasks"][page_num], status=msg, advance=1 if advance else 0,
            )
            tracker["status"].update(f"Page {page_num}: {msg}")

    # Step 1: Render + analyze
    status("Rendering & analyzing...")
    with fitz.open(str(pdf_path)) as doc:
        png_bytes, img_w, img_h = render_page(doc, page_idx)
        blocks = analyze_page(doc, page_idx)

    # Step 2: LLM call 1 -- detect layout
    status(f"Detecting layout ({len(blocks)} blocks)...", advance=True)
    layout = detect_layout(client, png_bytes, blocks, page_num, tables=tables)

    # Sanitize bboxes from LLM output
    for region in layout["regions"]:
        region["bbox"] = sanitize_bbox(region["bbox"], region["id"])

    layout_file = output_dir / f"page{page_num}_layout.json"
    layout_file.write_text(json.dumps({"page": page_num, **layout}, indent=2), encoding="utf-8")

    # Step 3: Draw annotated boxes
    annotated_png = draw_boxes(pdf_path, page_idx, layout)
    boxes_file = output_dir / f"page{page_num}_boxes.png"
    boxes_file.write_bytes(annotated_png)

    # Extract content for each non-skip region
    reading_order = layout.get("reading_order", [])
    non_skip = [r for r in layout["regions"] if r["type"] != "skip"]

    # Update progress total: render(1) + detect(1) + per-region + assemble(1)
    if tracker:
        tracker["progress"].update(
            tracker["tasks"][page_num], total=3 + len(non_skip),
        )

    # Extract text regions (single pdfplumber open)
    text_contents = {}
    text_regions = [(r["id"], r["bbox"]) for r in non_skip if r["type"] == "text"]
    if text_regions:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pg = pdf.pages[page_num - 1]
            w, h = pg.width, pg.height
            for rid, bbox in text_regions:
                status(f"Extracting {rid}...", advance=True)
                x0, y0, x1, y1 = bbox
                cropped = pg.crop((x0 * w, y0 * h, x1 * w, y1 * h))
                text_contents[rid] = cropped.extract_text() or ""

    # Crop image regions (single fitz open) and attempt table extraction
    image_contents = {}
    table_markdowns = {}  # rid -> markdown string (for successful table extractions)
    image_regions = [(r["id"], r["bbox"], r.get("image_kind")) for r in non_skip if r["type"] == "image"]
    if image_regions:
        with fitz.open(str(pdf_path)) as doc:
            pg = doc[page_idx]
            for rid, bbox, image_kind in image_regions:
                status(f"Cropping {rid}...", advance=True)
                image_contents[rid] = crop_image_region(pg, bbox)

        # Attempt table extraction for table-classified regions
        if tables:
            # Extract raw text for table regions (for LLM hints)
            table_raw_text = {}
            table_bboxes = [(rid, bbox) for rid, bbox, ik in image_regions if ik == "table"]
            if table_bboxes:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    pg = pdf.pages[page_num - 1]
                    w, h = pg.width, pg.height
                    for rid, bbox in table_bboxes:
                        x0, y0, x1, y1 = bbox
                        cropped = pg.crop((x0 * w, y0 * h, x1 * w, y1 * h))
                        table_raw_text[rid] = cropped.extract_text() or ""

            # Phase 1: try pdfplumber for all tables (fast, no LLM)
            pdfplumber_results = {}  # rid -> (md_table, score)
            need_llm = []  # rids that failed pdfplumber
            for rid, bbox, image_kind in image_regions:
                if image_kind == "table":
                    status(f"Extracting table {rid} (pdfplumber)...")
                    md_table, score = extract_table_as_markdown(pdf_path, page_num, bbox)
                    pdfplumber_results[rid] = (md_table, score)
                    if md_table is None:
                        need_llm.append(rid)

            # Phase 2: LLM fallback for failures (in parallel)
            if need_llm:
                status(f"LLM fallback for {len(need_llm)} table(s)...")
                with ThreadPoolExecutor(max_workers=len(need_llm)) as tpool:
                    llm_futures = {
                        tpool.submit(
                            extract_table_via_llm, client, image_contents[rid],
                            rid, page_num, table_raw_text.get(rid, ""),
                        ): rid
                        for rid in need_llm
                    }
                    for future in as_completed(llm_futures):
                        rid = llm_futures[future]
                        md_table, llm_score = future.result()
                        _, score = pdfplumber_results[rid]
                        score["llm_fallback"] = llm_score
                        pdfplumber_results[rid] = (md_table, score)

            # Record results
            for rid, bbox, image_kind in image_regions:
                if image_kind != "table":
                    continue
                md_table, score = pdfplumber_results[rid]
                score_file = output_dir / f"page{page_num}_{rid}_table.json"
                score_file.write_text(json.dumps(score, indent=2), encoding="utf-8")
                if md_table:
                    table_markdowns[rid] = md_table
                    method = "LLM" if score.get("llm_fallback", {}).get("accepted") else "pdfplumber"
                    status(f"Table {rid}: converted to markdown via {method}")
                else:
                    reason = score.get("llm_fallback", {}).get("reason", score.get("reason", "unknown"))
                    status(f"Table {rid}: kept as image ({reason})")

    # Build regions_with_content in original order
    regions_with_content = []
    for region in non_skip:
        rid = region["id"]
        if region["type"] == "text":
            regions_with_content.append({
                "id": rid, "type": "text", "content": text_contents[rid],
            })
        elif region["type"] == "image":
            cropped = image_contents[rid]
            filename = f"page{page_num}_{rid}.png"
            (output_dir / filename).write_bytes(cropped)

            if rid in table_markdowns:
                # Table successfully extracted as markdown
                regions_with_content.append({
                    "id": rid, "type": "text",
                    "content": table_markdowns[rid],
                })
            else:
                regions_with_content.append({
                    "id": rid, "type": "image", "content": cropped, "filename": filename,
                })

    # LLM call 2 -- assemble markdown
    status("Assembling markdown (LLM call)...", advance=True)
    markdown = assemble_markdown(
        client, annotated_png, regions_with_content, reading_order, page_num,
    )

    md_file = output_dir / f"page{page_num}.md"
    md_file.write_text(markdown, encoding="utf-8")

    status("Done", advance=True)
    return markdown


def main():
    parser = argparse.ArgumentParser(description="Extract PDF pages to Markdown + images")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("--pages", type=str, default=None,
                        help="Comma-separated page numbers (default: all pages)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory (default: slugified PDF name)")
    parser.add_argument("--parallel", "-j", default="5", metavar="N",
                        help="Number of parallel workers (default: 5, 'full' = one per page)")
    parser.add_argument("--env-file", type=Path, default=None,
                        help="Path to .env file with OPENAI_API_KEY (default: ~/.env)")
    parser.add_argument("--tables", action="store_true",
                        help="Attempt to extract simple tables as markdown instead of images")
    parser.add_argument("--no-describe", action="store_true",
                        help="Skip image description generation")
    parser.add_argument("--describe-only", action="store_true",
                        help="Skip extraction, only generate image descriptions from existing output")
    args = parser.parse_args()

    if not args.pdf.exists():
        log(f"Error: {args.pdf} not found")
        sys.exit(1)

    # Resolve API key: check env first, then .env file
    if not os.environ.get("OPENAI_API_KEY"):
        env_file = args.env_file or Path.home() / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        if not os.environ.get("OPENAI_API_KEY"):
            log("Error: OPENAI_API_KEY not found.")
            log("Set it in your environment, in ~/.env, or pass --env-file path/to/.env")
            sys.exit(1)

    # Get page count and determine pages
    with fitz.open(str(args.pdf)) as doc:
        page_count = len(doc)

    if args.pages:
        try:
            raw = [p.strip() for p in args.pages.split(",") if p.strip()]
            page_nums = [int(p) for p in raw]
        except ValueError:
            log("Error: --pages must be comma-separated integers (e.g., 3,6,7)")
            sys.exit(1)
        bad = [p for p in page_nums if p < 1 or p > page_count]
        if bad:
            log(f"Error: page(s) out of range 1..{page_count}: {bad}")
            sys.exit(1)
        seen = set()
        page_nums = [p for p in page_nums if not (p in seen or seen.add(p))]
    else:
        page_nums = list(range(1, page_count + 1))

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = find_output_dir(Path(slugify(args.pdf.name)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    global _error_log_path
    _error_log_path = args.output_dir / "errors.log"
    # Clear any previous error log
    if _error_log_path.exists():
        _error_log_path.unlink()

    t_start = time.time()

    client = OpenAI()
    if args.parallel.lower() == "full":
        workers = max(len(page_nums), 1)
    else:
        workers = min(int(args.parallel), max(len(page_nums), 1))

    log(f"{args.pdf} -> {args.output_dir}/")
    extras = []
    if args.tables:
        extras.append("tables")
    if not args.no_describe:
        extras.append("describe")
    opts = f", Options: {'+'.join(extras)}" if extras else ""
    log(f"Model: {MODEL}, Workers: {workers}, Pages: {len(page_nums)}{opts}")
    log("")

    if not args.describe_only:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TextColumn("{task.fields[status]}", style="dim"),
            console=console,
        )
        status_line = StatusLine()

        tasks = {}
        for pn in page_nums:
            tasks[pn] = progress.add_task(f"Page {pn:>3}", total=4, status="Waiting...")

        tracker = {"progress": progress, "tasks": tasks, "status": status_line}
        cost_line = CostLine()

        all_markdown = {}
        with Live(Group(cost_line, progress, status_line), console=console, refresh_per_second=10):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(process_page, client, args.pdf, pn, args.output_dir, tracker, tables=args.tables): pn
                    for pn in page_nums
                }
                errors = []
                for future in as_completed(futures):
                    pn = futures[future]
                    try:
                        all_markdown[pn] = future.result()
                    except Exception as e:
                        errors.append((pn, e))
                        progress.update(tasks[pn], status="ERROR")
                        status_line.update(f"Page {pn}: ERROR -- {e}")
                        log_error(f"Page {pn}", e)

        # Write combined markdown (in page order)
        if len(page_nums) > 1:
            combined = "\n\n---\n\n".join(all_markdown[pn] for pn in page_nums if pn in all_markdown)
            combined_file = args.output_dir / "combined.md"
            combined_file.write_text(combined, encoding="utf-8")
            log(f"Combined markdown -> {combined_file}")
        log("")

    if args.describe_only or not args.no_describe:
        run_describe(client, args.output_dir, workers)
        log("")

    # Output summary
    n_pages = len(list(args.output_dir.glob("page*.md")))
    desc_file = args.output_dir / "image_descriptions.json"
    if desc_file.exists():
        descs = json.loads(desc_file.read_text(encoding="utf-8"))
        log(f"{n_pages} pages, {len(descs)} images extracted:")
        for img_name, info in sorted(descs.items()):
            title = info.get("title", "") if isinstance(info, dict) else ""
            log(f"  {img_name:<28s} {title}")
    else:
        image_count = len([
            p for p in args.output_dir.glob("page*_*.png")
            if "_boxes" not in p.name and ("img" in p.name or "table" in p.name)
        ])
        if image_count:
            log(f"{n_pages} pages, {image_count} images extracted")
        else:
            log(f"{n_pages} pages extracted")
    log("")

    # Table extraction summary
    table_jsons = sorted(args.output_dir.glob("page*_*_table.json"))
    if table_jsons:
        converted = []
        kept_as_image = []
        for tj in table_jsons:
            info = json.loads(tj.read_text(encoding="utf-8"))
            name = tj.stem.replace("_table", "")  # e.g. "page3_img_1"
            if info.get("accepted") or info.get("llm_fallback", {}).get("accepted"):
                method = "pdfplumber" if info.get("accepted") else "LLM"
                converted.append((name, method))
            else:
                reason = info.get("llm_fallback", {}).get("reason", info.get("reason", "unknown"))
                kept_as_image.append((name, reason))
        if converted:
            log(f"Tables converted to markdown ({len(converted)}):")
            for name, method in converted:
                log(f"  {name:<28s} via {method}")
        if kept_as_image:
            log(f"Tables kept as images ({len(kept_as_image)}):")
            for name, reason in kept_as_image:
                log(f"  {name:<28s} {reason}")
        log("")

    output_files = [
        ("combined.md", "Full markdown with image references"),
        ("combined_text_only.md", "Images replaced with text descriptions"),
        ("combined_accessible.md", "Images kept with descriptions added"),
    ]
    found = [(f, desc) for f, desc in output_files if (args.output_dir / f).exists()]
    if found:
        log(f"Output files in {args.output_dir}/:")
        for f, desc in found:
            log(f"  {f:<28s} {desc}")
        log("")

    elapsed = time.time() - t_start
    minutes, seconds = divmod(int(elapsed), 60)
    time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    print_usage_summary()
    log(f"  Time: {time_str}")

    if _error_log_path and _error_log_path.exists():
        log(f"\n  Errors were logged to: {_error_log_path}")

    log("")
    log("If you run into a paper where this doesn't work, send it to")
    log("Christian Bird (cbird@microsoft.com) so we can fix it.")
    log("")
    log("Done.")


if __name__ == "__main__":
    main()
