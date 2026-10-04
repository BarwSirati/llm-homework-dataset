# Demo: Converting Complex PDF Tables into JSON

Extracting tables of increasing structural complexity from Word-generated PDFs into
structured JSON, using a **local model only** (no LLM APIs), running on a Google Colab
**T4 GPU**.

**Model:** `Qwen/Qwen2.5-VL-7B-Instruct` quantized to 4-bit NF4 (~6–7 GB VRAM).

The notebook evaluates two pipelines against the same ground truth:

- **A — VLM on a rendered image.** The general-purpose route: works on any PDF,
  including scans, because it reads pixels.
- **B — PDF geometry + LLM.** `find_tables()` supplies exact text and exact merge
  extents; the LLM decides how many header rows there are. Far more accurate, but only
  works when the PDF carries a text layer (the notebook demonstrates it failing on a
  scanned page).

The comparison is what justifies the model choice, which is the point of the assignment.

## Test cases

| # | Level | Structure |
|---|---|---|
| 1 | Simple | Single header row, no merged cells |
| 2 | Complex | Two-level header with merged header cells |
| 3 | Very Complex | Three-level header + merged cells throughout the body |

## Files

| File | Purpose |
|---|---|
| `make_docx.py` | Builds the three tables as Word `.docx` files |
| `convert_to_pdf.py` | Exports them to PDF by driving Microsoft Word (macOS, AppleScript) |
| `ground_truth.py` | Expected JSON for each table, used for scoring |
| `build_notebook.py` | Generates `pdf_table_to_json_demo.ipynb` |
| `pdf_table_to_json_demo.ipynb` | **The deliverable** — run this on Colab with a T4 |
| `data/*.pdf` | Input PDFs the notebook downloads |

## Regenerating the inputs

```bash
python3 make_docx.py        # .docx -> data/
python3 convert_to_pdf.py   # .pdf  -> data/  (requires Microsoft Word on macOS)
python3 build_notebook.py   # rebuild the notebook
```

## Running the demo

1. Push this repo to GitHub so the PDFs are reachable over `raw.githubusercontent.com`.
2. Open `pdf_table_to_json_demo.ipynb` in Google Colab (as a copy of the course template).
3. Set the runtime to **GPU / T4**.
4. Edit `REPO_RAW_BASE` in section 3 to point at your repository.
5. Run all cells.

## Notes on T4

T4 is Turing architecture, so the notebook uses `float16` compute (not `bfloat16`) and
the `sdpa` attention backend (FlashAttention-2 is unsupported). Visual tokens are capped
via the processor's `min_pixels` / `max_pixels` so a tall page cannot exhaust VRAM.
