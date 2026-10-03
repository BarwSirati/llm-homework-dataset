"""Generate three Word documents containing tables of increasing complexity.

The resulting .docx files are converted to PDF by convert_to_pdf.py, producing
the "PDF created from Word" inputs the demo notebook consumes.

Table content is in English so the demo isolates the variable under study --
table structure (merged cells, multi-level headers) -- rather than mixing in
non-Latin OCR difficulty.
"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

DATA_DIR = Path(__file__).parent / "data"
FONT = "Calibri"


def _style_cell(cell, text, bold=False, align_center=False):
    cell.text = ""
    para = cell.paragraphs[0]
    if align_center:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(10)
    run.font.name = FONT


def _new_table(doc, rows, cols):
    table = doc.add_table(rows=rows, cols=cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    return table


def _add_title(doc, text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.bold = True
    run.font.size = Pt(14)
    run.font.name = FONT


# --------------------------------------------------------------------------
# Case 1: Simple -- single header row, no merged cells
# --------------------------------------------------------------------------
SIMPLE_HEADERS = ["Student ID", "Full Name", "Department", "Grade"]
SIMPLE_ROWS = [
    ["65010001", "John Carter", "Computer Engineering", "A"],
    ["65010002", "Emily Watson", "Electrical Engineering", "B+"],
    ["65010003", "Michael Chen", "Computer Engineering", "A"],
    ["65010004", "Sarah Johnson", "Civil Engineering", "B"],
    ["65010005", "David Miller", "Electrical Engineering", "C+"],
]


def build_simple():
    doc = Document()
    _add_title(doc, "Table 1: Student Grade Report (Simple)")

    table = _new_table(doc, rows=1 + len(SIMPLE_ROWS), cols=4)
    for col, name in enumerate(SIMPLE_HEADERS):
        _style_cell(table.rows[0].cells[col], name, bold=True, align_center=True)
    for r, row in enumerate(SIMPLE_ROWS, start=1):
        for c, value in enumerate(row):
            _style_cell(table.rows[r].cells[c], value, align_center=(c == 3))

    out = DATA_DIR / "table_simple.docx"
    doc.save(out)
    return out


# --------------------------------------------------------------------------
# Case 2: Complex -- two-level header with merged header cells
# --------------------------------------------------------------------------
COMPLEX_ROWS = [
    ["65010001", "John Carter", "28", "35", "18", "81"],
    ["65010002", "Emily Watson", "25", "30", "20", "75"],
    ["65010003", "Michael Chen", "30", "38", "19", "87"],
    ["65010004", "Sarah Johnson", "22", "28", "15", "65"],
    ["65010005", "David Miller", "19", "24", "14", "57"],
]


def build_complex():
    doc = Document()
    _add_title(doc, "Table 2: Course Score Summary (Complex)")

    # 6 columns: id, name | midterm, final, groupwork | total
    table = _new_table(doc, rows=2 + len(COMPLEX_ROWS), cols=6)
    top, sub = table.rows[0], table.rows[1]

    # "Student Information" spans the two identity columns
    student = top.cells[0].merge(top.cells[1])
    _style_cell(student, "Student Information", bold=True, align_center=True)
    # "Assessment" spans the three score columns
    assessment = top.cells[2].merge(top.cells[4])
    _style_cell(assessment, "Assessment", bold=True, align_center=True)
    # "Total Score" is vertically merged across both header rows
    total = top.cells[5].merge(sub.cells[5])
    _style_cell(total, "Total Score (100)", bold=True, align_center=True)

    for col, name in enumerate(
        ["Student ID", "Full Name", "Midterm (30)", "Final (40)", "Group Work (20)"]
    ):
        _style_cell(sub.cells[col], name, bold=True, align_center=True)

    for r, row in enumerate(COMPLEX_ROWS, start=2):
        for c, value in enumerate(row):
            _style_cell(table.rows[r].cells[c], value, align_center=(c >= 2))

    out = DATA_DIR / "table_complex.docx"
    doc.save(out)
    return out


# --------------------------------------------------------------------------
# Case 3: Very complex -- multi-level header plus merges across the body
# --------------------------------------------------------------------------
# (category, [(item, q1_plan, q1_actual, q2_plan, q2_actual), ...], subtotal)
VERY_COMPLEX_BODY = [
    (
        "Personnel",
        [
            ("Salaries", "1,200", "1,180", "1,200", "1,195"),
            ("Overtime", "150", "162", "150", "148"),
            ("Benefits", "80", "78", "80", "85"),
        ],
        ("1,430", "1,420", "1,430", "1,428"),
    ),
    (
        "Operations",
        [
            ("Utilities", "300", "315", "300", "298"),
            ("Office Supplies", "120", "110", "120", "131"),
        ],
        ("420", "425", "420", "429"),
    ),
    (
        "Capital",
        [
            ("Computer Equipment", "900", "880", "450", "460"),
        ],
        ("900", "880", "450", "460"),
    ),
]
GRAND_TOTAL = ("2,750", "2,725", "2,300", "2,317")


def build_very_complex():
    doc = Document()
    _add_title(doc, "Table 3: Quarterly Budget Report (Very Complex)")

    # 6 columns: category, item, Q1 plan, Q1 actual, Q2 plan, Q2 actual
    body_rows = sum(len(items) + 1 for _, items, _ in VERY_COMPLEX_BODY)
    table = _new_table(doc, rows=3 + body_rows + 1, cols=6)
    h0, h1, h2 = table.rows[0], table.rows[1], table.rows[2]

    # Header row 0: merged title on the left, merged fiscal year on the right
    rubric = h0.cells[0].merge(h0.cells[1])
    _style_cell(rubric, "Budget Category / Item", bold=True, align_center=True)
    year = h0.cells[2].merge(h0.cells[5])
    _style_cell(year, "Fiscal Year 2024 (unit: THB thousand)",
                bold=True, align_center=True)

    # Header row 1: labels merged down rows 1-2, quarters merged across
    category = h1.cells[0].merge(h2.cells[0])
    _style_cell(category, "Category", bold=True, align_center=True)
    item = h1.cells[1].merge(h2.cells[1])
    _style_cell(item, "Item", bold=True, align_center=True)
    q1 = h1.cells[2].merge(h1.cells[3])
    _style_cell(q1, "Quarter 1", bold=True, align_center=True)
    q2 = h1.cells[4].merge(h1.cells[5])
    _style_cell(q2, "Quarter 2", bold=True, align_center=True)

    # Header row 2: plan/actual under each quarter
    for col, name in zip(range(2, 6), ["Plan", "Actual", "Plan", "Actual"]):
        _style_cell(h2.cells[col], name, bold=True, align_center=True)

    # Body: vertically merge each category label across its items + subtotal row
    r = 3
    for cat_name, items, subtotal in VERY_COMPLEX_BODY:
        block_start = r
        for item_name, *values in items:
            _style_cell(table.rows[r].cells[1], item_name)
            for c, value in enumerate(values, start=2):
                _style_cell(table.rows[r].cells[c], value, align_center=True)
            r += 1

        _style_cell(table.rows[r].cells[1], "Subtotal", bold=True, align_center=True)
        for c, value in enumerate(subtotal, start=2):
            _style_cell(table.rows[r].cells[c], value, bold=True, align_center=True)
        block_end = r
        r += 1

        merged_cat = table.rows[block_start].cells[0].merge(
            table.rows[block_end].cells[0]
        )
        _style_cell(merged_cat, cat_name, bold=True, align_center=True)

    # Grand total: label spans the two left columns
    last = table.rows[r]
    grand_label = last.cells[0].merge(last.cells[1])
    _style_cell(grand_label, "Grand Total", bold=True, align_center=True)
    for c, value in enumerate(GRAND_TOTAL, start=2):
        _style_cell(last.cells[c], value, bold=True, align_center=True)

    out = DATA_DIR / "table_very_complex.docx"
    doc.save(out)
    return out


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for build in (build_simple, build_complex, build_very_complex):
        path = build()
        print(f"wrote data/{path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
