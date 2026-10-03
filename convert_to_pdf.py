"""Convert the generated .docx files to PDF by driving Microsoft Word.

Uses AppleScript so the PDFs are genuinely exported by Word, matching the
assignment's "PDF created from Word" requirement. Word windows open briefly
during the run and macOS may ask once for automation permission.
"""

import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
NAMES = ["table_simple", "table_complex", "table_very_complex"]

# The default AppleEvent timeout (2 min) is not enough for a cold Word launch.
APPLESCRIPT = """
tell application "Microsoft Word"
    activate
    with timeout of 600 seconds
        open POSIX file "{docx}"
        set theDoc to active document
        save as theDoc file name POSIX file "{pdf}" file format format PDF
        close theDoc saving no
    end timeout
end tell
"""


def convert(docx: Path) -> Path:
    pdf = docx.with_suffix(".pdf")
    if pdf.exists():
        pdf.unlink()  # Word refuses to overwrite silently

    script = APPLESCRIPT.format(docx=docx.resolve(), pdf=pdf.resolve())
    result = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Word refused to convert {docx.name}:\n{result.stderr.strip()}"
        )
    if not pdf.exists() or pdf.stat().st_size < 1000:
        raise RuntimeError(f"{pdf.name} missing or suspiciously small after export")
    return pdf


def main():
    missing = [n for n in NAMES if not (DATA_DIR / f"{n}.docx").exists()]
    if missing:
        sys.exit(f"missing .docx files: {missing} — run make_docx.py first")

    for name in NAMES:
        pdf = convert(DATA_DIR / f"{name}.docx")
        print(f"wrote data/{pdf.name} ({pdf.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
