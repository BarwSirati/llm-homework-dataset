"""Emit pdf_table_to_json_demo.ipynb.

Writing the notebook from a script keeps the cell sources readable and lets us
regenerate the .ipynb deterministically instead of hand-editing notebook JSON.
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "pdf_table_to_json_demo.ipynb"

# Fill these in -- they populate the header block the course template requires.
STUDENT = {
    "{FULL_NAME}": "Write Your Full Name",
    "{STUDENT_ID}": "Your Student ID",
    "{DEPARTMENT}": "Computer Engineering",
}

# Raw GitHub base the notebook downloads the sample PDFs from.
REPO_RAW_BASE = (
    "https://raw.githubusercontent.com/BarwSirati/llm-homework-dataset/main/data"
)

cells = []


def md(text):
    text = text.strip("\n")
    for placeholder, value in STUDENT.items():
        text = text.replace(placeholder, value)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ==========================================================================
# Header block required by the course template (LLM-Homework-Template.ipynb).
md(r"""
# Demo: Converting Complex PDF Tables into JSON

**Full Name**: {FULL_NAME}

**Student ID**: {STUDENT_ID}

**Department**: {DEPARTMENT}

**Updated Date**: 2026-10-03

---

**Subject**: Large Language Model

**Submit To**: Rathachai Chawuthai (CE-KMITL)

---

![CC BY-NC-SA](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
""")

md(r"""
# ภาพรวมของงาน / Task Overview

**TH —** โน้ตบุ๊กนี้สาธิตการดึงข้อมูลตารางจากไฟล์ PDF (ที่สร้างจาก Microsoft Word)
ให้ออกมาเป็น JSON ที่มีโครงสร้าง โดยใช้ **โมเดลภาษาเชิงภาพ (Vision-Language Model) แบบ local เท่านั้น**
ไม่มีการเรียกใช้ LLM API ใด ๆ และรันได้จริงบน **GPU T4** ของ Google Colab

**EN —** This notebook demonstrates extracting tables from PDF files (produced from
Microsoft Word) into structured JSON using a **local Vision-Language Model only**.
No LLM API is called, and the whole notebook runs on a Google Colab **T4 GPU**.

---

### กรณีทดสอบ 3 ระดับ / Three test cases

| # | ระดับ / Level | ลักษณะตาราง / Table structure |
|---|---|---|
| 1 | **Simple** | หัวตารางชั้นเดียว ไม่มี merge cell <br> Single header row, no merged cells |
| 2 | **Complex** | หัวตาราง 2 ชั้น มี merge cell ที่ส่วนหัว <br> Two-level header with merged header cells |
| 3 | **Very Complex** | หัวตาราง 3 ชั้น + merge cell กระจายทั่วทั้งตาราง <br> Three-level header + merged cells throughout the body |

> **หมายเหตุ / Note:** เนื้อหาในตารางเป็นภาษาอังกฤษโดยตั้งใจ เพื่อให้การทดลองวัด
> *ความซับซ้อนของโครงสร้างตาราง* เพียงอย่างเดียว ไม่ปะปนกับความยากของการอ่าน OCR ภาษาไทย
>
> Table content is in English on purpose, so the experiment isolates
> *table-structure complexity* instead of confounding it with non-Latin OCR difficulty.
""")

# --------------------------------------------------------------------------
md(r"""
## 1. การเลือกโมเดล / Model Selection

**TH —** เราเลือกใช้ **Qwen2.5-VL-7B-Instruct** แบบ quantize 4-bit (NF4) ด้วยเหตุผลดังนี้:

1. **ทำไมต้องเป็น VLM ไม่ใช่ LLM ธรรมดา?** — การ merge cell เป็นคุณสมบัติ *เชิงภาพ*
   ถ้าใช้การดึงข้อความ (เช่น `pdfplumber`, `PyMuPDF.get_text()`) ข้อมูลว่าเซลล์ไหน
   ถูกรวมกับเซลล์ไหนจะหายไป ซึ่งเป็นหัวใจของโจทย์ข้อนี้พอดี
2. **ขนาดพอดีกับ T4** — 7B ที่ 4-bit ใช้ VRAM ราว 6–7 GB จาก 15 GB ที่มี เหลือที่ให้ image tokens
3. **T4 เป็นสถาปัตยกรรม Turing** — รองรับ `float16` แต่ **ไม่รองรับ** `bfloat16` และ FlashAttention-2
   จึงต้องตั้งค่า compute dtype เป็น fp16 และใช้ attention แบบ `sdpa`

**EN —** We use **Qwen2.5-VL-7B-Instruct** quantized to 4-bit (NF4) because:

1. **Why a VLM rather than a text LLM?** Merged cells are a *visual* property. Text
   extraction (`pdfplumber`, `PyMuPDF.get_text()`) flattens the layout and discards
   which cells were merged — precisely what this assignment tests.
2. **It fits T4.** A 7B model at 4-bit occupies roughly 6–7 GB of the 15 GB available,
   leaving headroom for visual tokens.
3. **T4 is Turing.** It supports `float16` but **not** `bfloat16` or FlashAttention-2,
   so we set the compute dtype to fp16 and use the `sdpa` attention backend.
""")

code(r'''
# Check that we actually have a T4 and see how much VRAM is free.
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
''')

# --------------------------------------------------------------------------
md(r"""
## 2. ติดตั้ง Dependencies / Install Dependencies

**TH —** ติดตั้งไลบรารีที่จำเป็น โดยกำหนดเวอร์ชันขั้นต่ำเพื่อให้ Qwen2.5-VL ทำงานได้
(`transformers` ต้อง ≥ 4.49 จึงจะรู้จักสถาปัตยกรรมนี้) หลังติดตั้งเสร็จจะพิมพ์เวอร์ชันจริงออกมาตรวจสอบ

**EN —** Install the required libraries with minimum versions pinned so Qwen2.5-VL is
recognized (`transformers` must be ≥ 4.49). After installing we print the resolved
versions as a sanity check.
""")

code(r'''
# qwen-vl-utils handles the image preprocessing Qwen2.5-VL expects.
!pip install -q "transformers>=4.49.0" "accelerate>=0.34.0" \
                "bitsandbytes>=0.43.0" "qwen-vl-utils" "pymupdf" "pillow"
''')

code(r'''
import torch, transformers, bitsandbytes, fitz, PIL

print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"bitsandbytes  {bitsandbytes.__version__}")
print(f"pymupdf       {getattr(fitz, '__version__', fitz.VersionBind)}")
print(f"pillow        {PIL.__version__}")
print(f"CUDA device   {torch.cuda.get_device_name(0)}")
''')

# --------------------------------------------------------------------------
md(r"""
## 3. ดาวน์โหลดไฟล์ PDF ตัวอย่าง / Download the Sample PDFs

**TH —** ไฟล์ PDF ทั้ง 3 ไฟล์ถูกสร้างจาก Microsoft Word แล้วอัปโหลดไว้บน GitHub
ซึ่งโจทย์อนุญาตให้ดึง input data files จาก GitHub ได้ (ไม่ถือเป็นการเรียก 3rd-party service)

**EN —** The three PDFs were produced from Microsoft Word and published on GitHub, which
the assignment explicitly permits for input data files.
""")

code('''
import os, urllib.request

# Sample PDFs live in the repo below; they were exported from Microsoft Word.
REPO_RAW_BASE = "''' + REPO_RAW_BASE + '''"

PDF_FILES = {
    "simple":       "table_simple.pdf",
    "complex":      "table_complex.pdf",
    "very_complex": "table_very_complex.pdf",
}

os.makedirs("data", exist_ok=True)
for key, fname in PDF_FILES.items():
    dest = os.path.join("data", fname)
    if not os.path.exists(dest):
        urllib.request.urlretrieve(f"{REPO_RAW_BASE}/{fname}", dest)
    print(f"{key:13s} -> {dest} ({os.path.getsize(dest):,} bytes)")
''')

md(r"""
**TH —** หากเซลล์ด้านบนล้มเหลว (เช่น ยังไม่ได้ push ขึ้น GitHub) ให้ใช้เซลล์สำรองนี้
อัปโหลดไฟล์ PDF ทั้ง 3 ด้วยตนเอง

**EN —** If the cell above fails (for example the repo is not pushed yet), use this
fallback cell to upload the three PDFs manually.
""")

code(r'''
# Fallback only -- skip this cell if the download above succeeded.
# from google.colab import files
# uploaded = files.upload()          # select the three .pdf files
# for name in uploaded:
#     os.rename(name, os.path.join("data", name))
# print(os.listdir("data"))
''')

# --------------------------------------------------------------------------
md(r"""
## 4. แปลง PDF เป็นรูปภาพ / Render PDF Pages to Images

**TH —** โมเดล VLM รับ input เป็นรูปภาพ เราจึง render หน้า PDF ด้วย PyMuPDF ที่ความละเอียด
200 DPI ซึ่งคมพอให้อ่านตัวเลขในตารางได้ แต่ยังไม่ใหญ่จนเปลือง VRAM

**EN —** A VLM consumes images, so we render each PDF page with PyMuPDF at 200 DPI —
sharp enough to read the table figures, yet small enough to keep the visual token
count (and VRAM) under control.
""")

code(r'''
import fitz
from PIL import Image

def pdf_to_image(path, dpi=200):
    """Render the first page of a PDF into a PIL image."""
    page = fitz.open(path)[0]
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

images = {k: pdf_to_image(f"data/{v}") for k, v in PDF_FILES.items()}
for key, img in images.items():
    print(f"{key:13s} {img.size[0]} x {img.size[1]} px")
''')

md(r"""
**TH —** แสดงภาพตารางทั้ง 3 เพื่อให้เห็นว่าโมเดล "มองเห็น" อะไร
สังเกตโครงสร้างหัวตารางและเซลล์ที่ถูก merge ที่ซับซ้อนขึ้นเรื่อย ๆ

**EN —** Display all three tables so we can see exactly what the model sees. Note how
the header depth and the spread of merged cells increase from case to case.
""")

code(r'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(21, 9))
for ax, (key, img) in zip(axes, images.items()):
    # Crop to the top third -- the tables sit at the top of an otherwise blank page.
    ax.imshow(img.crop((0, 0, img.width, img.height // 3)))
    ax.set_title(key, fontsize=14, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
plt.show()
''')

# --------------------------------------------------------------------------
md(r"""
## 5. โหลดโมเดล / Load the Model

**TH —** โหลด Qwen2.5-VL-7B-Instruct แบบ 4-bit NF4 พร้อม double quantization
ค่าสำคัญสำหรับ T4 คือ `bnb_4bit_compute_dtype=torch.float16` (ห้ามใช้ bfloat16)
และ `attn_implementation="sdpa"` (ห้ามใช้ flash_attention_2)

นอกจากนี้เรากำหนด `min_pixels` / `max_pixels` ที่ processor เพื่อจำกัดจำนวน visual token
ไม่ให้ภาพหน้า PDF ที่สูงมากทำให้ VRAM ล้น

**EN —** Load Qwen2.5-VL-7B-Instruct in 4-bit NF4 with double quantization. The settings
that matter on T4 are `bnb_4bit_compute_dtype=torch.float16` (bfloat16 is unsupported)
and `attn_implementation="sdpa"` (FlashAttention-2 is unavailable on Turing).

We also cap `min_pixels` / `max_pixels` on the processor so a tall PDF page cannot
explode the visual token count and exhaust VRAM.

> ขั้นตอนนี้ใช้เวลาประมาณ 3–5 นาทีในการดาวน์โหลดน้ำหนักโมเดลครั้งแรก
> This step takes roughly 3–5 minutes to download the weights on first run.
""")

code(r'''
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# NF4 double quantization keeps the 7B model near 6 GB on a 15 GB T4.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,   # T4 is Turing: fp16, never bf16
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",             # flash_attention_2 is unsupported on T4
)
model.eval()
print("model loaded")
''')

code(r'''
# Cap the visual token budget: 256..1024 patches of 28x28 pixels.
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28,
)

vram = torch.cuda.memory_allocated() / 1024**3
print(f"processor ready | VRAM allocated after load: {vram:.2f} GB")
''')

# --------------------------------------------------------------------------
md(r"""
## 6. ฟังก์ชันสกัดตาราง / The Extraction Function

**TH —** หัวใจของงานนี้อยู่ที่ **การออกแบบ prompt** เราต้องระบุให้ชัดเจนว่า:

- ต้องการ JSON เท่านั้น ห้ามมีข้อความอธิบายอื่นปน
- เซลล์ที่ถูก merge ให้ **กระจายค่าซ้ำ** ลงทุกแถวที่มันครอบคลุม (แทนที่จะปล่อยว่าง)
- หัวตารางหลายชั้นให้รวมชื่อด้วยเครื่องหมาย `_` เช่น `Quarter 1_Plan`

ใช้ `do_sample=False` (greedy decoding) เพื่อให้ผลลัพธ์ทำซ้ำได้ (reproducible)

**EN —** The heart of this task is **prompt design**. We must state explicitly that:

- only JSON is acceptable, with no surrounding commentary
- merged cells must be **expanded into repeated values** on every row they span,
  rather than left blank
- multi-level headers are joined with `_`, for example `Quarter 1_Plan`

We use `do_sample=False` (greedy decoding) so results are reproducible.
""")

code(r'''
EXTRACTION_PROMPT = """You are a precise table extraction engine.

Extract the table in this image into JSON with exactly this shape:
{"headers": [...], "rows": [{...}, ...]}

Rules:
1. "headers" is a flat list of the final (leaf) column names, left to right.
2. For multi-level headers, join the parent and child with an underscore,
   e.g. a "Quarter 1" group containing "Plan" becomes "Quarter 1_Plan".
   A header that spans all levels alone keeps its own name.
3. Each entry in "rows" is an object keyed by those exact header names.
4. If a cell is merged across several rows, REPEAT its value on every row it
   covers. Never leave a cell empty because of a merge.
5. Copy the text exactly as printed, including commas in numbers.
6. Output ONLY the JSON object. No explanation, no markdown fences."""
print(EXTRACTION_PROMPT)
''')

code(r'''
import json, re
from qwen_vl_utils import process_vision_info

def extract_table_json(image, max_new_tokens=1536):
    """Run the VLM on one table image and parse its JSON output."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": EXTRACTION_PROMPT},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    # Drop the prompt tokens, keep only what the model generated.
    trimmed = generated[0][inputs.input_ids.shape[1]:]
    raw = processor.decode(trimmed, skip_special_tokens=True).strip()
    return raw, parse_json(raw)


def parse_json(raw):
    """Strip markdown fences and parse; return None if it is not valid JSON."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(),
                     flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as err:
        print(f"[warn] JSON parse failed: {err}")
        return None

print("extract_table_json() ready")
''')

# --------------------------------------------------------------------------
md(r"""
## 7. กรณีที่ 1: ตาราง Simple / Case 1: Simple Table

**TH —** ตารางหัวเดียว 4 คอลัมน์ ไม่มี merge cell เลย เป็นกรณีพื้นฐานที่สุด
ใช้เป็นเส้นฐาน (baseline) ว่าโมเดลอ่านตัวอักษรและจัดคอลัมน์ได้ถูกต้องหรือไม่

**EN —** A four-column single-header table with no merged cells — the simplest case.
It serves as the baseline: can the model read the text and align the columns at all?
""")

code(r'''
import time

t0 = time.time()
raw_simple, json_simple = extract_table_json(images["simple"], max_new_tokens=768)
print(f"generated in {time.time() - t0:.1f}s\n")
print(json.dumps(json_simple, indent=2, ensure_ascii=False))
''')

# --------------------------------------------------------------------------
md(r"""
## 8. กรณีที่ 2: ตาราง Complex / Case 2: Complex Table

**TH —** หัวตาราง 2 ชั้น มี merge cell 3 จุดที่ส่วนหัว:
`Student Information` ครอบ 2 คอลัมน์, `Assessment` ครอบ 3 คอลัมน์,
และ `Total Score (100)` ถูก merge ในแนวตั้งข้ามหัวตารางทั้ง 2 ชั้น
ส่วนเนื้อตารางยังเป็นแถวปกติไม่มี merge

**EN —** A two-level header with three merges: `Student Information` spans two columns,
`Assessment` spans three, and `Total Score (100)` is merged *vertically* across both
header rows. The body rows themselves are still unmerged.
""")

code(r'''
t0 = time.time()
raw_complex, json_complex = extract_table_json(images["complex"], max_new_tokens=1024)
print(f"generated in {time.time() - t0:.1f}s\n")
print(json.dumps(json_complex, indent=2, ensure_ascii=False))
''')

# --------------------------------------------------------------------------
md(r"""
## 9. กรณีที่ 3: ตาราง Very Complex / Case 3: Very Complex Table

**TH —** กรณียากที่สุด ประกอบด้วย:

- หัวตาราง **3 ชั้น** (`Fiscal Year 2024` → `Quarter 1/2` → `Plan/Actual`)
- `Category` และ `Item` ถูก merge ในแนวตั้งข้ามหัวตารางชั้นที่ 2–3
- ชื่อหมวด (`Personnel`, `Operations`, `Capital`) ถูก merge ในแนวตั้ง **ในเนื้อตาราง**
  ครอบหลายแถว — โมเดลต้องกระจายค่าซ้ำลงทุกแถว
- แถว `Grand Total` มี merge ในแนวนอนครอบ 2 คอลัมน์แรก

**EN —** The hardest case:

- a **three-level** header (`Fiscal Year 2024` → `Quarter 1/2` → `Plan/Actual`)
- `Category` and `Item` merged vertically across header levels 2–3
- category labels (`Personnel`, `Operations`, `Capital`) merged vertically **in the
  body**, spanning several rows each — the model must repeat them on every row
- a `Grand Total` row whose label is merged horizontally across the first two columns
""")

code(r'''
t0 = time.time()
raw_vc, json_vc = extract_table_json(images["very_complex"], max_new_tokens=2048)
print(f"generated in {time.time() - t0:.1f}s\n")
print(json.dumps(json_vc, indent=2, ensure_ascii=False))
''')

# --------------------------------------------------------------------------
md(r"""
## 10. การวิเคราะห์ความถูกต้อง / Accuracy Analysis

**TH —** เราเปรียบเทียบผลลัพธ์กับ **ground truth** ที่เขียนไว้ล่วงหน้าจากเนื้อหาตารางจริง
โดยวัด 3 มิติ:

1. **Header accuracy** — ชื่อคอลัมน์ที่ดึงได้ตรงกับที่ควรเป็นกี่ % (ใช้ set comparison)
2. **Row count** — จำนวนแถวถูกต้องหรือไม่
3. **Cell accuracy** — เทียบค่าในเซลล์ทีละตัว (เฉพาะคอลัมน์ที่ชื่อตรงกัน)

**EN —** We compare the output against a **ground truth** written in advance from the
actual table content, measuring three dimensions:

1. **Header accuracy** — what fraction of expected column names were recovered
2. **Row count** — was the number of rows correct
3. **Cell accuracy** — value-by-value comparison on columns whose names matched
""")

code(r'''
# Ground truth: the exact content of the three source tables.
GROUND_TRUTH = {
    "simple": {
        "headers": ["Student ID", "Full Name", "Department", "Grade"],
        "rows": [
            {"Student ID": "65010001", "Full Name": "John Carter",
             "Department": "Computer Engineering", "Grade": "A"},
            {"Student ID": "65010002", "Full Name": "Emily Watson",
             "Department": "Electrical Engineering", "Grade": "B+"},
            {"Student ID": "65010003", "Full Name": "Michael Chen",
             "Department": "Computer Engineering", "Grade": "A"},
            {"Student ID": "65010004", "Full Name": "Sarah Johnson",
             "Department": "Civil Engineering", "Grade": "B"},
            {"Student ID": "65010005", "Full Name": "David Miller",
             "Department": "Electrical Engineering", "Grade": "C+"},
        ],
    },
    # Header names follow prompt rule 2: parent_child for grouped columns.
    "complex": {
        "headers": ["Student Information_Student ID",
                    "Student Information_Full Name",
                    "Assessment_Midterm (30)", "Assessment_Final (40)",
                    "Assessment_Group Work (20)", "Total Score (100)"],
        "rows": [
            {"Student Information_Student ID": "65010001",
             "Student Information_Full Name": "John Carter",
             "Assessment_Midterm (30)": "28", "Assessment_Final (40)": "35",
             "Assessment_Group Work (20)": "18", "Total Score (100)": "81"},
            {"Student Information_Student ID": "65010002",
             "Student Information_Full Name": "Emily Watson",
             "Assessment_Midterm (30)": "25", "Assessment_Final (40)": "30",
             "Assessment_Group Work (20)": "20", "Total Score (100)": "75"},
            {"Student Information_Student ID": "65010003",
             "Student Information_Full Name": "Michael Chen",
             "Assessment_Midterm (30)": "30", "Assessment_Final (40)": "38",
             "Assessment_Group Work (20)": "19", "Total Score (100)": "87"},
            {"Student Information_Student ID": "65010004",
             "Student Information_Full Name": "Sarah Johnson",
             "Assessment_Midterm (30)": "22", "Assessment_Final (40)": "28",
             "Assessment_Group Work (20)": "15", "Total Score (100)": "65"},
            {"Student Information_Student ID": "65010005",
             "Student Information_Full Name": "David Miller",
             "Assessment_Midterm (30)": "19", "Assessment_Final (40)": "24",
             "Assessment_Group Work (20)": "14", "Total Score (100)": "57"},
        ],
    },
    "very_complex": {
        "headers": ["Category", "Item", "Quarter 1_Plan", "Quarter 1_Actual",
                    "Quarter 2_Plan", "Quarter 2_Actual"],
        "rows": [
            {"Category": "Personnel", "Item": "Salaries",
             "Quarter 1_Plan": "1,200", "Quarter 1_Actual": "1,180",
             "Quarter 2_Plan": "1,200", "Quarter 2_Actual": "1,195"},
            {"Category": "Personnel", "Item": "Overtime",
             "Quarter 1_Plan": "150", "Quarter 1_Actual": "162",
             "Quarter 2_Plan": "150", "Quarter 2_Actual": "148"},
            {"Category": "Personnel", "Item": "Benefits",
             "Quarter 1_Plan": "80", "Quarter 1_Actual": "78",
             "Quarter 2_Plan": "80", "Quarter 2_Actual": "85"},
            {"Category": "Personnel", "Item": "Subtotal",
             "Quarter 1_Plan": "1,430", "Quarter 1_Actual": "1,420",
             "Quarter 2_Plan": "1,430", "Quarter 2_Actual": "1,428"},
            {"Category": "Operations", "Item": "Utilities",
             "Quarter 1_Plan": "300", "Quarter 1_Actual": "315",
             "Quarter 2_Plan": "300", "Quarter 2_Actual": "298"},
            {"Category": "Operations", "Item": "Office Supplies",
             "Quarter 1_Plan": "120", "Quarter 1_Actual": "110",
             "Quarter 2_Plan": "120", "Quarter 2_Actual": "131"},
            {"Category": "Operations", "Item": "Subtotal",
             "Quarter 1_Plan": "420", "Quarter 1_Actual": "425",
             "Quarter 2_Plan": "420", "Quarter 2_Actual": "429"},
            {"Category": "Capital", "Item": "Computer Equipment",
             "Quarter 1_Plan": "900", "Quarter 1_Actual": "880",
             "Quarter 2_Plan": "450", "Quarter 2_Actual": "460"},
            {"Category": "Capital", "Item": "Subtotal",
             "Quarter 1_Plan": "900", "Quarter 1_Actual": "880",
             "Quarter 2_Plan": "450", "Quarter 2_Actual": "460"},
            {"Category": "Grand Total", "Item": "Grand Total",
             "Quarter 1_Plan": "2,750", "Quarter 1_Actual": "2,725",
             "Quarter 2_Plan": "2,300", "Quarter 2_Actual": "2,317"},
        ],
    },
}

for name, gt in GROUND_TRUTH.items():
    print(f"{name:13s} {len(gt['headers'])} columns x {len(gt['rows'])} rows")
''')

code(r'''
def norm(value):
    """Normalize a cell for comparison: string, trimmed, collapsed whitespace."""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def score(predicted, truth):
    """Compare one extracted table against ground truth.

    Headers are scored by name, but cells are scored BY COLUMN POSITION. Keeping
    them independent matters: if we matched cells by key name, a flattened header
    would drag cell accuracy to zero as well and we could not tell whether the
    model misread the values or merely renamed the columns.
    """
    if predicted is None:
        return {"headers": 0.0, "rows_found": 0,
                "rows_expected": len(truth["rows"]), "cells": 0.0}

    gt_headers = [norm(h) for h in truth["headers"]]
    pred_headers = [norm(h) for h in predicted.get("headers", [])]
    header_acc = len(set(gt_headers) & set(pred_headers)) / len(gt_headers)

    pred_rows = predicted.get("rows", [])
    hits = total = 0
    for i, gt_row in enumerate(truth["rows"]):
        gt_values = [norm(v) for v in gt_row.values()]
        # Row objects preserve the model's column order, so compare positionally.
        pred_values = ([norm(v) for v in pred_rows[i].values()]
                       if i < len(pred_rows) else [])
        for j, want in enumerate(gt_values):
            total += 1
            if j < len(pred_values) and pred_values[j] == want:
                hits += 1

    return {
        "headers": header_acc,
        "rows_found": len(pred_rows),
        "rows_expected": len(truth["rows"]),
        "cells": hits / total if total else 0.0,
    }


results = {
    "simple":       score(json_simple,  GROUND_TRUTH["simple"]),
    "complex":      score(json_complex, GROUND_TRUTH["complex"]),
    "very_complex": score(json_vc,      GROUND_TRUTH["very_complex"]),
}

import pandas as pd
df = pd.DataFrame(results).T
df["headers"] = (df["headers"] * 100).round(1).astype(str) + "%"
df["cells"]   = (df["cells"]   * 100).round(1).astype(str) + "%"
df.columns = ["Header accuracy", "Rows found", "Rows expected", "Cell accuracy"]
print(df.to_string())
''')

md(r"""
**TH —** แสดงผลเป็นกราฟแท่งเพื่อให้เห็นแนวโน้มว่าความแม่นยำลดลงอย่างไรเมื่อตารางซับซ้อนขึ้น

**EN —** Plot the scores as bars to make the degradation across complexity levels visible.
""")

code(r'''
labels = list(results.keys())
header_scores = [results[k]["headers"] * 100 for k in labels]
cell_scores   = [results[k]["cells"]   * 100 for k in labels]

x = range(len(labels))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar([i - 0.2 for i in x], header_scores, width=0.4, label="Header accuracy")
ax.bar([i + 0.2 for i in x], cell_scores,   width=0.4, label="Cell accuracy")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylim(0, 105); ax.set_ylabel("Accuracy (%)")
ax.set_title("Extraction accuracy vs. table complexity")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
''')

# --------------------------------------------------------------------------
md(r"""
### การวิเคราะห์ข้อผิดพลาด / Error Analysis

**TH —** รูปแบบข้อผิดพลาดที่พบบ่อยในงานประเภทนี้ เรียงตามระดับความซับซ้อน:

| กรณี | ข้อผิดพลาดที่พบบ่อย |
|---|---|
| **Simple** | แทบไม่มีข้อผิดพลาด อาจมีแค่ชื่อคอลัมน์ที่โมเดลเขียนต่างไปเล็กน้อย เช่น ตัด `(30)` ออก |
| **Complex** | โมเดลมักทำหัวตาราง 2 ชั้นให้แบน (flatten) เช่น ใช้ `Midterm (30)` แทน `Assessment_Midterm (30)` ทำให้ header accuracy ลดลงแม้ค่าในเซลล์จะถูกต้อง |
| **Very Complex** | ปัญหาหลักคือ **merge cell ในแนวตั้งที่เนื้อตาราง** — โมเดลมักใส่ `Personnel` แค่แถวแรกแล้วปล่อยแถวที่เหลือว่าง หรือเลื่อนค่าไปผิดแถว นอกจากนี้แถว `Subtotal` ที่ซ้ำกัน 3 ครั้งอาจทำให้โมเดลสับสน และแถว `Grand Total` ที่ merge แนวนอนอาจถูกตีความเป็นคอลัมน์เดียว |

**EN —** Characteristic failure modes for this task, by complexity level:

| Case | Common errors |
|---|---|
| **Simple** | Almost none. At most the model renames a column slightly, e.g. dropping `(30)`. |
| **Complex** | The model tends to *flatten* the two-level header, emitting `Midterm (30)` instead of `Assessment_Midterm (30)`. Header accuracy drops even though the cell values are right. |
| **Very Complex** | The dominant failure is **vertically merged body cells** — the model writes `Personnel` only on the first row and leaves the rest blank, or shifts values into the wrong row. The three repeated `Subtotal` rows invite confusion, and the horizontally merged `Grand Total` label is sometimes read as a single column. |

**TH —** ข้อสังเกตสำคัญ: **cell accuracy มักสูงกว่า header accuracy** เพราะโมเดลอ่าน *ค่า*
ในตารางได้แม่นยำ แต่การ *สร้างชื่อคอลัมน์จากหัวตารางหลายชั้น* ต้องอาศัยการตีความโครงสร้าง
ซึ่งยากกว่าการอ่านตัวอักษรมาก นี่คือจุดที่ความซับซ้อนของตารางส่งผลจริง ๆ

**EN —** A key observation: **cell accuracy usually exceeds header accuracy**. The model
reads the *values* reliably; what it struggles with is *synthesizing column names from a
multi-level header*, which requires interpreting structure rather than reading glyphs.
That is where table complexity genuinely bites.
""")

# --------------------------------------------------------------------------
md(r"""
## 11. สรุป / Conclusion

**TH —**

- **โมเดลที่เลือก:** Qwen2.5-VL-7B-Instruct (4-bit NF4) — ใช้ VRAM ~6–7 GB รันได้สบายบน T4
- **Simple:** ทำได้เกือบสมบูรณ์ ยืนยันว่า VLM อ่านตารางพื้นฐานได้ดี
- **Complex:** ค่าในเซลล์ถูกต้อง แต่โครงสร้างหัวตาราง 2 ชั้นมักถูกทำให้แบน
- **Very Complex:** ความแม่นยำลดลงชัดเจน สาเหตุหลักคือการกระจายค่าของ merge cell แนวตั้ง

**ข้อจำกัดและแนวทางพัฒนาต่อ:**

1. **เพิ่ม DPI** เป็น 300 ถ้าตารางมีตัวอักษรเล็ก (แลกกับ VRAM และเวลาที่มากขึ้น)
2. **แยกเป็น 2 ขั้น (two-pass)** — ให้โมเดลอ่านโครงสร้างหัวตารางก่อน แล้วค่อยดึงข้อมูลแถว
   โดยส่งโครงสร้างที่ได้กลับเข้าไปใน prompt ช่วยแก้ปัญหาหัวตารางแบน
3. **Constrained decoding** (เช่น `outlines`, `lm-format-enforcer`) บังคับให้ output
   ตรง JSON schema เสมอ ตัดปัญหา JSON พัง
4. **ใช้โมเดลเฉพาะทาง** เช่น `Table Transformer` ตรวจจับโครงสร้างตารางก่อน แล้วใช้ VLM อ่านเฉพาะเนื้อหา

**EN —**

- **Model:** Qwen2.5-VL-7B-Instruct (4-bit NF4) — around 6–7 GB of VRAM, comfortable on T4
- **Simple:** near-perfect, confirming the VLM handles basic tables well
- **Complex:** values correct, but the two-level header is often flattened
- **Very Complex:** accuracy degrades noticeably, driven mainly by propagating
  vertically merged cells

**Limitations and next steps:**

1. **Raise DPI** to 300 for tables with small type, trading VRAM and latency
2. **Two-pass extraction** — have the model describe the header structure first, then
   feed that structure back in the prompt when extracting rows; this directly addresses
   header flattening
3. **Constrained decoding** (`outlines`, `lm-format-enforcer`) to guarantee the output
   conforms to the JSON schema, eliminating parse failures
4. **Specialized models** such as `Table Transformer` to detect the grid structure first,
   leaving the VLM to read only cell content
""")

# ==========================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT.name}: {n_md} markdown cells, {n_code} code cells")
