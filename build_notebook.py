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
import torch, transformers, bitsandbytes, pymupdf, PIL

print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"bitsandbytes  {bitsandbytes.__version__}")
print(f"pymupdf       {pymupdf.__version__}")
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

**TH —** โมเดล VLM รับ input เป็นรูปภาพ ขั้นตอนนี้มีผลต่อความแม่นยำ **มากที่สุด**
เราทำ 2 อย่างเพื่อเพิ่มความแม่นยำ:

1. **Render ที่ 300 DPI** (แทน 200) — ตัวเลขในตารางคมขึ้น ลดการอ่านผิด เช่น `1,180` vs `1,100`
2. **Auto-crop ขอบขาวทิ้ง** — หน้า A4 มีตารางอยู่แค่ด้านบนราว 25% ที่เหลือเป็นกระดาษเปล่า
   ถ้าส่งทั้งหน้าเข้าโมเดล **visual token กว่า 70% จะถูกใช้ไปกับพื้นที่ว่าง**
   และตารางจะถูกย่อจนเล็กเกินอ่าน การ crop ทำให้ token ทั้งหมดตกอยู่ที่ตัวตารางจริง ๆ

**EN —** A VLM consumes images, and this step affects accuracy **more than any other**.
Two things matter here:

1. **Render at 300 DPI** (up from 200) so digits stay sharp — this is what separates
   `1,180` from `1,100` in the model's reading.
2. **Auto-crop the whitespace.** The table occupies only the top ~25% of an A4 page.
   Feeding the whole page spends **over 70% of the visual token budget on blank paper**,
   and the resizing that follows shrinks the table below legibility. Cropping puts the
   entire token budget on the table itself.
""")

code(r'''
try:
    import pymupdf as fitz          # PyMuPDF >= 1.24 renamed the module
except ImportError:
    import fitz
from PIL import Image, ImageOps

def pdf_to_image(path, dpi=300, pad=12):
    """Render page 1 of a PDF and crop it tight to the inked region.

    Cropping matters more than it looks: the table sits in the top quarter of an
    otherwise blank A4 page, so without this the model spends most of its visual
    tokens looking at white paper.
    """
    page = fitz.open(path)[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # getbbox() finds non-zero pixels, so invert first: white paper -> black.
    bbox = ImageOps.invert(img.convert("L")).getbbox()
    if bbox:
        left, top, right, bottom = bbox
        img = img.crop((max(left - pad, 0), max(top - pad, 0),
                        min(right + pad, img.width),
                        min(bottom + pad, img.height)))
    return img

images = {k: pdf_to_image(f"data/{v}") for k, v in PDF_FILES.items()}
for key, img in images.items():
    w, h = img.size
    print(f"{key:13s} {w} x {h} px  ({w * h / 1e6:.2f} MP after crop)")
''')

md(r"""
**TH —** แสดงภาพตารางทั้ง 3 เพื่อให้เห็นว่าโมเดล "มองเห็น" อะไร
สังเกตโครงสร้างหัวตารางและเซลล์ที่ถูก merge ที่ซับซ้อนขึ้นเรื่อย ๆ

**EN —** Display all three tables so we can see exactly what the model sees. Note how
the header depth and the spread of merged cells increase from case to case.
""")

code(r'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 13))
for ax, (key, img) in zip(axes, images.items()):
    ax.imshow(img)
    ax.set_title(key, fontsize=13, fontweight="bold")
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

อีกค่าที่ต้องตั้งให้พอดีคือ `max_pixels` ซึ่งเป็นงบ visual token
**ข้อจำกัดจริงอยู่ที่ vision encoder ไม่ใช่ตัว LLM** — Qwen2.5-VL แบ่งภาพเป็น patch 14×14
แล้วรวม 2×2 patch เป็น 1 visual token ดังนั้น sequence ภายใน ViT จึงยาวเป็น **4 เท่า** ของงบที่ตั้ง
และต้นทุน attention โตตาม **กำลังสอง** ของความยาวนั้น

เราทดสอบแล้วพบว่าตั้ง 2560 patches ทำให้ ViT ขอ memory ก้อนเดียว ~3.9 GB และ T4 OOM จริง
จึงใช้ 1280 patches ซึ่งคุมเทอมนี้ไว้ต่ำกว่า ~1 GB และยังให้ความละเอียดสูงกว่าเดิมมาก
เพราะเรา crop ขอบขาวทิ้งไปแล้ว

**EN —** Load Qwen2.5-VL-7B-Instruct in 4-bit NF4 with double quantization. The settings
that matter on T4 are `bnb_4bit_compute_dtype=torch.float16` (bfloat16 is unsupported)
and `attn_implementation="sdpa"` (FlashAttention-2 is unavailable on Turing).

The other number to get right is `max_pixels`, the visual token budget. **The binding
constraint is the vision encoder, not the language model**: Qwen2.5-VL splits the image
into 14×14 patches and merges 2×2 of them per visual token, so the ViT's internal
sequence is **4×** the budget and its attention cost grows with the **square** of that.

We measured this: a budget of 2560 patches makes the ViT request a single ~3.9 GB
allocation and the T4 genuinely runs out of memory. 1280 keeps that term under ~1 GB
while still giving far more resolution than before, because the whitespace is already
cropped away.

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
# Visual token budget, in patches of 28x28 px.
#
# This number is bounded by the VISION ENCODER, not by the language model. Qwen2.5-VL
# splits the image into 14x14 patches and merges 2x2 of them into one visual token, so
# the ViT's internal sequence is 4x this budget and its attention cost grows with the
# SQUARE of that. At 2560 the ViT tries to allocate ~3.9 GB for one attention matrix
# and a T4 runs out of memory; 1280 keeps that term under ~1 GB.
MAX_PATCHES = 1280

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=MAX_PATCHES * 28 * 28,
)

vram = torch.cuda.memory_allocated() / 1024**3
print(f"processor ready | VRAM allocated after load: {vram:.2f} GB")
print(f"visual token budget: {MAX_PATCHES} patches "
      f"(ViT sequence ~{MAX_PATCHES * 4} tokens)")
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
# The worked example uses an UNRELATED table, so it teaches the output format
# without leaking any answer for the three tables we actually evaluate on.
EXTRACTION_PROMPT = """You are a precise table extraction engine.

Extract the table in this image into JSON with exactly this shape:
{"headers": [...], "rows": [{...}, ...]}

Rules:
1. Count the table's data columns by its vertical grid lines. "headers" must have
   exactly that many entries, left to right, and every row object must have exactly
   that many fields. NEVER combine two columns into one field.
2. Name each column using AT MOST THE TWO INNERMOST header rows that apply to it,
   joined by an underscore: a "Quarter 1" group over a "Plan" column becomes
   "Quarter 1_Plan". Ignore any outer header row that merely captions the whole
   table or states units. If only one header row applies, use that name alone.
3. Each entry in "rows" is an object keyed by those exact header names, in the
   same left-to-right order.
4. If a cell is merged across several rows, REPEAT its value on every row it
   covers. Never leave a field empty or omit it because of a merge.
5. If a label is merged horizontally across the leading columns (such as a
   total row), repeat that label in each column it spans.
6. Copy text exactly as printed, including commas in numbers. Do not compute,
   reformat, or correct any value.
7. Output ONLY the JSON object. No explanation, no markdown fences.

Worked example for an unrelated 3-column table. Its top row is the caption
"Annual Report 2023 (unit: tons)"; below it "Region" stands alone while "H1"
spans "Jan" and "Feb"; the body has "North" merged down two rows.
The caption is ignored and "North" is repeated:
{"headers": ["Region", "H1_Jan", "H1_Feb"],
 "rows": [{"Region": "North", "H1_Jan": "10", "H1_Feb": "12"},
          {"Region": "North", "H1_Jan": "14", "H1_Feb": "9"}]}"""

print(EXTRACTION_PROMPT)
''')

code(r'''
import json, re, gc
from qwen_vl_utils import process_vision_info

def extract_table_json(image, max_new_tokens=1536):
    """Run the VLM on one table image and parse its JSON output."""
    # Release the previous case's activations; on a 15 GB T4 the three runs would
    # otherwise accumulate and the largest table would fail with an OOM.
    gc.collect()
    torch.cuda.empty_cache()

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
raw_vc, json_vc = extract_table_json(images["very_complex"], max_new_tokens=1536)
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
3. **Cell accuracy** — เทียบค่าในเซลล์ทีละตัว **ตามตำแหน่งคอลัมน์** (เข้มงวดที่สุด)
4. **Value recall** — ค่าที่ถูกต้องไปอยู่ใน *แถว* ที่ถูกต้องหรือไม่ โดย**ไม่สนชื่อและตำแหน่งคอลัมน์**

> ทำไมต้องมี 3 ตัว: ถ้ามีแต่ cell accuracy เราจะแยกไม่ออกว่าโมเดล "อ่านตัวเลขผิด"
> หรือแค่ "จัดคอลัมน์ต่างไป" เช่น ถ้าโมเดลรวม 2 คอลัมน์เป็นช่องเดียว ตำแหน่งจะเลื่อนหมด
> ทำให้ cell accuracy = 0% ทั้งที่อ่านตัวเลขถูกทุกตัว — **value recall จะจับกรณีนี้ได้**

**EN —** We compare the output against a **ground truth** written in advance from the
actual table content, measuring three dimensions:

1. **Header accuracy** — what fraction of expected column names were recovered
2. **Row count** — was the number of rows correct
3. **Cell accuracy** — value-by-value comparison **matched by column position** (strictest)
4. **Value recall** — did the right values reach the right *row* at all, **ignoring both
   column names and column positions**

> Why three metrics: cell accuracy alone cannot tell "the model misread the digits" from
> "the model organised the columns differently". If the model merges two columns into one
> field, every position shifts and cell accuracy collapses to 0% even when every value was
> read correctly — **value recall is what exposes that case**.
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
                "rows_expected": len(truth["rows"]), "cells": 0.0, "values": 0.0}

    gt_headers = [norm(h) for h in truth["headers"]]
    pred_headers = [norm(h) for h in predicted.get("headers", [])]
    header_acc = len(set(gt_headers) & set(pred_headers)) / len(gt_headers)

    pred_rows = predicted.get("rows", [])
    hits = total = 0
    recalled = 0
    for i, gt_row in enumerate(truth["rows"]):
        gt_values = [norm(v) for v in gt_row.values()]
        # Row objects preserve the model's column order, so compare positionally.
        pred_values = ([norm(v) for v in pred_rows[i].values()]
                       if i < len(pred_rows) else [])
        for j, want in enumerate(gt_values):
            total += 1
            if j < len(pred_values) and pred_values[j] == want:
                hits += 1

        # Value recall ignores naming AND position: did the right values land in
        # the right row at all? This separates "misread the text" from "organised
        # the columns differently", which the positional score cannot distinguish.
        pool = list(pred_values)
        for want in gt_values:
            if want in pool:
                pool.remove(want)
                recalled += 1

    return {
        "headers": header_acc,
        "rows_found": len(pred_rows),
        "rows_expected": len(truth["rows"]),
        "cells": hits / total if total else 0.0,
        "values": recalled / total if total else 0.0,
    }


results = {
    "simple":       score(json_simple,  GROUND_TRUTH["simple"]),
    "complex":      score(json_complex, GROUND_TRUTH["complex"]),
    "very_complex": score(json_vc,      GROUND_TRUTH["very_complex"]),
}

import pandas as pd
df = pd.DataFrame(results).T
for col in ("headers", "cells", "values"):
    df[col] = (df[col] * 100).round(1).astype(str) + "%"
df.columns = ["Header accuracy", "Rows found", "Rows expected",
              "Cell accuracy", "Value recall"]
print(df.to_string())
''')

md(r"""
**TH —** แสดงผลเป็นกราฟแท่งเพื่อให้เห็นแนวโน้มว่าความแม่นยำลดลงอย่างไรเมื่อตารางซับซ้อนขึ้น

**EN —** Plot the scores as bars to make the degradation across complexity levels visible.
""")

code(r'''
labels = list(results.keys())
series = [("Header accuracy", "headers"),
          ("Cell accuracy", "cells"),
          ("Value recall", "values")]

x = range(len(labels))
fig, ax = plt.subplots(figsize=(9, 4.5))
for offset, (title, key) in zip((-0.27, 0.0, 0.27), series):
    ax.bar([i + offset for i in x], [results[k][key] * 100 for k in labels],
           width=0.26, label=title)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylim(0, 105); ax.set_ylabel("Accuracy (%)")
ax.set_title("Extraction accuracy vs. table complexity")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
''')

# --------------------------------------------------------------------------
md(r"""
### การวิเคราะห์ข้อผิดพลาด / Error Analysis

**TH —** ผลที่วัดได้จริงจากการรันครั้งนี้ (ตัวเลขอาจต่างเล็กน้อยในแต่ละรอบ):

| กรณี | Rows | Cell acc. | Value recall | สิ่งที่เกิดขึ้นจริง |
|---|---|---|---|---|
| **Simple** | 5/5 | 100% | 100% | อ่านถูกทุกเซลล์ ชื่อคอลัมน์ตรงบางส่วน (โมเดลเขียนต่างไปเล็กน้อย) |
| **Complex** | 5/5 | 100% | 100% | **ค่าถูกต้องทั้งหมด** แต่ชื่อคอลัมน์ไม่ตรงรูปแบบที่กำหนด โมเดลไม่ใส่ prefix `Assessment_` |
| **Very Complex** | 10/10 | 0% | 66.7% | จำนวนแถวถูกต้อง และ **กระจายค่า merge cell แนวตั้งได้ถูกต้อง** (เช่น `Personnel_Benefits`, `Personnel_Subtotal`) แต่ **ยุบคอลัมน์ `Category` กับ `Item` รวมเป็นช่องเดียว** ทำให้ตำแหน่งเลื่อนทั้งแถว → cell accuracy = 0% ขณะที่ value recall = 66.7% (ตัวเลข 4 จาก 6 ช่องยังถูกต้อง) |

**EN —** What actually happened in this run (numbers can shift slightly between runs):

| Case | Rows | Cell acc. | Value recall | Observed behaviour |
|---|---|---|---|---|
| **Simple** | 5/5 | 100% | 100% | Every cell read correctly; column names only partly matched our convention. |
| **Complex** | 5/5 | 100% | 100% | **All values correct**, but the model did not apply the `Assessment_` prefix we asked for, so header accuracy scored 0 despite a perfect reading. |
| **Very Complex** | 10/10 | 0% | 66.7% | Row count correct, and **vertically merged cells WERE propagated** (it emitted `Personnel_Benefits`, `Personnel_Subtotal`). But it **collapsed `Category` and `Item` into a single field**, shifting every position — hence 0% cell accuracy while value recall stays at 66.7% (the four numeric columns were still right). |

**TH —** ข้อสรุปที่สำคัญกว่าตัวเลข: ความล้มเหลวของโมเดล **ไม่ใช่การอ่านตัวอักษรผิด**
แต่เป็นการ **ไม่ทำตามข้อกำหนดเชิงโครงสร้าง** — ตั้งชื่อคอลัมน์คนละแบบ และยุบ 2 คอลัมน์เป็นคอลัมน์เดียว
สังเกตว่าเราสั่งใน prompt ชัดเจนแล้วว่า "ห้ามรวม 2 คอลัมน์เป็นช่องเดียว" แต่โมเดลขนาด 7B
ก็ยังทำ นี่คือข้อจำกัดจริงของการบังคับ schema ด้วย prompt เพียงอย่างเดียว

**EN —** The conclusion that matters more than the numbers: the model's failures are **not
misread glyphs** — they are **non-compliance with structural instructions**. It names
columns its own way and fuses two columns into one. Note that the prompt explicitly
forbids merging columns, and a 7B model still does it. That is the real limitation of
enforcing a schema through prompting alone, and it is exactly what constrained decoding
(listed below) is designed to fix.
""")

# --------------------------------------------------------------------------
md(r"""
## 11. สรุป / Conclusion

**TH —**

- **โมเดลที่เลือก:** Qwen2.5-VL-7B-Instruct (4-bit NF4) — ใช้ VRAM ~6–7 GB รันได้สบายบน T4
- **Simple:** 100% ทั้ง cell accuracy และ value recall
- **Complex:** 100% เช่นกัน — merge cell ที่ *หัวตาราง* ไม่ได้ทำให้อ่านค่าผิดเลย
- **Very Complex:** อ่านค่าได้ 66.7% และกระจาย merge cell แนวตั้งถูกต้อง
  แต่ยุบ 2 คอลัมน์ซ้ายรวมกัน ทำให้โครงสร้างผิด
- **แนวโน้มที่เห็นชัด:** ความซับซ้อนของตารางกระทบ **โครงสร้างของผลลัพธ์** มากกว่า **ความถูกต้องของค่า**

**เทคนิคที่ใช้เพิ่มความแม่นยำในโน้ตบุ๊กนี้:**

1. **Auto-crop ขอบขาว** — ทำให้ visual token ทั้งหมดตกที่ตัวตาราง ไม่เสียไปกับกระดาษเปล่า
   (ผลตอบแทนสูงที่สุด)
2. **Render ที่ 300 DPI** + ตั้ง `max_pixels` ที่ 1280 patches ซึ่งเป็นจุดสูงสุดที่ T4 รับไหว
   (สูงกว่านี้ vision encoder จะ OOM)
3. **Prompt ที่ระบุกฎชัดเจน + worked example** ของตารางอื่นที่ไม่เกี่ยวข้อง
   เพื่อสอน format โดยไม่เฉลยคำตอบ
4. **Greedy decoding** (`do_sample=False`) ให้ผลลัพธ์ทำซ้ำได้

**แนวทางพัฒนาต่อ:**

1. **แยกเป็น 2 ขั้น (two-pass)** — ให้โมเดลอ่านโครงสร้างหัวตารางก่อน แล้วส่งโครงสร้างนั้น
   กลับเข้า prompt ตอนดึงข้อมูลแถว ช่วยแก้ปัญหาหัวตารางแบนโดยตรง
2. **Constrained decoding** (เช่น `outlines`, `lm-format-enforcer`) บังคับให้ output
   ตรง JSON schema เสมอ ตัดปัญหา JSON พัง
3. **ใช้โมเดลเฉพาะทาง** เช่น `Table Transformer` ตรวจจับโครงสร้างตารางก่อน แล้วใช้ VLM อ่านเฉพาะเนื้อหา

**EN —**

- **Model:** Qwen2.5-VL-7B-Instruct (4-bit NF4) — around 6–7 GB of VRAM, comfortable on T4
- **Simple:** 100% on both cell accuracy and value recall
- **Complex:** also 100% — merged cells in the *header* did not cost a single value
- **Very Complex:** 66.7% value recall with merged body cells correctly propagated, but
  the two leftmost columns were fused, breaking the structure
- **The clear trend:** table complexity degrades the **structure of the output** far more
  than the **correctness of the values**

**Accuracy techniques applied in this notebook:**

1. **Auto-cropping the whitespace**, so the whole visual token budget lands on the table
   rather than on blank paper — the highest-return change by a wide margin
2. **300 DPI rendering** with `max_pixels` at 1280 patches — empirically the ceiling a
   T4 sustains, since the vision encoder OOMs above it
3. **An explicit rule list and a worked example** on an unrelated table, teaching the
   output format without leaking any evaluated answer
4. **Greedy decoding** (`do_sample=False`) for reproducible results

**Further improvements:**

1. **Two-pass extraction** — have the model describe the header structure first, then
   feed that structure back into the prompt when extracting rows; this attacks header
   flattening directly
2. **Constrained decoding** (`outlines`, `lm-format-enforcer`) to guarantee the output
   conforms to the JSON schema, eliminating parse failures
3. **Specialized models** such as `Table Transformer` to detect the grid structure first,
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
