# utils/pdf_utils.py
"""
import io
import base64
import requests
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes

GEMINI_OCR_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

def gemini_ocr_image(image_pil, api_key):
    #Send a PIL image to Gemini OCR API using inline base64 encoding (same as your curl).
    # Convert PIL to bytes
    img_bytes = io.BytesIO()
    image_pil.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Base64 encode
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64
                        }
                    },
                    {"text": "Extract the text from this legal document page as accurately as possible."}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    response = requests.post(GEMINI_OCR_URL, json=payload, headers=headers, timeout=60)

    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def extract_text_from_pdfs(file_list, gemini_api_key=None):
    combined = ""

    for uploaded_file in file_list:
        pdf_bytes = uploaded_file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()

            # If extracted text is empty or useless → OCR fallback
            if not text or len(text.strip()) < 10:
                print(f"⚠️ Page {page_number+1}: No text found → using Gemini OCR")
                
                # Convert only this page to image
                images = convert_from_bytes(
                    pdf_bytes, 
                    first_page=page_number+1, 
                    last_page=page_number+1
                )

                ocr_text = gemini_ocr_image(images[0], gemini_api_key)

                if not ocr_text.strip():
                    combined += f"[OCR failed on page {page_number+1}]\n"
                else:
                    combined += ocr_text + "\n"
            else:
                combined += text + "\n"

    return combined
"""
import io
import base64
import requests
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from docx import Document as DocxDocument

GEMINI_OCR_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


# -------------------------
# GEMINI OCR (for images)
# -------------------------
def gemini_ocr_image(image_pil, api_key):
    """OCR using Gemini with inline base64 encoding."""
    img_bytes = io.BytesIO()
    image_pil.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                { "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64
                }},
                {"text": "Extract all text from this legal document page."}
            ]
        }]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    response = requests.post(GEMINI_OCR_URL, json=payload, headers=headers, timeout=60)

    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return ""


# -------------------------
# PDF + OCR extraction
# -------------------------
def extract_pdf_text(uploaded_file, gemini_api_key=None):
    combined = ""
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        # If text missing → OCR scanned page
        if not text or len(text.strip()) < 10:
            images = convert_from_bytes(
                pdf_bytes,
                first_page=page_num + 1,
                last_page=page_num + 1
            )
            ocr_text = gemini_ocr_image(images[0], gemini_api_key)
            combined += ocr_text + "\n"
        else:
            combined += text + "\n"

    return combined


# -------------------------
# DOCX extraction
# -------------------------
def extract_docx_text(uploaded_file):
    doc = DocxDocument(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])


# -------------------------
# TXT extraction
# -------------------------
def extract_txt_text(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


# -------------------------
# Master function
# -------------------------
def extract_text_from_documents(file_list, gemini_api_key=None):
    full_text = ""

    for f in file_list:
        filename = f.name.lower()

        if filename.endswith(".pdf"):
            full_text += extract_pdf_text(f, gemini_api_key) + "\n"

        elif filename.endswith(".docx"):
            full_text += extract_docx_text(f) + "\n"

        elif filename.endswith(".txt"):
            full_text += extract_txt_text(f) + "\n"

    return full_text

