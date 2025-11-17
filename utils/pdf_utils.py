# utils/pdf_utils.py
import io
import base64
import requests
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from docx import Document as DocxDocument
import fitz  # PyMuPDF
from PIL import Image
import io

GEMINI_OCR_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

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
                {"text": "Extract all text from this legal document page. Preserve proper spacing between words, sentences, and paragraphs. Maintain the document structure and formatting."}
            ]
        }]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    response = requests.post(GEMINI_OCR_URL, json=payload, headers=headers, timeout=60)
    print("Gemini OCR raw response:", response.text)
    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return ""


def fix_text_spacing(text):
    """
    Fix common spacing issues in extracted text.
    Adds spaces between concatenated words.
    """
    import re
    
    # Add space between lowercase and uppercase letters (e.g., "wordAnother" -> "word Another")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Add space between letter and number (e.g., "month25" -> "month 25")
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    
    # Add space between number and letter (e.g., "25per" -> "25 per")
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    
    # Add space after periods if followed by uppercase (sentence boundaries)
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()



def extract_pdf_text(uploaded_file, gemini_api_key=None):
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    combined = ""

    # Open PDF with PyMuPDF (handles malformed PDFs)
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return f"[Failed to open PDF: {e}]"

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Extract text normally first
        text = page.get_text("text")  # plain text extraction

        # If no text → perform OCR
        if not text or len(text.strip()) < 10:
            if gemini_api_key:
                try:
                    # Render page as an image
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")

                    image = Image.open(io.BytesIO(img_bytes))

                    # OCR via Gemini
                    ocr_text = gemini_ocr_image(image, gemini_api_key)
                    ocr_text = fix_text_spacing(ocr_text)
                    combined += ocr_text + "\n"

                except Exception as e:
                    print(f"OCR failed for page {page_num+1}: {e}")
                    combined += f"[OCR failed on page {page_num+1}]\n"

            else:
                combined += f"[No text on page {page_num+1}]\n"

        else:
            # Extracted text OK
            text = fix_text_spacing(text)
            combined += text + "\n"

    return combined


def extract_docx_text(uploaded_file):
    """
    Extract text from DOCX file.
    IMPORTANT: Resets file pointer to beginning before reading.
    """
    # Reset file pointer to beginning
    uploaded_file.seek(0)
    doc = DocxDocument(uploaded_file)
    
    # Reset again for potential reuse
    uploaded_file.seek(0)
    
    return "\n".join([para.text for para in doc.paragraphs])



def extract_txt_text(uploaded_file):
    """
    Extract text from TXT file.
    IMPORTANT: Resets file pointer to beginning before reading.
    """
    # Reset file pointer to beginning
    uploaded_file.seek(0)
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    
    # Reset again for potential reuse
    uploaded_file.seek(0)
    
    return text



def extract_text_from_documents(file_list, gemini_api_key=None):
    """
    Extract text from a list of uploaded files (PDF, DOCX, TXT).
    Handles file pointer reset to allow multiple reads.
    """
    full_text = ""

    for f in file_list:
        filename = f.name.lower()

        try:
            if filename.endswith(".pdf"):
                full_text += extract_pdf_text(f, gemini_api_key) + "\n"

            elif filename.endswith(".docx"):
                full_text += extract_docx_text(f) + "\n"

            elif filename.endswith(".txt"):
                full_text += extract_txt_text(f) + "\n"
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            full_text += f"[Error processing {filename}]\n"

    return full_text