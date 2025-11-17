import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Test 1: API Key
api_key = os.getenv("GEMINI_API_KEY")
print(f"1. API Key loaded: {bool(api_key)}")
if api_key:
    print(f"   First 10 chars: {api_key[:10]}...")

# Test 2: Network connectivity
try:
    r = requests.get("https://generativelanguage.googleapis.com", timeout=5)
    print(f"2. Network: OK")
except Exception as e:
    print(f"2. Network: FAILED - {e}")

# Test 3: API call
model = "gemini-2.5-flash"
endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

headers = {
    "x-goog-api-key": api_key,
    "Content-Type": "application/json"
}

payload = {
    "contents": [{"parts": [{"text": "Say hello"}]}]
}

try:
    response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    print(f"3. API Call Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"   Response: {text}")
        print("   ✅ API WORKING!")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"3. API Call: FAILED - {e}")


#--------------------------------------------------------------
def extract_pdf_text(uploaded_file, gemini_api_key=None):
    combined = ""
    
    # CRITICAL: Reset file pointer to beginning
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    
    # Reset again for potential reuse
    uploaded_file.seek(0)
    
    reader = PdfReader(io.BytesIO(pdf_bytes))

    for page_num, page in enumerate(reader.pages):
        # Try extraction with layout mode for better spacing
        try:
            text = page.extract_text(extraction_mode="layout")
        except:
            text = page.extract_text()  # Fallback to default mode

        # If text missing → OCR scanned page
        if not text or len(text.strip()) < 10:
            if gemini_api_key:
                try:
                    images = convert_from_bytes(
                        pdf_bytes,
                        first_page=page_num + 1,
                        last_page=page_num + 1,
                        poppler_path=r"C:\Users\hp\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"

                    )
                    ocr_text = gemini_ocr_image(images[0], gemini_api_key)
                    # Fix spacing in OCR text
                    ocr_text = fix_text_spacing(ocr_text)
                    combined += ocr_text + "\n"
                except Exception as e:
                    print(f"OCR failed for page {page_num + 1}: {e}")
                    combined += f"[OCR failed on page {page_num + 1}]\n"
            else:
                combined += f"[No text on page {page_num + 1}]\n"
        else:
            # Fix spacing in extracted text
            text = fix_text_spacing(text)
            combined += text + "\n"

    return combined