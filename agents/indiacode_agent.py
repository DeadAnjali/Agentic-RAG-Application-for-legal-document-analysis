# agents/indiacode_agent.py
import os
import json
import io
import base64
import requests
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from difflib import SequenceMatcher
from utils.vectorstore_utils import chunk_texts, build_faiss_from_texts

GEMINI_OCR_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

def load_indiacode_json(path="data/indiacode_data.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"IndiaCode JSON not found at: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    docs = []
    # iterate over top-level keys (allacts, allregulations, ...) and their entries
    for top_key, entries in data.items():
        if not isinstance(entries, dict):
            continue
        for title, details in entries.items():
            metadata = details.get("metadata", {}) or {}
            pdf_links = details.get("pdfLinks", []) or []

            short_title = metadata.get("Act Short Title:", metadata.get("Short Title", "")).strip()
            act_id = metadata.get("Act ID:", metadata.get("ActID", "")).strip()
            act_number = metadata.get("Act Number:", "").strip()
            act_year = metadata.get("Act Year:", "").strip()
            long_title = metadata.get("Long Title:", "").strip()
            enactment_date = metadata.get("Enactment Date:", "").strip()
            enforcement_date = metadata.get("Enforcement Date:", "").strip()

            lines = [
                f"Source: IndiaCode ({top_key})",
                f"Title: {title}",
            ]
            if short_title:
                lines.append(f"Short Title: {short_title}")
            if act_id:
                lines.append(f"Act ID: {act_id}")
            if act_number:
                lines.append(f"Act Number: {act_number}")
            if act_year:
                lines.append(f"Act Year: {act_year}")
            if enactment_date:
                lines.append(f"Enactment Date: {enactment_date}")
            if enforcement_date:
                lines.append(f"Enforcement Date: {enforcement_date}")
            if long_title:
                lines.append(f"Long Title: {long_title}")
            if pdf_links:
                lines.append("PDF Links:")
                for p in pdf_links:
                    lines.append(p)

            text_block = "\n".join([ln for ln in lines if ln and ln.strip()])
            docs.append(text_block)
            print(text_block[:200] + "\n---")

    return docs


def build_indiacode_vectorstore(json_path="data/indiacode_data.json"):
    docs = load_indiacode_json(json_path)
    if not docs:
        raise ValueError("No IndiaCode docs found in JSON.")
    chunks = chunk_texts(docs)
    vs = build_faiss_from_texts(chunks)
    return vs


# =====================================================
# NEW FUNCTIONALITY: Match Acts and Fetch PDFs
# =====================================================

def similarity_score(str1, str2):
    """Calculate similarity between two strings (0-1 range)."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def extract_act_references(text):
    """
    Extract potential Act references from text.
    Looks for patterns like "Act", "Code", numbered sections, etc.
    Returns list of probable act mentions.
    """
    import re
    
    # Common patterns for Act references
    patterns = [
        r'\b([A-Z][A-Za-z\s,]+(?:Act|Code|Regulation|Rules?))\s*(?:,\s*)?(?:\d{4})?\b',
        r'\b(?:Section|Article|Chapter|Rule)\s+\d+[A-Za-z]?\s+of\s+(?:the\s+)?([A-Z][A-Za-z\s,]+(?:Act|Code))\b',
        r'\b([A-Z]{2,})\b'  # Acronyms like IPC, CrPC, etc.
    ]
    
    references = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        references.update(matches)
    
    return list(references)


def find_matching_acts(user_text, indiacode_json_path="data/indiacode_data.json", threshold=0.6):
    """
    Find Acts from IndiaCode that match references in the user's uploaded text.
    Returns list of matching acts with their metadata and PDF links.
    """
    if not os.path.exists(indiacode_json_path):
        print(f"Warning: IndiaCode JSON not found at {indiacode_json_path}")
        return []

    with open(indiacode_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract Act references from user text
    user_references = extract_act_references(user_text)
    print(f"Found {len(user_references)} potential Act references in uploaded document")

    matched_acts = []
    
    for top_key, entries in data.items():
        if not isinstance(entries, dict):
            continue
            
        for title, details in entries.items():
            metadata = details.get("metadata", {}) or {}
            pdf_links = details.get("pdfLinks", []) or []
            
            long_title = metadata.get("Long Title:", "").strip()
            short_title = metadata.get("Act Short Title:", metadata.get("Short Title", "")).strip()
            act_year = metadata.get("Act Year:", "").strip()
            
            # Check similarity with user references
            for ref in user_references:
                # Compare with long title
                long_sim = similarity_score(ref, long_title) if long_title else 0
                short_sim = similarity_score(ref, short_title) if short_title else 0
                title_sim = similarity_score(ref, title)
                
                max_sim = max(long_sim, short_sim, title_sim)
                
                if max_sim >= threshold:
                    matched_acts.append({
                        "title": title,
                        "short_title": short_title,
                        "long_title": long_title,
                        "act_year": act_year,
                        "pdf_links": pdf_links,
                        "similarity": max_sim,
                        "matched_reference": ref
                    })
                    print(f"✓ Matched: {ref} → {short_title or title} (similarity: {max_sim:.2f})")
                    break
    
    # Sort by similarity score (descending)
    matched_acts.sort(key=lambda x: x["similarity"], reverse=True)
    
    return matched_acts


def gemini_ocr_image(image_pil, api_key):
    """OCR using Gemini with inline base64 encoding."""
    img_bytes = io.BytesIO()
    image_pil.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {
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


def extract_text_from_pdf_url(pdf_url, gemini_api_key=None, max_pages=10):
    """
    Download PDF from URL and extract text (with OCR fallback for scanned pages).
    Limits to max_pages to avoid excessive processing time.
    """
    try:
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        pdf_bytes = response.content
        reader = PdfReader(io.BytesIO(pdf_bytes))
        
        total_pages = len(reader.pages)
        pages_to_process = min(total_pages, max_pages)
        
        print(f"Processing {pages_to_process} of {total_pages} pages...")
        
        combined_text = ""
        
        for page_num in range(pages_to_process):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            # If text missing → OCR scanned page
            if not text or len(text.strip()) < 10:
                if gemini_api_key:
                    try:
                        images = convert_from_bytes(
                            pdf_bytes,
                            first_page=page_num + 1,
                            last_page=page_num + 1
                        )
                        ocr_text = gemini_ocr_image(images[0], gemini_api_key)
                        combined_text += ocr_text + "\n"
                    except Exception as e:
                        print(f"OCR failed for page {page_num + 1}: {e}")
                        combined_text += f"[OCR failed on page {page_num + 1}]\n"
            else:
                combined_text += text + "\n"
        
        if pages_to_process < total_pages:
            combined_text += f"\n[Note: Only first {pages_to_process} pages processed out of {total_pages} total pages]\n"
        
        return combined_text
        
    except Exception as e:
        print(f"Error extracting text from PDF URL: {e}")
        return ""


def get_act_context_from_matched_pdfs(matched_acts, gemini_api_key=None, llm_client=None, max_acts=3):
    """
    For matched Acts, download their PDFs, extract text, and summarize.
    Returns summarized context to be added to the retrieval pipeline.
    """
    context_parts = []
    
    for i, act in enumerate(matched_acts[:max_acts]):  # Limit to top N matches
        print(f"\n{'='*60}")
        print(f"Processing Act {i+1}/{min(len(matched_acts), max_acts)}: {act['short_title'] or act['title']}")
        print(f"{'='*60}")
        
        pdf_links = act.get("pdf_links", [])
        
        if not pdf_links:
            print(f"No PDF links available for {act['title']}")
            continue
        
        # Try first PDF link
        pdf_url = pdf_links[0]
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf_url(pdf_url, gemini_api_key, max_pages=15)
        
        if not pdf_text or len(pdf_text.strip()) < 50:
            print(f"Failed to extract meaningful text from {act['title']}")
            continue
        
        # Summarize using LLM if available
        if llm_client:
            summary_prompt = f"""
            Summarize the following legal Act text into key points relevant for legal research.
            Focus on: main provisions, important sections, key definitions, and scope.
            Keep it concise (max 500 words).
            
            Act: {act['short_title'] or act['title']}
            Year: {act['act_year']}
            
            Text:
            {pdf_text[:8000]}  
            """
            
            try:
                summary = llm_client.generate(summary_prompt, max_output_tokens=1024)
            except Exception as e:
                print(f"Failed to generate summary: {e}")
                summary = pdf_text[:2000]  # Fallback to truncated text
        else:
            summary = pdf_text[:2000]  # No LLM available, use truncated text
        
        context_parts.append({
            "act_title": act['short_title'] or act['title'],
            "act_year": act['act_year'],
            "pdf_url": pdf_url,
            "summary": summary,
            "matched_reference": act['matched_reference']
        })
        
        print(f"✓ Successfully processed {act['short_title'] or act['title']}")
    
    return context_parts


def format_act_context(act_contexts):
    """
    Format the extracted Act contexts into a readable string for the LLM.
    """
    if not act_contexts:
        return ""
    
    formatted = "\n\n" + "="*80 + "\n"
    formatted += "ADDITIONAL CONTEXT FROM MATCHED ACTS IN INDIACODE:\n"
    formatted += "(These Acts were identified from your uploaded document)\n"
    formatted += "="*80 + "\n\n"
    
    for ctx in act_contexts:
        formatted += f"📖 Act: {ctx['act_title']}\n"
        if ctx['act_year']:
            formatted += f"   Year: {ctx['act_year']}\n"
        formatted += f"   Matched Reference: {ctx['matched_reference']}\n"
        formatted += f"   Source PDF: {ctx['pdf_url']}\n"
        formatted += f"\n   Summary:\n   {ctx['summary']}\n"
        formatted += "\n" + "-"*80 + "\n\n"
    
    return formatted