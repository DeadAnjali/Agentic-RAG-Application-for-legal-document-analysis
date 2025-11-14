# agents/pdf_agent.py
from utils.vectorstore_utils import chunk_texts, build_faiss_from_texts
from utils.pdf_utils import extract_text_from_documents

def build_document_vectorstore(uploaded_pdf_files,gemini_api_key):
    # extract combined text
    raw_text = extract_text_from_documents(uploaded_pdf_files,gemini_api_key)
    if not raw_text or not raw_text.strip():
        raise ValueError("No text extracted from provided PDFs.")
    chunks = chunk_texts([raw_text])
    vs = build_faiss_from_texts(chunks)
    return vs
