# agents/pdf_agent.py
from utils.vectorstore_utils import chunk_texts, build_faiss_from_texts
from utils.pdf_utils import extract_text_from_pdfs

def build_pdf_vectorstore(uploaded_pdf_files):
    """
    uploaded_pdf_files: list of file-like objects (Streamlit UploadFile)
    returns: FAISS vectorstore (LangChain FAISS object)
    """
    # extract combined text
    raw_text = extract_text_from_pdfs(uploaded_pdf_files)
    if not raw_text or not raw_text.strip():
        raise ValueError("No text extracted from provided PDFs.")
    chunks = chunk_texts([raw_text])
    vs = build_faiss_from_texts(chunks)
    return vs
