# utils/pdf_utils.py
from PyPDF2 import PdfReader

def extract_text_from_pdfs(file_list):
    """
    file_list: list of Streamlit UploadedFile or file-like objects
    returns: concatenated text
    """
    combined = ""
    for f in file_list:
        reader = PdfReader(f)
        for p in reader.pages:
            text = p.extract_text()
            if text:
                combined += text + "\n"
    return combined
