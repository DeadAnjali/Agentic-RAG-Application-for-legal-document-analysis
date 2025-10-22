# utils/vectorstore_utils.py
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

def chunk_texts(texts, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    out_chunks = []
    for t in texts:
        if not t or not str(t).strip():
            continue
        out_chunks.extend(splitter.split_text(t))
    return out_chunks

def build_faiss_from_texts(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Build FAISS index from a list of text chunks.
    """
    emb = SentenceTransformerEmbeddings(model_name=model_name)
    vs = FAISS.from_texts(texts=chunks, embedding=emb)
    return vs
