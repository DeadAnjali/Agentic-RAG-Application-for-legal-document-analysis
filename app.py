# app.py
import streamlit as st
from dotenv import load_dotenv
import os
import warnings

# initialize local modules
from agents.gemini_client import GeminiClient
from agents.pdf_agent import build_pdf_vectorstore
from agents.indiacode_agent import build_indiacode_vectorstore, load_indiacode_json
from agents.retrieval_agent import RetrievalAgent
from agents.summarizer_agent import SummarizerAgent
from agents.reasoning_agent import ReasoningAgent
from utils.vectorstore_utils import chunk_texts

from htmlTemplates import css, bot_template, user_template

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# ensure GEMINI_API_KEY present (UI will still run but Gemini calls will fail without a key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def main():
    st.set_page_config(page_title="Agentic Indian Legal RAG", page_icon=":scales:")
    st.write(css, unsafe_allow_html=True)
    st.title("Agentic RAG — Indian Legal Assistant")

    # Session initialization
    if "pdf_vectorstore" not in st.session_state:
        st.session_state.pdf_vectorstore = None
    if "corpus_vectorstore" not in st.session_state:
        st.session_state.corpus_vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = GeminiClient(api_key=GEMINI_API_KEY)

    # SIDEBAR: corpus loading & pdf upload
    with st.sidebar:
        st.subheader("Knowledge sources")
        st.markdown("**1) Load IndiaCode JSON corpus**")
        json_path = st.text_input("IndiaCode JSON path", value="data/indiacode_data.json")
        if st.button("Load IndiaCode corpus"):
            with st.spinner("Indexing IndiaCode corpus..."):
                try:
                    vs = build_indiacode_vectorstore(json_path)
                    st.session_state.corpus_vectorstore = vs
                    st.success("IndiaCode corpus indexed and ready.")
                except Exception as e:
                    st.error(f"Failed to load IndiaCode corpus: {e}")

        st.markdown("---")
        st.markdown("**2) Upload PDFs**")
        uploaded_pdfs = st.file_uploader("Upload PDFs (judgments, acts)", accept_multiple_files=True)
        if st.button("Process PDFs") and uploaded_pdfs:
            with st.spinner("Processing and indexing PDFs..."):
                try:
                    vs = build_pdf_vectorstore(uploaded_pdfs)
                    st.session_state.pdf_vectorstore = vs
                    st.success("PDFs processed and indexed.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")

        st.markdown("---")
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()

    # MAIN: Query input
    st.subheader("Ask a legal question about your documents or IndiaCode")
    user_q = st.text_input("Type your question and press Enter")

    if user_q:
        with st.spinner("Retrieving and reasoning..."):
            retrieval = RetrievalAgent(
                pdf_vectorstore=st.session_state.get("pdf_vectorstore"),
                corpus_vectorstore=st.session_state.get("corpus_vectorstore"),
                top_k=6
            )
            summarizer = SummarizerAgent(st.session_state.llm_client)
            reasoner = ReasoningAgent(st.session_state.llm_client, retrieval, summarizer)

            out = reasoner.run(user_q)

            # format and store
            formatted = f"Plan:\n{out['plan']}\n\nAnswer:\n{out['answer']}\n\nContext summary:\n{out['summary']}"
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            st.session_state.chat_history.append({"role": "bot", "content": formatted})

    # Render chat history
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            st.write(user_template.replace("{{MSG}}", m["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", m["content"]), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
