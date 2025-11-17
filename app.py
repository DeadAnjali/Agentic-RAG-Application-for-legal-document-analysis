import streamlit as st
from dotenv import load_dotenv
import os
import warnings

# initialize local modules
from agents.gemini_client import GeminiClient
from agents.pdf_agent import build_document_vectorstore
from agents.indiacode_agent import build_indiacode_vectorstore
from agents.scraper_agent import build_judgment_vectorstore
from agents.retrieval_agent import RetrievalAgent
from agents.summarizer_agent import SummarizerAgent
from agents.reasoning_agent import ReasoningAgent
from utils.pdf_utils import extract_text_from_documents
from htmlTemplates import css, bot_template, user_template

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def main():
    st.set_page_config(page_title="Agentic Indian Legal RAG", page_icon="⚖️")
    st.write(css, unsafe_allow_html=True)
    st.title("Agentic RAG – Indian Legal Assistant")

    # Session initialization
    if "pdf_vectorstore" not in st.session_state:
        st.session_state.pdf_vectorstore = None
    if "corpus_vectorstore" not in st.session_state:
        st.session_state.corpus_vectorstore = None
    if "judgments_vectorstore" not in st.session_state:
        st.session_state.judgments_vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = GeminiClient(api_key=GEMINI_API_KEY)
    if "user_document_text" not in st.session_state:
        st.session_state.user_document_text = None

    # Sidebar: knowledge management (IndiaCode + Scraper + clear chat)
    with st.sidebar:
        st.subheader("Knowledge sources")
        st.markdown("**Load IndiaCode JSON corpus**")
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
        st.markdown("**Scrape & index Supreme Court landmark judgments**")
        col1, col2 = st.columns([1,2])
        with col1:
            refresh = st.checkbox("Force refresh", value=False)
        with col2:
            if st.button("Scrape & Index Judgments"):
                with st.spinner("Scraping and indexing Supreme Court landmark judgments..."):
                    try:
                        vs = build_judgment_vectorstore(refresh=refresh)
                        st.session_state.judgments_vectorstore = vs
                        st.success("Judgments scraped and indexed.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to scrape/index judgments: {e}")

        st.markdown("---")
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()

    # Upload PDFs moved to main page
    st.markdown("## Upload and index your legal documents")
    st.markdown("*The system will automatically identify relevant Acts and fetch their full text from IndiaCode*")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs, Word files, or Text files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing and indexing documents..."):
                try:
                    # Step 1: Extract text from uploaded documents
                    st.info("Step 1/3: Extracting text from uploaded documents...")
                    user_text = extract_text_from_documents(uploaded_files, gemini_api_key=GEMINI_API_KEY)
                    st.session_state.user_document_text = user_text
                    
                    # Step 2: Build vectorstore from documents
                    st.info("Step 2/3: Building vector index from documents...")
                    vs = build_document_vectorstore(uploaded_files, gemini_api_key=GEMINI_API_KEY)
                    st.session_state.pdf_vectorstore = vs
                    
                    # Step 3: Match Acts and fetch PDFs (this will happen on first query)
                    st.info("Step 3/3: Document processing complete. Acts will be matched on first query...")
                    
                    st.success("✓ Documents processed and indexed successfully!")
                    st.success("✓ Acts mentioned in your document will be automatically identified and their full text fetched from IndiaCode when you ask questions.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to process documents: {e}")
                    import traceback
                    st.error(traceback.format_exc())

    # Main Q&A input section
    st.markdown("---")
    st.subheader("Ask a legal question about your documents, IndiaCode or judgments")
    user_q = st.text_input("Type your question and press Enter")

    if user_q:
        with st.spinner("Retrieving and reasoning..."):
            try:
                # Create retrieval agent with user document text for Act matching
                retrieval = RetrievalAgent(
                    pdf_vectorstore=st.session_state.get("pdf_vectorstore"),
                    corpus_vectorstore=st.session_state.get("corpus_vectorstore"),
                    scraper_vectorstore=st.session_state.get("judgments_vectorstore"),
                    top_k=6,
                    llm_client=st.session_state.llm_client,
                    gemini_api_key=GEMINI_API_KEY,
                    user_document_text=st.session_state.get("user_document_text")
                )
                
                summarizer = SummarizerAgent(st.session_state.llm_client)
                reasoner = ReasoningAgent(st.session_state.llm_client, retrieval, summarizer)

                out = reasoner.run(user_q)

                # Store reasoning & final answer separately
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_q
                })
                
                # Get IndiaCode citations if available
                indiacode_citations = retrieval.get_matched_acts_citations()
                
                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": {
                        "plan": out.get("plan", ""),
                        "answer": out.get("answer", ""),
                        "summary": out.get("summary", ""),
                        "citations": indiacode_citations
                    }
                })
            except Exception as e:
                st.error(f"Error processing query: {e}")
                import traceback
                st.error(traceback.format_exc())

    # Render chat
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            st.write(user_template.replace("{{MSG}}", m["content"]), unsafe_allow_html=True)
        else:
            # Add toggle for reasoning vs answer
            bot_msg = m["content"]
            with st.expander("💡 View reasoning steps"):
                st.markdown(f"**Plan:**\n\n{bot_msg['plan']}")
                st.markdown(f"**Context Summary:**\n\n{bot_msg['summary']}")
            st.markdown(bot_template.replace("{{MSG}}", bot_msg['answer']), unsafe_allow_html=True)

if __name__ == "__main__":
    main()