# 📘 Legal Document Question-Answering System (Agentic RAG)

This project is an **Agentic RAG (Retrieval-Augmented Generation)** system that allows users to upload PDFs or text-based documents and ask detailed questions about their contents. The system performs intelligent retrieval, semantic understanding, and structured reasoning to produce accurate, explainable answers.

---

## Features

### 1. Document Upload & Processing
- Supports **PDF**, **MS Word (.docx)**, and **text (.txt)** files.
- Extracts text using:
  - Standard PDF parsers  
  - OCR fallback for scanned PDFs  
  - DOCX and TXT loaders  
- Cleans and prepares the extracted text for indexing.

### 2. Multi-Agent RAG Pipeline
- **Scraper Agent** (optional for automated document sourcing)
- **Embedding & Indexing Pipeline**  
  Creates vectorstores for:
  - Uploaded PDFs
  - External documents like data scraped from IndiaCode website (if enabled)

### 3. Reasoning Agent
- Breaks down user queries into a **reasoning plan**
- Retrieves the most relevant document chunks
- Summarizes context
- Produces a final answer with clear **reasoning and evidence**

### 4. Vector-Based Retrieval
- Converts text into embeddings (semantic vectors)
- Retrieves top-K relevant results from:
  - PDF vectorstore  
  - Other document stores (if configured)

### 5. Weighted Vectorstores (Importance-Based Retrieval)
- Each document source (PDF uploads, external documents, scraped data) is assigned a **custom weight**. Currently set as user PDF(1.5) ,IndiaCode website data(1.0), Landmark Judgements Data(0.8).
- During retrieval, relevance scores are multiplied by these weights.
- Allows giving **higher priority** to certain sources (e.g., user-uploaded PDFs > external data).
- Ensures the reasoning agent bases answers on the *most authoritative* or *user-preferred* source.

### 6. Accurate, Context-Grounded Responses
- Structured output:
  - **Short answer**
  - **Reasoning / analysis**
  - **Citations from retrieved content**

---

## System Architecture (High-Level)
Uploaded Document (PDF, DOCX, TXT)→ Text Extraction → Chunking & Embeddings → Vectorstore → User Query → Reasoning Agent → Plans → Retrieves → Summarizes → Answers

### **How to set up the application**
1. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```
2. Install Dependencies( use python version 3.9)
```bash
pip install -r requirements.txt
```
3. Set up api key
```bash
touch .env
GEMINI_API_KEY=your_key_here
```
4. Run the application
```bash
streamlit run app.py
```

