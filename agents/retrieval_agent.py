# agents/retrieval_agent.py
from agents.indiacode_agent import (
    find_matching_acts, 
    get_act_context_from_matched_pdfs, 
    format_act_context
)

class RetrievalAgent:
    """
    Provides weighted retrieval from PDF, IndiaCode, and Scraper vectorstores.
    Now also matches Acts from uploaded documents and fetches their full PDFs.
    """

    def __init__(
        self,
        pdf_vectorstore=None,
        corpus_vectorstore=None,
        scraper_vectorstore=None,
        top_k=5,
        pdf_weight=1.5,
        indiacode_weight=1.0,
        judgment_weight=0.8,
        llm_client=None,
        gemini_api_key=None,
        indiacode_json_path="data/indiacode_data.json",
        user_document_text=None,  # NEW: Store extracted text from user's upload
    ):
        self.pdf_vectorstore = pdf_vectorstore
        self.corpus_vectorstore = corpus_vectorstore
        self.scraper_vectorstore = scraper_vectorstore
        self.top_k = top_k
        
        # importance weights
        self.pdf_weight = pdf_weight
        self.indiacode_weight = indiacode_weight
        self.judgment_weight = judgment_weight
        
        # NEW: For Act matching and PDF extraction
        self.llm_client = llm_client
        self.gemini_api_key = gemini_api_key
        self.indiacode_json_path = indiacode_json_path
        self.user_document_text = user_document_text
        self.matched_act_context = None  # Cache for matched Acts context
        self.matched_acts_metadata = []  # Store metadata for citation in final answer

    def set_user_document_text(self, text):
        """
        Set the text extracted from user's uploaded documents.
        This triggers Act matching and PDF extraction.
        """
        self.user_document_text = text
        self.matched_act_context = None  # Reset cache
        self.matched_acts_metadata = []  # Reset metadata

    def get_matched_acts_context(self):
        """
        Find Acts mentioned in user's document and fetch their full PDFs.
        Returns formatted context string to add to retrieval.
        """
        if self.matched_act_context is not None:
            return self.matched_act_context  # Use cached result
        
        if not self.user_document_text:
            self.matched_act_context = ""
            return ""
        
        print("\n" + "="*80)
        print("MATCHING ACTS FROM USER DOCUMENT WITH INDIACODE DATABASE")
        print("="*80)
        
        # Step 1: Find matching Acts
        matched_acts = find_matching_acts(
            self.user_document_text,
            indiacode_json_path=self.indiacode_json_path,
            threshold=0.6
        )
        
        if not matched_acts:
            print("No matching Acts found in IndiaCode database.")
            self.matched_act_context = ""
            return ""
        
        print(f"\nFound {len(matched_acts)} matching Acts")
        
        # Step 2: Fetch PDFs and extract context
        act_contexts = get_act_context_from_matched_pdfs(
            matched_acts,
            gemini_api_key=self.gemini_api_key,
            llm_client=self.llm_client,
            max_acts=3  # Limit to top 3 matches to avoid overwhelming context
        )
        
        # Step 3: Format for LLM consumption
        formatted_context = format_act_context(act_contexts)
        
        # Store metadata for later citation in final answer
        self.matched_acts_metadata = act_contexts
        
        # Cache the result
        self.matched_act_context = formatted_context
        
        print(f"\n✓ Successfully extracted context from {len(act_contexts)} Acts")
        print("="*80 + "\n")
        
        return formatted_context
    
    def get_matched_acts_citations(self):
        """
        Get formatted citations for matched Acts (to append to final answer).
        Returns a formatted string with Act names and their PDF links.
        """
        if not self.matched_acts_metadata:
            return ""
        
        citations = "\n\n" + "="*80 + "\n"
        citations += "📚 INDIACODE REFERENCES USED:\n"
        citations += "="*80 + "\n\n"
        
        for i, act in enumerate(self.matched_acts_metadata, 1):
            citations += f"{i}. **{act['act_title']}**"
            if act['act_year']:
                citations += f" ({act['act_year']})"
            citations += "\n"
            citations += f"   - Matched from user document: _{act['matched_reference']}_\n"
            citations += f"   - Source PDF: {act['pdf_url']}\n\n"
        
        return citations

    def retrieve(self, query):
        ranked_docs = []   # (weighted_score, text_chunk)

        
        if self.pdf_vectorstore:
            retr = self.pdf_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)

            for d in docs:
                score = getattr(d.metadata, "score", 1) if hasattr(d, "metadata") else 1
                text = d.page_content if getattr(d, "page_content", None) else ""
                ranked_docs.append((self.pdf_weight * score, f"[PDF] {text}"))

        
        if self.corpus_vectorstore:
            retr = self.corpus_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)

            for d in docs:
                score = getattr(d.metadata, "score", 1) if hasattr(d, "metadata") else 1
                text = d.page_content if getattr(d, "page_content", None) else ""
                ranked_docs.append((self.indiacode_weight * score, f"[IndiaCode] {text}"))

        
        if self.scraper_vectorstore:
            retr = self.scraper_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)

            for d in docs:
                score = getattr(d.metadata, "score", 1) if hasattr(d, "metadata") else 1
                text = d.page_content if getattr(d, "page_content", None) else ""
                ranked_docs.append((self.judgment_weight * score, f"[Judgments] {text}"))

        
        ranked_docs.sort(key=lambda x: x[0], reverse=True)

        
        parts = [chunk for _, chunk in ranked_docs]
        base_context = "\n\n---\n\n".join(parts)

        
        matched_acts_context = self.get_matched_acts_context()
        
        if matched_acts_context:
            final_context = base_context + matched_acts_context
        else:
            final_context = base_context

        return final_context if final_context else ""