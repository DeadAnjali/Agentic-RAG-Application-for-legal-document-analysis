# agents/retrieval_agent.py
class RetrievalAgent:
    """
    Provides weighted retrieval from PDF, IndiaCode, and Scraper vectorstores.
    Higher weight => higher priority in final merged context.
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
    ):
        self.pdf_vectorstore = pdf_vectorstore
        self.corpus_vectorstore = corpus_vectorstore
        self.scraper_vectorstore = scraper_vectorstore
        self.top_k = top_k
        
        # importance weights
        self.pdf_weight = pdf_weight
        self.indiacode_weight = indiacode_weight
        self.judgment_weight = judgment_weight

    def retrieve(self, query):
        ranked_docs = []   # (weighted_score, text_chunk)

        # ---------------------------
        # Retrieve from PDF vectorstore (highest weight)
        # ---------------------------
        if self.pdf_vectorstore:
            retr = self.pdf_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)

            for d in docs:
                score = getattr(d.metadata, "score", 1) if hasattr(d, "metadata") else 1
                text = d.page_content if getattr(d, "page_content", None) else ""
                ranked_docs.append((self.pdf_weight * score, f"[PDF] {text}"))

        # ---------------------------
        # Retrieve from IndiaCode vectorstore
        # ---------------------------
        if self.corpus_vectorstore:
            retr = self.corpus_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)

            for d in docs:
                score = getattr(d.metadata, "score", 1) if hasattr(d, "metadata") else 1
                text = d.page_content if getattr(d, "page_content", None) else ""
                ranked_docs.append((self.indiacode_weight * score, f"[IndiaCode] {text}"))

        # ---------------------------
        # Retrieve from Judgments vectorstore
        # ---------------------------
        if self.scraper_vectorstore:
            retr = self.scraper_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)

            for d in docs:
                score = getattr(d.metadata, "score", 1) if hasattr(d, "metadata") else 1
                text = d.page_content if getattr(d, "page_content", None) else ""
                ranked_docs.append((self.judgment_weight * score, f"[Judgments] {text}"))

        # If nothing retrieved, return blank
        if not ranked_docs:
            return ""

        # ---------------------------
        # Sort by weighted score, descending (higher = more relevant)
        # ---------------------------
        ranked_docs.sort(key=lambda x: x[0], reverse=True)

        # ---------------------------
        # Combine into final context string
        # ---------------------------
        parts = [chunk for _, chunk in ranked_docs]

        return "\n\n---\n\n".join(parts)

"""
class RetrievalAgent:
    #Provides retrieval from PDF, IndiaCode, and Scraper vectorstores.
    #Methods return concatenated context text (with separators).
    def __init__(self, pdf_vectorstore=None, corpus_vectorstore=None, scraper_vectorstore=None, top_k=5):
        self.pdf_vectorstore = pdf_vectorstore
        self.corpus_vectorstore = corpus_vectorstore
        self.scraper_vectorstore = scraper_vectorstore
        self.top_k = top_k

    def retrieve(self, query):
        parts = []

        # Retrieve from PDF vectorstore
        if self.pdf_vectorstore:
            retr = self.pdf_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)
            parts += [
                f"[PDF] {d.page_content}" 
                for d in docs if getattr(d, "page_content", None)
            ]

        # Retrieve from IndiaCode corpus vectorstore
        if self.corpus_vectorstore:
            retr = self.corpus_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)
            parts += [
                f"[IndiaCode] {d.page_content}" 
                for d in docs if getattr(d, "page_content", None)
            ]

        # Retrieve from Scraper (Judgments) vectorstore
        if self.scraper_vectorstore:
            retr = self.scraper_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)
            parts += [
                f"[Judgments] {d.page_content}" 
                for d in docs if getattr(d, "page_content", None)
            ]

        if not parts:
            return ""

        # Separate different sources for clarity
        return "\n\n---\n\n".join(parts)

"""
