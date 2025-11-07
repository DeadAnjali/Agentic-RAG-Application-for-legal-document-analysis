# agents/retrieval_agent.py
class RetrievalAgent:
    """
    Provides retrieval from PDF, IndiaCode, and Scraper vectorstores.
    Methods return concatenated context text (with separators).
    """

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
class RetrievalAgent:

    #Provides retrieval from both PDF and IndiaCode vectorstores.
    #Methods return concatenated context text (with separators).

    def __init__(self, pdf_vectorstore=None, corpus_vectorstore=None, top_k=5):
        self.pdf_vectorstore = pdf_vectorstore
        self.corpus_vectorstore = corpus_vectorstore
        self.top_k = top_k
    def __init__(self, pdf_vectorstore=None, corpus_vectorstore=None, judgments_vectorstore=None, top_k=5):
        self.pdf_vectorstore = pdf_vectorstore
        self.corpus_vectorstore = corpus_vectorstore
        self.judgments_vectorstore = judgments_vectorstore
        self.top_k = top_k

    def retrieve(self, query):
        parts = []

        if self.pdf_vectorstore:
            retr = self.pdf_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)
            parts += [d.page_content for d in docs if getattr(d, "page_content", None)]

        if self.corpus_vectorstore:
            retr = self.corpus_vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retr.get_relevant_documents(query)
            parts += [d.page_content for d in docs if getattr(d, "page_content", None)]

        if not parts:
            return ""

        # separate sources for clarity
        return "\n\n---\n\n".join(parts)
"""
