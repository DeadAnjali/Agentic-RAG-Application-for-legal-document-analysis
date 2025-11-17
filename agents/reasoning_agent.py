# agents/reasoning_agent.py

class ReasoningAgent:
    def __init__(self, llm_client, retrieval_agent, summarizer_agent):
        self.llm = llm_client
        self.retriever = retrieval_agent
        self.summarizer = summarizer_agent

    def plan(self, query):
        plan_prompt = (
            "You are a legal planner for Indian law. Given the question below, "
            "return 2-3 short numbered steps describing how you'd answer it "
            "(identify relevant Acts/Sections, retrieve materials, summarize, reason).\n\n"
            f"Question: {query}"
        )
        return self.llm.generate(plan_prompt)

    def run(self, query):
        plan = self.plan(query)
        context = self.retriever.retrieve(query)
        summary = self.summarizer.summarize(context)

        final_prompt = (
            "You are an expert in Indian law. Use the context summary and the retrieved materials to answer the question. "
            "Be precise and include citations where applicable (e.g., Section 420 IPC, Act: The Coinage Act, 2011). "
            "Respond with: Short answer and Relevant citations.\n\n"
            f"Context summary:\n{summary}\n\nQuestion: {query}"
        )
        answer = self.llm.generate(final_prompt)
        
        # Get IndiaCode citations if available
        indiacode_citations = self.retriever.get_matched_acts_citations()
        
        # Append citations to answer if they exist
        if indiacode_citations:
            answer = answer + indiacode_citations

        return {
            "plan": plan,
            "summary": summary,
            "answer": answer
        }