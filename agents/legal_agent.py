class LegalAgent:
    def __init__(self, llm, retriever_agent, summarizer_agent):
        self.llm = llm
        self.retriever_agent = retriever_agent
        self.summarizer_agent = summarizer_agent

    def plan(self, query):
        plan_prompt = f"""
        You are a legal reasoning planner.
        Given this question: "{query}"
        Break it into 2-3 reasoning steps for Indian law analysis.
        Example:
        1. Identify relevant Acts or Sections.
        2. Retrieve and summarize them.
        3. Provide reasoning with citations.
        """
        return self.llm.generate(plan_prompt)

    def run(self, query):
        plan = self.plan(query)

        # Step 1: Retrieve context
        docs = self.retriever_agent.retrieve(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Step 2: Summarize
        summary = self.summarizer_agent.summarize(context)

        # Step 3: Final reasoning
        reasoning_prompt = f"""
        You are a legal expert. 
        Context:
        {summary}

        Based on the above, answer this legal question clearly and accurately:
        {query}

        Include citations (e.g., 'Section 420 IPC', 'Article 21 Constitution') wherever applicable.
        """
        answer = self.llm.generate(reasoning_prompt)

        return f"**Plan:**\n{plan}\n\n**Answer:**\n{answer}"
