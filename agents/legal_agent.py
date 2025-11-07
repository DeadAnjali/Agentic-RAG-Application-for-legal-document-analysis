'''class LegalAgent:
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
'''
import json
import re

class LegalAgent:
    def __init__(self, llm, retriever_agent, summarizer_agent, indiacode_metadata_path="data/indiacode_data.json"):
        self.llm = llm
        self.retriever_agent = retriever_agent
        self.summarizer_agent = summarizer_agent

        # Load IndiaCode metadata for Act sources
        try:
            with open(indiacode_metadata_path, "r", encoding="utf-8") as f:
                self.indiacode_metadata = json.load(f)
        except Exception as e:
            print(f"[Warning] Could not load IndiaCode metadata: {e}")
            self.indiacode_metadata = {}

    def plan(self, query):
        """
        Generate concise legal reasoning steps (not verbose).
        """
        plan_prompt = f"""
        You are a concise legal reasoning planner for Indian law.
        Given this question: "{query}"

        Break the reasoning into **2 or 3 brief steps** only.
        Focus on:
        1. Identifying relevant Acts or key legal provisions.
        2. Retrieving and summarizing relevant content.
        3. Giving a short reasoning or conclusion.

        Keep the steps concise and numbered.
        """
        return self.llm.generate(plan_prompt).strip()

    def _extract_acts(self, text):
        """
        Extract probable Act names or short forms (e.g., IPC, Constitution, etc.)
        and attach metadata sources if found in IndiaCode.
        """
        acts = set()
        # Simple regex to capture phrases like 'Act', 'Code', 'Constitution', etc.
        matches = re.findall(r"\b([A-Z][A-Za-z\s]*(?:Act|Code|Constitution))\b", text)
        acts.update(matches)

        # Build Act → Source map if available
        acts_with_sources = []
        for act in acts:
            source_url = None
            for entry in self.indiacode_metadata.get("acts", []):
                if act.lower() in entry.get("title", "").lower():
                    source_url = entry.get("source_url") or entry.get("link") or None
                    break

            if source_url:
                acts_with_sources.append(f"{act} — [Source]({source_url})")
            else:
                acts_with_sources.append(act)

        return acts_with_sources

    def run(self, query):
        # Step 1: Generate plan
        plan = self.plan(query)

        # Step 2: Retrieve relevant legal text
        context = self.retriever_agent.retrieve(query)
        summary = self.summarizer_agent.summarize(context)

        # Step 3: Generate final reasoning (concise)
        reasoning_prompt = f"""
        You are a legal expert specialized in Indian law.
        Using the following summarized context, provide a **clear and concise answer**.

        Context:
        {summary}

        Question:
        {query}

        - Focus on Acts, Articles, and Sections directly relevant.
        - Keep the explanation short and formal.
        - Include citations like 'Section 420 IPC' or 'Article 21 Constitution of India'.
        - Avoid redundant restating of context.
        """
        answer = self.llm.generate(reasoning_prompt).strip()

        # Step 4: Identify Acts from the answer and attach sources
        acts_with_sources = self._extract_acts(answer)

        # Format the final output neatly
        result = f"**Plan:**\n{plan}\n\n**Answer:**\n{answer}"
        if acts_with_sources:
            result += "\n\n**Identified Acts & Sources:**\n" + "\n".join(f"- {a}" for a in acts_with_sources)

        return result
