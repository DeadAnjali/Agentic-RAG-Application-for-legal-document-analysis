# agents/summarizer_agent.py
class SummarizerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def summarize(self, text, max_chars=4000):
        if not text:
            return ""
        text = text if len(text) <= max_chars else text[:max_chars]
        prompt = f"Summarize the following legal text into concise bullet points for a lawyer:\n\n{text}"
        return self.llm.generate(prompt)
