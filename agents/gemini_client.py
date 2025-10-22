# agents/gemini_client.py
import os
import requests

class GeminiClient:
    """
    Minimal Gemini REST wrapper.
    Expects GEMINI_API_KEY in env or passed to constructor.
    Parses typical v1beta response structure.
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    def generate(self, prompt: str, max_output_tokens: int = 512) -> str:
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            # you can add other params if needed (temperature, safety settings) per API
        }
        resp = requests.post(self.endpoint, headers=headers, json=payload)
        raw = resp.text
        try:
            resp.raise_for_status()
            data = resp.json()
            # typical v1beta response path
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            # include status and raw body for debugging in the UI
            return f"Error: Could not parse Gemini response.\nStatus: {resp.status_code}\nBody: {raw}"
