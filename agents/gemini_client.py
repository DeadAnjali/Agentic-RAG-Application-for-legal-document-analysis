# agents/gemini_client.py

import os
import requests

class GeminiClient:
    
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
'''
# agents/gemini_client.py
import os
import time
import random
import requests


class GeminiClient:

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.5-flash",
        timeout: int = 30,
        max_retries: int = 6,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent"
        )

    def generate(self, prompt: str, max_output_tokens: int = 512) -> str:
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_output_tokens},
        }

        # ---- RETRY LOOP ----
        for attempt in range(self.max_retries):

            try:
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                # Retry on server overload / transient server errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = (2 ** attempt) + random.random()
                    print(
                        f"[Gemini] Retry {attempt+1}/{self.max_retries} "
                        f"(HTTP {resp.status_code}), waiting {wait:.2f}s..."
                    )
                    time.sleep(wait)
                    continue

                # Raise other HTTP errors normally
                resp.raise_for_status()

                # Try to parse JSON result
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as net_err:

                # Network or connection-level failures → retry
                wait = (2 ** attempt) + random.random()
                print(
                    f"[Gemini] Network error, retry {attempt+1}/{self.max_retries}: "
                    f"{net_err} | waiting {wait:.2f}s..."
                )
                time.sleep(wait)

            except Exception as parse_err:
                # If response is not valid JSON / parse failure → retry
                raw = getattr(resp, "text", "")
                wait = (2 ** attempt) + random.random()
                print(
                    f"[Gemini] Parse error, retry {attempt+1}/{self.max_retries}: "
                    f"{parse_err} | waiting {wait:.2f}s..."
                )
                time.sleep(wait)

        # ---- ALL RETRIES FAILED ----
        return (
            f"Error: Gemini request failed after {self.max_retries} retries.\n"
            f"Last Status: {getattr(resp, 'status_code', 'N/A')}\n"
            f"Last Body: {getattr(resp, 'text', '')}"
        )
'''