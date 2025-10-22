# agents/indiacode_agent.py
import os
import json
from utils.vectorstore_utils import chunk_texts, build_faiss_from_texts

def load_indiacode_json(path="data/indiacode_data.json"):
    """
    Loads indiacode JSON and returns list of text blocks (one per entry).
    Accepts JSON shaped like:
    {"allacts": {...}, "allregulations": {...}, "allrules": {...}, ...}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"IndiaCode JSON not found at: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    docs = []
    # iterate over top-level keys (allacts, allregulations, ...) and their entries
    for top_key, entries in data.items():
        if not isinstance(entries, dict):
            continue
        for title, details in entries.items():
            metadata = details.get("metadata", {}) or {}
            pdf_links = details.get("pdfLinks", []) or []

            short_title = metadata.get("Act Short Title:", metadata.get("Short Title", "")).strip()
            act_id = metadata.get("Act ID:", metadata.get("ActID", "")).strip()
            act_number = metadata.get("Act Number:", "").strip()
            act_year = metadata.get("Act Year:", "").strip()
            long_title = metadata.get("Long Title:", "").strip()
            enactment_date = metadata.get("Enactment Date:", "").strip()
            enforcement_date = metadata.get("Enforcement Date:", "").strip()

            lines = [
                f"Source: IndiaCode ({top_key})",
                f"Title: {title}",
            ]
            if short_title:
                lines.append(f"Short Title: {short_title}")
            if act_id:
                lines.append(f"Act ID: {act_id}")
            if act_number:
                lines.append(f"Act Number: {act_number}")
            if act_year:
                lines.append(f"Act Year: {act_year}")
            if enactment_date:
                lines.append(f"Enactment Date: {enactment_date}")
            if enforcement_date:
                lines.append(f"Enforcement Date: {enforcement_date}")
            if long_title:
                lines.append(f"Long Title: {long_title}")
            if pdf_links:
                lines.append("PDF Links:")
                for p in pdf_links:
                    lines.append(p)

            text_block = "\n".join([ln for ln in lines if ln and ln.strip()])
            docs.append(text_block)

    return docs


def build_indiacode_vectorstore(json_path="data/indiacode_data.json"):
    docs = load_indiacode_json(json_path)
    if not docs:
        raise ValueError("No IndiaCode docs found in JSON.")
    chunks = chunk_texts(docs)
    vs = build_faiss_from_texts(chunks)
    return vs
