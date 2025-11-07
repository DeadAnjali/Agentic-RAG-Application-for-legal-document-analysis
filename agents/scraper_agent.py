# agents/scraper_agent.py
import os
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
from utils.vectorstore_utils import chunk_texts, build_faiss_from_texts

BASE_URL = "https://www.sci.gov.in/landmark-judgment-summaries/"

def fetch_year_data(year: int):
    """Scrape all landmark judgments for a specific year."""
    print(f"Fetching year {year}...")
    url = f"{BASE_URL}?judgment_year={year}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    for tr in soup.select("table tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        sno = tds[0].get_text(strip=True)
        date = tds[1].get_text(strip=True)
        case_name = tds[2].get_text(separator=" ", strip=True)
        summary = tds[3].get_text(separator=" ", strip=True)

        details_div = tds[4]
        justices = details_div.find(text=lambda t: "Justice" in t or "J." in t)
        pdf_link = details_div.select_one("a[href*='view-pdf']")
        pdf_url = pdf_link["href"] if pdf_link else None

        rows.append({
            "Year": year,
            "Serial": sno,
            "Date": date,
            "Case": case_name,
            "Summary": summary,
            "Justices": justices,
            "PDF_Link": pdf_url
        })
    return rows

def scrape_all_years(start=2000, end=2025):
    """Scrape all available years from SCI site."""
    all_data = []
    for year in range(start, end + 1):
        try:
            all_data.extend(fetch_year_data(year))
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching {year}: {e}")
    return all_data

def build_judgment_vectorstore(refresh=False):
    """Build or load vectorstore from landmark judgments."""
    data_path = "data/landmark_judgments.csv"
    if refresh or not os.path.exists(data_path):
        print("Scraping Supreme Court landmark judgments...")
        all_data = scrape_all_years()
        df = pd.DataFrame(all_data)
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(df)} judgments.")
    else:
        df = pd.read_csv(data_path)

    texts = []
    for _, row in df.iterrows():
        block = (
            f"Case: {row.get('Case','')}\n"
            f"Date: {row.get('Date','')}\n"
            f"Justices: {row.get('Justices','')}\n"
            f"Summary: {row.get('Summary','')}\n"
            f"PDF_Link: {row.get('PDF_Link','')}"
        )
        texts.append(block)

    chunks = chunk_texts(texts)
    return build_faiss_from_texts(chunks)
