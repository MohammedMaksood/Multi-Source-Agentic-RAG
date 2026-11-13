"""Download the Phase 1 corpus (public arXiv papers) into data/papers/."""
import re
import time
import defusedxml.ElementTree as ET
from pathlib import Path

import requests

ARXIV_IDS = [
    "2005.11401",  # RAG
    "2004.04906",  # DPR
    "2004.12832",  # ColBERT
    "2104.08663",  # BEIR
    "2212.10496",  # HyDE
    "2208.03299",  # Atlas
    "2301.12652",  # REPLUG
    "2305.06983",  # FLARE
    "2307.03172",  # Lost in the Middle
    "2309.15217",  # RAGAS
    "2310.11511",  # Self-RAG
    "2312.06648",  # Dense X Retrieval
    "2312.10997",  # RAG survey
    "2401.15884",  # CRAG
    "2401.18059",  # RAPTOR
    "2402.03216",  # BGE M3-Embedding
    "2403.14403",  # Adaptive-RAG
    "2404.16130",  # GraphRAG
    "2506.05176",  # Qwen3 Embedding
]

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"
API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


def fetch_title(arxiv_id: str) -> str:
    resp = requests.get(API, params={"id_list": arxiv_id}, timeout=30)
    resp.raise_for_status()
    entry = ET.fromstring(resp.text).find(f"{ATOM}entry")
    return re.sub(r"\s+", " ", entry.find(f"{ATOM}title").text).strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arxiv_id in ARXIV_IDS:
        pdf_path = OUT_DIR / f"{arxiv_id}.pdf"
        title = fetch_title(arxiv_id)
        if not pdf_path.exists():
            resp = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=120)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
        print(f"{arxiv_id}  {title}")
        time.sleep(3)  # arXiv API etiquette


if __name__ == "__main__":
    main()
