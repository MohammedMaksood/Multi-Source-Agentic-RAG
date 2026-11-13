"""Extract text from the corpus PDFs and split it into overlapping word chunks."""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"
CHUNK_WORDS = 220
OVERLAP_WORDS = 40


@dataclass(frozen=True)
class Chunk:
    paper_id: str
    index: int
    text: str


def normalize(text: str) -> str:
    """Make extracted PDF text comparable: join hyphenated line breaks, fold ligatures, collapse spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def paper_text(pdf_path: Path) -> str:
    with pymupdf.open(pdf_path) as doc:
        text = normalize("\n".join(page.get_text() for page in doc))
    cut = text.rfind(" References ")
    return text[:cut] if cut > len(text) // 2 else text


def chunk_paper(paper_id: str, text: str) -> list[Chunk]:
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [
        Chunk(paper_id, i, " ".join(words[start:start + CHUNK_WORDS]))
        for i, start in enumerate(range(0, len(words), step))
    ]


def load_chunks() -> list[Chunk]:
    chunks = []
    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        chunks.extend(chunk_paper(pdf_path.stem, paper_text(pdf_path)))
    return chunks
