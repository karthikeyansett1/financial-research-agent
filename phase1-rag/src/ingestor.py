import os
from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """
    Reads a PDF file and extracts all text from it.
    Returns one big string with all the text.
    """
    reader = PdfReader(file_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Splits a large text into overlapping chunks.
    
    chunk_size: how many words per chunk
    overlap: how many words to repeat between chunks
    so we don't lose context at boundaries
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        # grab chunk_size words starting from 'start'
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text_str,
            "word_count": len(chunk_words)
        })

        # move forward by (chunk_size - overlap)
        # this creates the overlap between chunks
        start += chunk_size - overlap
        chunk_index += 1

    return chunks



if __name__ == "__main__":
    pdf_path = "phase1-rag/data/10Q-Q1-2025-as-filed.pdf"
    text = load_pdf(pdf_path)

    print(f"Total characters extracted: {len(text)}")

    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")
    print(f"\nFirst chunk:")
    print(chunks[0]["text"])
    print(f"\nSecond chunk (notice the overlap with first):")
    print(chunks[1]["text"][:200])