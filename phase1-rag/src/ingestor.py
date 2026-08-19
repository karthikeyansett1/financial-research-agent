import os
import re
from pypdf import PdfReader
from html.parser import HTMLParser


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


def load_html(file_path: str) -> str:
    """
    Extracts clean readable text from an HTML or inline XBRL file.
    SEC filings are often inline XBRL — this skips the metadata
    and extracts only the human-readable financial content.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # skip everything before <body> — that's all XBRL metadata
    body_start = content.lower().find("<body")
    if body_start != -1:
        content = content[body_start:]

    # also skip the hidden XBRL data section at the top of body
    # it usually lives inside <ix:header> or hidden divs
    hidden_end_markers = ["</ix:header>", "</head>"]
    for marker in hidden_end_markers:
        pos = content.lower().find(marker)
        if pos != -1:
            content = content[pos + len(marker):]
            break

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
            self.skip = False
            self.skip_tags = {"script", "style", "ix:header", "ix:hidden"}
            self.depth = 0

        def handle_starttag(self, tag, attrs):
            tag_lower = tag.lower()
            if tag_lower in self.skip_tags:
                self.skip = True
                self.depth += 1
            elif self.skip:
                self.depth += 1

        def handle_endtag(self, tag):
            tag_lower = tag.lower()
            if self.skip:
                self.depth -= 1
                if self.depth <= 0:
                    self.skip = False
                    self.depth = 0

        def handle_data(self, data):
            if self.skip:
                return
            cleaned = data.strip()
            if not cleaned or len(cleaned) < 2:
                return
            # skip XBRL namespace data
            if any(x in cleaned for x in [
                "us-gaap:", "msft:", "dei:", "xbrl",
                "fasb.org", "xbrl.org", "0000789019"
            ]):
                return
            # skip pure URLs
            if cleaned.startswith("http"):
                return
            self.text.append(cleaned)

    parser = TextExtractor()
    parser.feed(content)

    full_text = " ".join(parser.text)
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    return full_text


def load_document(file_path: str) -> str:
    """
    Loads either a PDF or HTML document based on file extension.
    """
    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".html") or file_path.endswith(".htm"):
        return load_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


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
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text_str,
            "word_count": len(chunk_words)
        })

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