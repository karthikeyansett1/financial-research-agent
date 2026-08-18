import os
import math
import sys
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# add phase1-rag/src to path so we can import from it
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
phase1_src = os.path.join(project_root, "phase1-rag", "src")
if phase1_src not in sys.path:
    sys.path.insert(0, phase1_src)

from embedder import load_index
from retriever import retrieve


def search_document(query: str, top_k: int = 5) -> str:
    try:
        index, chunks = load_index(save_dir=os.path.join(project_root, "phase1-rag/data"))
        results = retrieve(query, index, chunks, top_k=top_k)
        formatted = "\n\n".join([
            f"[Chunk {r['chunk_index']}]: {r['text'][:400]}"
            for r in results
        ])
        return formatted
    except Exception as e:
        return f"Document search failed: {str(e)}"


def calculate(expression: str) -> str:
    try:
        allowed = {
            'abs': abs, 'round': round,
            'min': min, 'max': max,
            'pow': pow, 'sqrt': math.sqrt,
            'log': math.log, 'pi': math.pi
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def web_search(query: str) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        formatted = "\n\n".join([
            f"Title: {r['title']}\nSummary: {r['body']}"
            for r in results
        ])
        return formatted
    except Exception as e:
        return f"Web search failed: {str(e)}"


TOOLS = {
    "search_document": {
        "fn": search_document,
        "description": "Search the loaded financial document for relevant information. Use this for questions about the document contents."
    },
    "calculate": {
        "fn": calculate,
        "description": "Evaluate a mathematical expression. Use this for financial calculations like growth rates, margins, ratios."
    },
    "web_search": {
        "fn": web_search,
        "description": "Search the web for current news and information. Use this for recent events not covered in the document."
    }
}


if __name__ == "__main__":
    print("Testing search_document...")
    print(search_document("What was Apple's total revenue?"))

    print("\nTesting calculate...")
    print(calculate("(124300 - 119575) / 119575 * 100"))

    print("\nTesting web_search...")
    print(web_search("Apple earnings 2025"))