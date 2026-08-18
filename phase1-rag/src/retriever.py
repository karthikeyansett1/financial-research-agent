from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, index, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Takes a user's question, converts it to a vector,
    searches FAISS for the top_k most similar chunks,
    and returns those chunks with their similarity scores.
    """
    # convert the question to a vector
    query_vector = model.encode([query]).astype("float32")

    # search FAISS — returns distances and indices of top_k results
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # -1 means no result found
            results.append({
                "chunk_index": chunks[idx]["chunk_index"],
                "text": chunks[idx]["text"],
                "score": float(distances[0][i])  # lower = more similar
            })

    return results


if __name__ == "__main__":
    from embedder import load_index

    index, chunks = load_index()

    # test with a real question about the document
    query = "What was Apple's total revenue in Q1 2025?"
    results = retrieve(query, index, chunks, top_k=3)

    print(f"Query: {query}\n")
    print(f"Top 3 most relevant chunks:\n")
    for i, result in enumerate(results):
        print(f"--- Result {i+1} (score: {result['score']:.4f}) ---")
        print(result["text"][:300])
        print()