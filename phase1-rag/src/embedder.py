from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Load the embedding model once at module level
# all-MiniLM-L6-v2 is small, fast, and good enough for our use case
# it converts text into 384-dimensional vectors
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Takes our list of chunk dicts and converts each chunk's text
    into a vector using the sentence transformer model.
    Returns a numpy array of shape (num_chunks, 384)
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index from our embeddings.
    IndexFlatL2 = exact search using L2 (euclidean) distance.
    Simple but works great for small datasets like ours.
    """
    dimension = embeddings.shape[1]  # 384 for our model
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index(index, chunks, save_dir: str = "phase1-rag/data"):
    """
    Saves the FAISS index and chunks to disk so we don't have
    to rebuild every time we run the app.
    """
    os.makedirs(save_dir, exist_ok=True)
    faiss.write_index(index, f"{save_dir}/index.faiss")
    with open(f"{save_dir}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index and chunks to {save_dir}")


def load_index(save_dir: str = "phase1-rag/data"):
    """
    Loads a previously saved FAISS index and chunks from disk.
    """
    index = faiss.read_index(f"{save_dir}/index.faiss")
    with open(f"{save_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded index with {index.ntotal} vectors")
    return index, chunks


if __name__ == "__main__":
    from ingestor import load_pdf, chunk_text

    pdf_path = "phase1-rag/data/10Q-Q1-2025-as-filed.pdf"
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)

    embeddings = embed_chunks(chunks)
    print(f"Embedding shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    save_index(index, chunks)