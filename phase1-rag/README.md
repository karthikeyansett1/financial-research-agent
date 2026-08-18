# Phase 1 — Financial Document RAG

## What this does
A retrieval-augmented generation pipeline that lets users query financial 
documents in plain English and get answers grounded directly in the source text.

## Why I built it this way

**Chunking strategy:** Word-based chunking with 500-word chunks and 50-word overlap.
The overlap ensures answers that span chunk boundaries are not lost during retrieval.

**Embedding model:** all-MiniLM-L6-v2 from sentence-transformers. Chosen for speed 
and efficiency — 384-dimensional vectors, runs locally with no API cost.

**Vector store:** FAISS IndexFlatL2. Exact search over 51 vectors is fast enough 
that approximate search would add complexity without any real benefit at this scale.

**LLM:** Qwen3.6-27b via Groq. Temperature set to 0.1 to prioritize factual, 
grounded responses over creative generation.

**Prompt design:** Explicitly instructs the model to answer only from provided 
context and to cite specific figures. Reduces hallucination.

## Stack
- pypdf — PDF text extraction
- sentence-transformers — local embeddings
- FAISS — vector similarity search
- Groq API — LLM inference
- Gradio — UI

## How to run
pip install -r requirements.txt
python3 phase1-rag/app.py

## What I learned
- Chunking strategy directly impacts retrieval quality
- Overlap between chunks is critical for preserving context at boundaries
- Low temperature significantly reduces hallucination in financial Q&A
- FAISS exact search is fast enough for document-scale RAG without approximation
