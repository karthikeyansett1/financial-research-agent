# Phase 1 — Financial Document RAG

## What this does
A retrieval-augmented generation pipeline that lets users query financial documents in plain English and get answers grounded directly in the source text.

## Architecture decisions

**Chunking:** 500-word chunks with 50-word overlap. Overlap prevents answers that span chunk boundaries from being lost during retrieval.

**Embedding model:** all-MiniLM-L6-v2 from sentence-transformers. 384-dimensional vectors, runs locally with no API cost.

**Vector store:** FAISS IndexFlatL2. Exact search is fast enough at this scale without needing approximate methods.

**LLM:** Qwen3.6-27b via Groq. Temperature 0.1 to prioritize factual, grounded responses.

**Prompt design:** Model is explicitly instructed to answer only from provided context and cite specific figures. Reduces hallucination significantly.

## Stack
pypdf, sentence-transformers, FAISS, Groq API, Gradio

## How to run
```bash
python3 phase1-rag/app.py
```
