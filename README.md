# Financial Research Agent

A multi-phase AI system that lets users query financial documents using natural language and get answers grounded directly in the source text.

## Project Structure

- `phase1-rag/` — RAG pipeline: PDF ingestion, chunking, embeddings, FAISS retrieval, Groq generation
- `phase2-agents/` — Agent layer: tool use, web search, calculator, multi-step reasoning
- `phase3-eval/` — Evaluation: RAGAs scoring, faithfulness and relevance metrics

## Stack
Python, FAISS, sentence-transformers, LangChain, Groq API, Gradio, DuckDuckGo Search

## How to run
```bash
pip install -r requirements.txt
python3 phase1-rag/app.py
```
