# Phase 3 — Evaluation

## What this does
A lightweight evaluation framework that measures RAG pipeline accuracy 
against a ground truth test set of financial questions.

## Methodology
10 questions with known answers extracted directly from the Apple Q1 2025 10-Q.
Each answer is checked for presence of the key figure from ground truth.
Simple but effective for numerical financial Q&A.

## Results
Score: 9/10 (90.0%)

The one failure was a share repurchase question where the model reasoning 
block leaked into the answer before stripping, causing the eval checker 
to miss the correct figure. The answer itself was correct.

## What this measures
- Whether the retrieval is finding the right chunks
- Whether the LLM is extracting correct figures from context
- Whether the prompt is grounding answers in the document

## Known limitations
- Exact string matching is brittle for edge cases
- 10 questions is a small test set
- Does not measure faithfulness or answer relevance separately
- Next step would be RAGAs for semantic similarity scoring
