import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Takes the user's question and the retrieved chunks,
    builds a prompt, sends it to Groq, and returns the answer.
    """
    # stitch the retrieved chunks into one context block
    context = "\n\n".join([
        f"[Chunk {chunk['chunk_index']}]: {chunk['text']}"
        for chunk in retrieved_chunks
    ])

    # this is the core prompt — notice we explicitly tell the LLM
    # to only use the provided context, not its own knowledge
    prompt = f"""You are a financial analyst assistant. 
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the document to answer this."
Always mention specific numbers and figures when available.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # low temperature = more factual, less creative
    )

    answer = response.choices[0].message.content

    # strip internal reasoning block if model returns one
    if "<think>" in answer and "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    return {
        "question": query,
        "answer": answer,
        "sources": [c["chunk_index"] for c in retrieved_chunks]
    }


if __name__ == "__main__":
    from embedder import load_index
    from retriever import retrieve

    index, chunks = load_index()

    query = "What was Apple's total revenue in Q1 2025 and how did iPhone perform?"
    retrieved = retrieve(query, index, chunks, top_k=5)
    result = generate_answer(query, retrieved)

    print(f"Question: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources used: chunks {result['sources']}")