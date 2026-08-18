import gradio as gr
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.embedder import load_index, embed_chunks, build_faiss_index, save_index
from src.retriever import retrieve
from src.generator import generate_answer
from src.ingestor import load_pdf, chunk_text

# load the existing index on startup
index, chunks = load_index(save_dir="phase1-rag/data")


def answer_question(question: str) -> tuple:
    """
    Main function that ties everything together.
    Takes a question, retrieves relevant chunks, generates an answer.
    """
    if not question.strip():
        return "Please enter a question.", ""

    retrieved = retrieve(question, index, chunks, top_k=5)
    result = generate_answer(question, retrieved)

    sources_text = "\n\n".join([
        f"Chunk {r['chunk_index']} (relevance score: {r['score']:.4f}):\n{r['text'][:300]}..."
        for r in retrieved
    ])

    return result["answer"], sources_text


def process_uploaded_pdf(file) -> str:
    """
    Allows users to upload a new PDF and rebuilds the index against it.
    """
    global index, chunks

    if file is None:
        return "No file uploaded."

    text = load_pdf(file.name)
    chunks = chunk_text(text)

    from src.embedder import embed_chunks, build_faiss_index, save_index
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    save_index(index, chunks, save_dir="phase1-rag/data")

    return f"Indexed {len(chunks)} chunks from uploaded PDF. Ready for queries."


with gr.Blocks(title="Financial Research Agent", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Financial Research Agent")
    gr.Markdown(
        "Query financial documents using retrieval-augmented generation. "
        "Currently loaded: Apple Q1 2025 Form 10-Q"
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Query",
                placeholder="e.g. What was Apple's total revenue in Q1 2025?",
                lines=2
            )
            submit_btn = gr.Button("Submit", variant="primary")
            answer_output = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### Upload Document")
            pdf_upload = gr.File(
                label="Upload a financial PDF",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("Process Document")
            upload_status = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("### Retrieved Sources")
            sources_output = gr.Textbox(
                label="Chunks used to generate the answer",
                lines=10,
                interactive=False
            )

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )

    upload_btn.click(
        fn=process_uploaded_pdf,
        inputs=[pdf_upload],
        outputs=[upload_status]
    )

if __name__ == "__main__":
    app.launch()