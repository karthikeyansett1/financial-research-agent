import gradio as gr
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
phase2_src = os.path.join(project_root, "phase2-agents", "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if phase2_src not in sys.path:
    sys.path.insert(0, phase2_src)

from agent import run_agent


def ask_agent(question: str) -> tuple:
    if not question.strip():
        return "Please enter a question.", ""

    result = run_agent(question)

    tool_log = "\n\n".join([
        f"Tool: {t['tool']}\nInput: {t['input']}\nResult: {t['result']}..."
        for t in result["tool_calls"]
    ])

    return result["answer"], tool_log


with gr.Blocks(title="Financial Research Agent", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Financial Research Agent")
    gr.Markdown(
        "An autonomous agent that searches financial documents, "
        "runs calculations, and queries live web data to answer your questions."
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Query",
                placeholder="e.g. What was Apple's revenue growth rate and what is the latest news about Apple?",
                lines=3
            )
            submit_btn = gr.Button("Run Agent", variant="primary")
            answer_output = gr.Textbox(
                label="Answer",
                lines=10,
                interactive=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### Agent Reasoning Log")
            tool_log_output = gr.Textbox(
                label="Tools called and results",
                lines=20,
                interactive=False
            )

    submit_btn.click(
        fn=ask_agent,
        inputs=[question_input],
        outputs=[answer_output, tool_log_output]
    )

if __name__ == "__main__":
    app.launch()