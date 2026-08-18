import os
import sys
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
phase2_src = os.path.join(project_root, "phase2-agents", "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if phase2_src not in sys.path:
    sys.path.insert(0, phase2_src)

from tools import TOOLS

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def run_agent(user_question: str, max_iterations: int = 5) -> dict:
    """
    The agent brain. Given a question it:
    1. Decides which tool to use
    2. Calls the tool
    3. Looks at the result
    4. Either calls another tool or returns a final answer
    """

    # build the tool descriptions for the system prompt
    tool_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    ])

    system_prompt = f"""You are a financial research assistant with access to the following tools:

{tool_descriptions}

To use a tool, respond with a JSON object in this exact format:
{{"tool": "tool_name", "input": "your input here"}}

When you have enough information to answer the question, respond with:
{{"tool": "final_answer", "input": "your complete answer here"}}

Always use search_document first for questions about the loaded financial document.
Use web_search for current news or information not in the document.
Use calculate for any mathematical operations.
Only call one tool at a time."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    tool_calls_log = []

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=messages,
            temperature=0.1
        )

        raw = response.choices[0].message.content

        # strip thinking block if present
        if "<think>" in raw and "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        # try to parse the tool call
        try:
            # find the JSON block in the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_str = raw[start:end]
            tool_call = json.loads(json_str)
        except Exception:
            # if we can't parse JSON, treat the whole response as final answer
            return {
                "question": user_question,
                "answer": raw,
                "tool_calls": tool_calls_log
            }

        tool_name = tool_call.get("tool")
        tool_input = tool_call.get("input", "")

        # if agent is done, return the final answer
        if tool_name == "final_answer":
            return {
                "question": user_question,
                "answer": tool_input,
                "tool_calls": tool_calls_log
            }

        # call the tool
        if tool_name in TOOLS:
            print(f"Calling tool: {tool_name} with input: {tool_input[:80]}...")
            tool_result = TOOLS[tool_name]["fn"](tool_input)
            tool_calls_log.append({
                "tool": tool_name,
                "input": tool_input,
                "result": tool_result[:200]
            })

            # add the tool result back to the conversation
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Tool result from {tool_name}:\n{tool_result}\n\nNow continue."
            })
        else:
            break

    return {
        "question": user_question,
        "answer": "Agent could not complete the task within the iteration limit.",
        "tool_calls": tool_calls_log
    }


if __name__ == "__main__":
    # test with a question that needs both document search and calculation
    question = "What was Apple's revenue growth rate from Q1 2024 to Q1 2025?"
    result = run_agent(question)

    print(f"\nQuestion: {result['question']}")
    print(f"\nTools used: {[t['tool'] for t in result['tool_calls']]}")
    print(f"\nAnswer: {result['answer']}")