import os
import sys
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
phase1_src = os.path.join(project_root, "phase1-rag", "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if phase1_src not in sys.path:
    sys.path.insert(0, phase1_src)

from embedder import load_index
from retriever import retrieve
from generator import generate_answer

# ground truth test set — questions where we know the exact answer
# this is what separates a toy from something production-grade
TEST_SET = [
    {
        "question": "What was Apple's total net sales in Q1 2025?",
        "ground_truth": "124300 million"
    },
    {
        "question": "What was iPhone revenue in Q1 2025?",
        "ground_truth": "69138 million"
    },
    {
        "question": "What was Apple's Services revenue in Q1 2025?",
        "ground_truth": "26340 million"
    },
    {
        "question": "What was Apple's net income in Q1 2025?",
        "ground_truth": "36330 million"
    },
    {
        "question": "What was Mac revenue in Q1 2025?",
        "ground_truth": "8987 million"
    },
    {
        "question": "What was iPad revenue in Q1 2025?",
        "ground_truth": "8088 million"
    },
    {
        "question": "How much did Apple spend on research and development in Q1 2025?",
        "ground_truth": "8268 million"
    },
    {
        "question": "What was Apple's gross margin in Q1 2025?",
        "ground_truth": "58275 million"
    },
    {
        "question": "How many shares did Apple repurchase in Q1 2025?",
        "ground_truth": "100 million shares"
    },
    {
        "question": "What was Apple's cash and cash equivalents as of December 28 2024?",
        "ground_truth": "30299 million"
    }
]


def check_answer_contains_key_figures(answer: str, ground_truth: str) -> bool:
    """
    Simple evaluation — checks if the key figure from ground truth
    appears in the generated answer.
    """
    key_number = ground_truth.split()[0].replace(",", "")
    answer_clean = answer.replace(",", "")
    return key_number in answer_clean


def run_eval():
    print("Loading index...")
    index, chunks = load_index(save_dir=os.path.join(project_root, "phase1-rag/data"))

    results = []
    passed = 0

    print(f"Running eval on {len(TEST_SET)} questions...\n")

    for i, test in enumerate(TEST_SET):
        question = test["question"]
        ground_truth = test["ground_truth"]

        retrieved = retrieve(question, index, chunks, top_k=5)
        result = generate_answer(question, retrieved)
        answer = result["answer"]

        correct = check_answer_contains_key_figures(answer, ground_truth)
        if correct:
            passed += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer[:200],
            "correct": correct
        })

        status = "PASS" if correct else "FAIL"
        print(f"[{status}] Q{i+1}: {question[:60]}")
        if not correct:
            print(f"       Expected: {ground_truth}")
            print(f"       Got: {answer[:100]}")

    score = passed / len(TEST_SET) * 100
    print(f"\nFinal Score: {passed}/{len(TEST_SET)} ({score:.1f}%)")
    return score, results


if __name__ == "__main__":
    score, results = run_eval()