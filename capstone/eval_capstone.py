"""
Day 5, Part 2: a tiny golden-set eval for the capstone agent.

Evals aren't a separate, later-stage thing you bolt on before shipping —
they're a small, checkable question: "for these known questions, does the
agent do the right thing?" This script asks 6 questions with a known-good
tool and/or keyword, runs each through capstone_agent's real agentic loop,
and scores tool-call accuracy the same way a production eval suite would
— just at a scale you can read end to end in one sitting.

Setup:
    pip install -r requirements.txt
    cp ../.env .env

Run:
    python eval_capstone.py
"""

import anthropic
import voyageai
from dotenv import load_dotenv

from capstone_agent import embed_documents, load_documents, make_tool_executor, run_agentic_loop

load_dotenv()

GOLDEN_SET = [
    {
        "question": "How many vacation days do I get per year?",
        "expect_tools": ["search_policies"],
        "expect_keyword": "18",
    },
    {
        "question": "What policies do you have available?",
        "expect_tools": ["list_policy_titles"],
        "expect_keyword": None,
    },
    {
        "question": "If I request PTO today, what's the earliest day I could take off?",
        "expect_tools": ["search_policies", "add_business_days"],
        "expect_keyword": None,
    },
    {
        "question": "Can I expense a new monitor without pre-approval?",
        "expect_tools": ["search_policies"],
        "expect_keyword": None,
    },
    {
        "question": "What's the capital of Australia?",
        "expect_tools": [],
        "expect_keyword": None,
    },
    {
        "question": "How quickly must an on-call engineer acknowledge a production incident?",
        "expect_tools": ["search_policies"],
        "expect_keyword": "15",
    },
]


def check_tools(actual_tools, expected_tools):
    return all(t in actual_tools for t in expected_tools)


def check_keyword(answer, keyword):
    if keyword is None:
        return True
    return keyword.lower() in answer.lower()


def run_eval(claude_client, execute):
    results = []
    for case in GOLDEN_SET:
        tools_called = []
        messages = [{"role": "user", "content": case["question"]}]
        answer = run_agentic_loop(
            claude_client, messages, execute,
            on_tool_call=lambda name: tools_called.append(name),
        )

        tools_ok = check_tools(tools_called, case["expect_tools"])
        keyword_ok = check_keyword(answer, case["expect_keyword"])
        passed = tools_ok and keyword_ok

        results.append({
            "question": case["question"],
            "passed": passed,
            "tools_called": tools_called,
            "expect_tools": case["expect_tools"],
            "tools_ok": tools_ok,
            "keyword_ok": keyword_ok,
            "answer": answer,
        })
    return results


def print_scorecard(results):
    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 70)
    print(f"SCORECARD: {passed}/{len(results)} passed")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{status}] {r['question']}")
        print(f"  tools called: {r['tools_called']}  (expected: {r['expect_tools']})")
        if not r["tools_ok"]:
            print("  -> tool-call mismatch")
        if not r["keyword_ok"]:
            print("  -> expected keyword missing from answer")
        print(f"  answer: {r['answer'][:150]}")


def main():
    voyage_client = voyageai.Client()
    claude_client = anthropic.Anthropic()

    filenames, texts = load_documents()
    print(f"Indexing {len(texts)} documents...")
    doc_vectors = embed_documents(voyage_client, texts)
    execute = make_tool_executor(voyage_client, filenames, texts, doc_vectors)

    print(f"Running {len(GOLDEN_SET)}-question golden set against the capstone agent...")
    results = run_eval(claude_client, execute)
    print_scorecard(results)


if __name__ == "__main__":
    main()
