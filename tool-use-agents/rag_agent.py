"""
Day 4, Part 2: A Small Agent — RAG as a Tool.

Builds directly on rag_search.py from Session 2: the same embedding +
retrieval logic, but instead of always running on every question, it's
wrapped as a tool Claude can choose to call. A second tool (business-day
date math) is included so some questions genuinely need both tools
together — e.g. asking when the earliest day off could be, given the
PTO policy's advance-notice requirement.

Setup:
    pip install -r requirements.txt
    cp ../.env .env      # reuse the same VOYAGE_API_KEY / ANTHROPIC_API_KEY

Run:
    python rag_agent.py
"""

from datetime import date, datetime, timedelta

import anthropic
import numpy as np
import voyageai
from dotenv import load_dotenv

from knowledge_base import DOCUMENTS

load_dotenv()

VOYAGE_MODEL = "voyage-3.5"
CLAUDE_MODEL = "claude-haiku-4-5"
MAX_TOOL_ITERATIONS = 5

SYSTEM = (
    "You are a helpful assistant for NorthStar Analytics employees. Use "
    "the search_policies tool for any question about company policy, "
    "process, or rules — do not answer those from memory. Use the "
    "add_business_days tool for any date math involving business days "
    f"instead of computing it yourself. Today's date is {date.today().isoformat()}."
)

TOOLS = [
    {
        "name": "search_policies",
        "description": (
            "Search the NorthStar Analytics policy knowledge base for text "
            "relevant to a question. Returns the top matching policy "
            "excerpts with similarity scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A natural-language search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_business_days",
        "description": (
            "Add a number of business days (Mon-Fri, skipping weekends) to "
            "a date. Use this for anything involving 'business days' or "
            "'working days' — never compute this yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Start date, YYYY-MM-DD"},
                "business_days": {"type": "integer", "description": "Number of business days to add"},
            },
            "required": ["start_date", "business_days"],
        },
    },
]


# ----------------------------------------------------------- retrieval --


def embed_documents(voyage_client, texts):
    result = voyage_client.embed(texts, model=VOYAGE_MODEL, input_type="document")
    return np.array(result.embeddings)


def embed_query(voyage_client, text):
    result = voyage_client.embed([text], model=VOYAGE_MODEL, input_type="query")
    return np.array(result.embeddings[0])


def top_k_matches(query_vector, doc_vectors, k=2):
    doc_norms = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    query_norm = query_vector / np.linalg.norm(query_vector)
    scores = doc_norms @ query_norm
    ranked = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in ranked]


# ----------------------------------------------------------- tool logic --


def add_business_days(start_date, business_days):
    d = datetime.strptime(start_date, "%Y-%m-%d")
    added = 0
    while added < business_days:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            added += 1
    return d.strftime("%Y-%m-%d (%A)")


def make_tool_executor(voyage_client, doc_vectors):
    def execute(name, tool_input):
        if name == "search_policies":
            query_vector = embed_query(voyage_client, tool_input["query"])
            matches = top_k_matches(query_vector, doc_vectors)
            return "\n\n".join(f"[score {score:.3f}] {DOCUMENTS[i]}" for i, score in matches)
        if name == "add_business_days":
            return add_business_days(tool_input["start_date"], tool_input["business_days"])
        raise ValueError(f"Unknown tool: {name}")

    return execute


# ------------------------------------------------------------- the loop --


def run_agentic_loop(claude_client, messages, execute):
    response = claude_client.messages.create(
        model=CLAUDE_MODEL, max_tokens=500, system=SYSTEM, tools=TOOLS, messages=messages,
    )

    iterations = 0
    while response.stop_reason == "tool_use":
        iterations += 1
        if iterations > MAX_TOOL_ITERATIONS:
            return "[stopped: too many tool calls in a row — likely stuck in a loop]"

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [agent] calling {block.name}({block.input})")
                try:
                    result = execute(block.name, block.input)
                    print(f"  [agent] -> {result}\n")
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(result)})
                except Exception as e:
                    print(f"  [agent] tool error -> {e}\n")
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True})
        messages.append({"role": "user", "content": tool_results})

        response = claude_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=500, system=SYSTEM, tools=TOOLS, messages=messages,
        )

    return "".join(b.text for b in response.content if b.type == "text")


def main():
    voyage_client = voyageai.Client()
    claude_client = anthropic.Anthropic()

    print(f"Indexing {len(DOCUMENTS)} documents with Voyage AI ({VOYAGE_MODEL})...")
    doc_vectors = embed_documents(voyage_client, DOCUMENTS)
    execute = make_tool_executor(voyage_client, doc_vectors)
    print("Ready. Two tools available: search_policies, add_business_days.\n")

    print("Try: 'If I request PTO today, what's the earliest day I could take off?'")
    print("Or:  'How many vacation days do I get?' (only needs one tool)")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question or question.lower() in {"quit", "exit"}:
            break

        messages = [{"role": "user", "content": question}]
        answer = run_agentic_loop(claude_client, messages, execute)
        print(f"\nClaude: {answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
