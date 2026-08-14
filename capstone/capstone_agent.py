"""
Day 5 Capstone: a three-tool agent over the Mini Project 1 PDFs.

Every piece from the last four sessions shows up here: embeddings (Voyage
AI), retrieval (cosine similarity over PDF-extracted text), prompt/context
engineering (a tight system prompt, grounded-or-refuse instructions), and
tool use (an agentic loop choosing between three tools).

Uses rag_search_pdfs.py's retrieval by default. If you finished Mini
Project 1, swap in your own retrieval code instead — see the README's
"make it yours" section. That's the actual capstone: the same tool-use
pattern from Day 4, now sitting on top of work you built yourself.

Setup:
    pip install -r requirements.txt
    cp ../.env .env      # same VOYAGE_API_KEY / ANTHROPIC_API_KEY

Run:
    python capstone_agent.py
"""

from datetime import date, datetime, timedelta

import anthropic
import voyageai
from dotenv import load_dotenv

from rag_search_pdfs import embed_documents, embed_query, load_documents, top_k_matches

load_dotenv()

CLAUDE_MODEL = "claude-haiku-4-5"
MAX_TOOL_ITERATIONS = 5

SYSTEM = (
    "You are a helpful assistant for NorthStar Analytics employees. Use "
    "the search_policies tool for any question about company policy, "
    "process, or rules — do not answer those from memory. Use "
    "list_policy_titles if someone asks what policies exist, without "
    "asking about a specific one. Use add_business_days for any date math "
    f"involving business days instead of computing it yourself. Today's "
    f"date is {date.today().isoformat()}."
)

TOOLS = [
    {
        "name": "search_policies",
        "description": (
            "Search the NorthStar Analytics policy documents for text "
            "relevant to a question. Returns the top matching excerpts "
            "with similarity scores and source filenames."
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
        "name": "list_policy_titles",
        "description": (
            "List the filenames of every available policy document. Use "
            "this when someone asks what policies exist in general, "
            "rather than asking about one specific policy."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_business_days",
        "description": (
            "Add a number of business days (Mon-Fri, skipping weekends) "
            "to a date. Use this for anything involving 'business days' "
            "or 'working days' — never compute this yourself."
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


def add_business_days(start_date, business_days):
    d = datetime.strptime(start_date, "%Y-%m-%d")
    added = 0
    while added < business_days:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            added += 1
    return d.strftime("%Y-%m-%d (%A)")


def make_tool_executor(voyage_client, filenames, texts, doc_vectors):
    def execute(name, tool_input):
        if name == "search_policies":
            query_vector = embed_query(voyage_client, tool_input["query"])
            matches = top_k_matches(query_vector, doc_vectors, k=2)
            return "\n\n".join(f"[{filenames[i]}, score {score:.3f}] {texts[i]}" for i, score in matches)
        if name == "list_policy_titles":
            return ", ".join(filenames)
        if name == "add_business_days":
            return add_business_days(tool_input["start_date"], tool_input["business_days"])
        raise ValueError(f"Unknown tool: {name}")

    return execute


def run_agentic_loop(claude_client, messages, execute, on_tool_call=None):
    """on_tool_call(name) fires each time the agent calls a tool — optional
    hook used by eval_capstone.py to check which tools got used, without
    duplicating this loop."""
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
                if on_tool_call:
                    on_tool_call(block.name)
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

    filenames, texts = load_documents()
    print(f"Extracted text from {len(texts)} PDFs.")
    print("Indexing with Voyage AI...")
    doc_vectors = embed_documents(voyage_client, texts)
    execute = make_tool_executor(voyage_client, filenames, texts, doc_vectors)
    print("Ready. Three tools available: search_policies, list_policy_titles, add_business_days.\n")

    print("Try: 'What policies do you have?'")
    print("Or:  'If I request PTO today, what's the earliest day I could take off?'")
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
