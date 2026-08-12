"""
The "hook" demo for the start of the RAG session: ask Claude the exact same
question with and without the retrieved policy document, side by side.
Meant to be run live and projected — no interaction needed.

Run:
    python demo_hallucination.py
"""

import anthropic
from dotenv import load_dotenv

from knowledge_base import DOCUMENTS

load_dotenv()

CLAUDE_MODEL = "claude-haiku-4-5"
QUESTION = "How many vacation days does a NorthStar Analytics employee get per year?"


def ask(client, prompt):
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main():
    client = anthropic.Anthropic()

    print(f"Question: {QUESTION}\n")

    print("=" * 60)
    print("WITHOUT retrieval — Claude has never seen this company's policy")
    print("=" * 60)
    print(ask(client, QUESTION))

    print("\n" + "=" * 60)
    print("WITH retrieval — grounded in the actual policy document")
    print("=" * 60)
    context = "\n\n".join(DOCUMENTS)
    grounded_prompt = (
        "Answer the question using ONLY the context below. If it's not in "
        "the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {QUESTION}"
    )
    print(ask(client, grounded_prompt))


if __name__ == "__main__":
    main()
