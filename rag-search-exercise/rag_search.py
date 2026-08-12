"""
Hands-on Session 2: Semantic Search & Retrieval-Augmented Generation (RAG).

Builds directly on Session 1's vectorization exercise: same embedding model,
same cosine similarity math — but now used to retrieve relevant documents
for a question, and Claude answers grounded in that retrieved text instead
of guessing.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # same VOYAGE_API_KEY / ANTHROPIC_API_KEY as Session 1

Run:
    python rag_search.py
"""

import os

import anthropic
import numpy as np
import voyageai
from dotenv import load_dotenv

from knowledge_base import DOCUMENTS

load_dotenv()

VOYAGE_MODEL = "voyage-3.5"
CLAUDE_MODEL = "claude-haiku-4-5"
TOP_K = 3


def embed_documents(voyage_client, texts):
    # input_type="document" tells Voyage these are things that will be searched
    result = voyage_client.embed(texts, model=VOYAGE_MODEL, input_type="document")
    return np.array(result.embeddings)


def embed_query(voyage_client, text):
    # input_type="query" uses a different internal representation, tuned for
    # short questions searching against longer documents
    result = voyage_client.embed([text], model=VOYAGE_MODEL, input_type="query")
    return np.array(result.embeddings[0])


def top_k_matches(query_vector, doc_vectors, k=TOP_K):
    doc_norms = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    query_norm = query_vector / np.linalg.norm(query_vector)
    scores = doc_norms @ query_norm
    ranked = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in ranked]


def build_prompt(question, retrieved_docs):
    context = "\n\n".join(f"[Document {i + 1}]\n{text}" for i, text in enumerate(retrieved_docs))
    return (
        "Answer the question using ONLY the context below. If the context "
        "doesn't contain the answer, say you don't have that information "
        "instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def ask_claude(claude_client, question, retrieved_docs):
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": build_prompt(question, retrieved_docs)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main():
    voyage_client = voyageai.Client()
    claude_client = anthropic.Anthropic()

    print(f"Indexing {len(DOCUMENTS)} documents with Voyage AI ({VOYAGE_MODEL})...")
    doc_vectors = embed_documents(voyage_client, DOCUMENTS)
    print("Index ready.\n")

    print("Ask a question about the NorthStar Analytics knowledge base.")
    print("Try: 'How many vacation days do I get?' or 'Can I work from Bali for a month?'")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if not question or question.lower() in {"quit", "exit"}:
            break

        query_vector = embed_query(voyage_client, question)
        matches = top_k_matches(query_vector, doc_vectors)

        print("\nTop matches (cosine similarity):")
        retrieved_texts = []
        for i, score in matches:
            print(f"  [{score:.3f}] {DOCUMENTS[i][:70]}...")
            retrieved_texts.append(DOCUMENTS[i])

        answer = ask_claude(claude_client, question, retrieved_texts)
        print(f"\nClaude (grounded in retrieved docs):\n{answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
