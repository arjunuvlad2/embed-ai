"""
Day 3, Part 2: Context Engineering — what you put in the context window
matters as much as how you phrase the prompt.

Same question, three ways:
  1. NO CONTEXT     — Claude answers from training data alone
  2. BLOATED CONTEXT — every document dumped in, relevant or not
  3. CURATED CONTEXT — only the one relevant document, found the way
                        rag_search.py found it in Session 1 (Voyage AI +
                        cosine similarity)

This is the same knowledge base from the RAG exercise, reused on purpose:
retrieval is one specific technique for context engineering, not a
separate topic.

Setup:
    pip install -r requirements.txt
    cp ../.env .env      # reuse Session 1's VOYAGE_API_KEY / ANTHROPIC_API_KEY

Run:
    python context_engineering_exercise.py
"""

import anthropic
import numpy as np
import voyageai
from dotenv import load_dotenv

from knowledge_base import DOCUMENTS

load_dotenv()

VOYAGE_MODEL = "voyage-3.5"
CLAUDE_MODEL = "claude-haiku-4-5"

QUESTION = "How many vacation days does a NorthStar Analytics employee get per year?"

SYSTEM = (
    "Answer using ONLY the context provided, if any. If there is no "
    "context, or the context doesn't contain the answer, say you don't "
    "have that information instead of guessing."
)


def embed_documents(voyage_client, texts):
    result = voyage_client.embed(texts, model=VOYAGE_MODEL, input_type="document")
    return np.array(result.embeddings)


def embed_query(voyage_client, text):
    result = voyage_client.embed([text], model=VOYAGE_MODEL, input_type="query")
    return np.array(result.embeddings[0])


def most_relevant_doc(query_vector, doc_vectors):
    doc_norms = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    query_norm = query_vector / np.linalg.norm(query_vector)
    scores = doc_norms @ query_norm
    return int(np.argmax(scores))


def ask_with_context(claude_client, question, context_docs):
    if context_docs:
        context = "\n\n".join(f"[Document {i + 1}]\n{d}" for i, d in enumerate(context_docs))
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_content = question

    messages = [{"role": "user", "content": user_content}]

    token_count = claude_client.messages.count_tokens(
        model=CLAUDE_MODEL, system=SYSTEM, messages=messages
    ).input_tokens

    response = claude_client.messages.create(
        model=CLAUDE_MODEL, max_tokens=200, system=SYSTEM, messages=messages,
    )
    answer = "".join(block.text for block in response.content if block.type == "text")
    return answer, token_count


def run_comparison(voyage_client, claude_client, doc_vectors):
    print(f"Question: {QUESTION}\n")

    print("=" * 70)
    print("1. NO CONTEXT — Claude answers from training data alone")
    print("=" * 70)
    answer, tokens = ask_with_context(claude_client, QUESTION, [])
    print(f"Input tokens: {tokens}")
    print(f"Answer: {answer}\n")

    print("=" * 70)
    print(f"2. BLOATED CONTEXT — all {len(DOCUMENTS)} documents, relevant or not")
    print("=" * 70)
    answer, tokens = ask_with_context(claude_client, QUESTION, DOCUMENTS)
    print(f"Input tokens: {tokens}")
    print(f"Answer: {answer}\n")

    print("=" * 70)
    print("3. CURATED CONTEXT — just the 1 relevant document, found via retrieval")
    print("=" * 70)
    query_vector = embed_query(voyage_client, QUESTION)
    best_idx = most_relevant_doc(query_vector, doc_vectors)
    answer, tokens = ask_with_context(claude_client, QUESTION, [DOCUMENTS[best_idx]])
    print(f"Input tokens: {tokens}")
    print(f"Answer: {answer}\n")

    print(
        "Notice: options 2 and 3 both answer correctly here — but option 3 "
        "used a fraction of the tokens. At 10 documents that difference is "
        "small change; at 10,000 documents, dumping everything into context "
        "isn't just wasteful, it stops being possible at all. Curating "
        "context (retrieval) is what makes RAG scale."
    )


def main():
    voyage_client = voyageai.Client()
    claude_client = anthropic.Anthropic()

    print(f"Indexing {len(DOCUMENTS)} documents with Voyage AI ({VOYAGE_MODEL})...\n")
    doc_vectors = embed_documents(voyage_client, DOCUMENTS)

    run_comparison(voyage_client, claude_client, doc_vectors)


if __name__ == "__main__":
    main()
