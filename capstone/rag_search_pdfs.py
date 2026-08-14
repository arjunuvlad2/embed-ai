"""
Day 5 reference solution: RAG over the Mini Project 1 PDFs.

This is what Mini Project 1 asked you to build — the exact same retrieval
pattern as rag_search.py, with one new step in front of it: extracting
text from the 10 PDF files in mini-project-1-pdfs/ instead of importing a
Python list.

If you already finished Mini Project 1, use YOUR OWN code for today's
capstone instead of this file — that's the point of the "make it yours"
challenge. This script exists so nobody is blocked from today's hands-on
if Mini Project 1 isn't done yet.

Setup:
    pip install -r requirements.txt
    cp ../.env .env      # same VOYAGE_API_KEY / ANTHROPIC_API_KEY

Run:
    python rag_search_pdfs.py
"""

import glob
import os

import anthropic
import numpy as np
import voyageai
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

VOYAGE_MODEL = "voyage-3.5"
CLAUDE_MODEL = "claude-haiku-4-5"
TOP_K = 3
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "mini-project-1-pdfs")


def load_documents(pdf_dir=PDF_DIR):
    """Read every PDF in pdf_dir and return (filenames, texts) in sorted order."""
    paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not paths:
        raise FileNotFoundError(
            f"No PDFs found in {os.path.abspath(pdf_dir)!r}. "
            "Run this script from inside the capstone/ folder, with "
            "mini-project-1-pdfs/ as its sibling (../mini-project-1-pdfs)."
        )
    filenames, texts = [], []
    for path in paths:
        reader = PdfReader(path)
        text = reader.pages[0].extract_text()
        if not text or not text.strip():
            raise ValueError(f"{os.path.basename(path)} extracted as empty text — the PDF may be a scanned image with no selectable text.")
        filenames.append(os.path.basename(path))
        texts.append(text)
    return filenames, texts


def embed_documents(voyage_client, texts):
    result = voyage_client.embed(texts, model=VOYAGE_MODEL, input_type="document")
    return np.array(result.embeddings)


def embed_query(voyage_client, text):
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

    filenames, texts = load_documents()
    print(f"Extracted text from {len(texts)} PDFs: {', '.join(filenames)}\n")

    print(f"Indexing with Voyage AI ({VOYAGE_MODEL})...")
    doc_vectors = embed_documents(voyage_client, texts)
    print("Index ready.\n")

    print("Ask a question about the NorthStar Analytics policies.")
    print("Try: 'How many vacation days do I get?' or 'Can I expense a new monitor?'")
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
            print(f"  [{score:.3f}] {filenames[i]}: {texts[i][:60]}...")
            retrieved_texts.append(texts[i])

        answer = ask_claude(claude_client, question, retrieved_texts)
        print(f"\nClaude (grounded in retrieved docs):\n{answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
