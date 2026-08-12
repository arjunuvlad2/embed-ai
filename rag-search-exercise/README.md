# Session 2: Semantic Search & RAG

The follow-on to Session 1's vectorization exercise. Same embedding model,
same cosine similarity math — now used to retrieve relevant documents for a
question, so Claude answers grounded in real text instead of guessing.

## Setup

```bash
pip install -r requirements.txt
```

This exercise uses the same Voyage AI and Anthropic API keys as Session 1.
Copy your already-configured `.env` from the parent folder:

```bash
cp ../.env .env
```

(Or copy `.env.example` to `.env` here and fill in your own keys if you're
running this exercise standalone.)

## Run the hook demo first (instructor-led)

Shows the problem RAG solves — ask Claude a question about a private
company policy it has never seen, with and without retrieved context:

```bash
python demo_hallucination.py
```

## Then run the interactive exercise

```bash
python rag_search.py
```

Ask it things like:
- "How many vacation days do I get?"
- "Can I work from Bali for a month?"
- "Do I need approval to expense a $50 lunch?"
- "What happens if production goes down at 2am?"
- "What's the capital of France?" ← not in the knowledge base — watch it
  refuse to guess instead of making something up

## What it does

1. Loads 10 short internal policy documents (`knowledge_base.py`) — a
   fictional company's HR, engineering, and IT docs
2. Embeds all 10 documents once with Voyage AI (`input_type="document"`)
3. For each question you type, embeds the question itself
   (`input_type="query"` — a different, retrieval-tuned representation)
4. Ranks all documents by cosine similarity to the question and keeps the
   top 3
5. Builds a prompt: "answer using ONLY this context" + the top 3 documents
   + your question
6. Sends that to Claude Haiku and prints the grounded answer

## Challenges to try

- **Add an 11th document** to `knowledge_base.py` (e.g., a parental leave
  policy) and ask about it — no code changes needed, just re-run
- **Lower `TOP_K` to 1** in `rag_search.py` — does retrieval quality drop
  for questions that span two documents?
- **Ask a question with no answer in the knowledge base** — confirm Claude
  says so instead of inventing a policy
- **Compare models** — swap `CLAUDE_MODEL` to a larger model and see if the
  grounded answers change quality on ambiguous questions
