# Day 3: Prompt & Context Engineering

Two hands-on scripts. The first isolates five prompt engineering techniques
so you can see each one's effect on its own. The second connects "context
engineering" back to the RAG exercise from Session 1 — retrieval turns out
to be one specific technique for engineering context, not a separate topic.

## Setup

```bash
pip install -r requirements.txt
```

Same API keys as Session 1 and 2. Copy your already-configured `.env` from
the parent folder:

```bash
cp ../.env .env
```

(Or copy `.env.example` to `.env` and fill in your own keys if running this
standalone.)

## Part 1 — Prompt Engineering Fundamentals

```bash
python prompt_engineering_basics.py
```

Runs 5 before/after demos against the real API, back to back:

1. **Be clear and direct** — a vague prompt vs. a specific one
2. **Zero-shot vs. few-shot** — classification consistency with and without examples
3. **System prompts** — same question, different role/tone instruction
4. **Structured outputs** — asking nicely for JSON vs. schema-enforced JSON
5. **Chain of thought** — forcing an instant answer vs. asking Claude to reason first

Run just one demo instead of all five:

```bash
python prompt_engineering_basics.py 4    # structured outputs only
```

## Part 2 — Context Engineering

```bash
python context_engineering_exercise.py
```

Asks the same question three ways — no context, every document dumped in
regardless of relevance, and just the one relevant document found via the
same Voyage AI + cosine similarity retrieval from `rag_search.py` — and
prints the input token count for each. The point isn't that curated
context answers *better* here (with only 10 documents, bloated context
usually still gets the right answer) — it's that curated context does it
in a fraction of the tokens, and that gap becomes make-or-break at real
scale (thousands of documents instead of ten).

## Challenges to try

- **Part 1:** rewrite the vague prompt in demo 1 to be *maximally*
  specific — how much can you improve the output with prompting alone,
  no other changes?
- **Part 1, demo 2:** add a 4th example to the few-shot prompt for a
  category the model tends to get wrong — does one more example fix it?
- **Part 2:** add a question that genuinely needs *two* documents to
  answer fully (e.g., something touching both expenses and client
  communication) — does the curated (top-1) version now give an
  incomplete answer? What does that tell you about choosing `top_k`?
- **Part 2:** print the token count for the bloated-context version if
  the knowledge base had 500 documents instead of 10 (rough math, no
  code needed) — at what point does "just include everything" stop being
  an option at all?
