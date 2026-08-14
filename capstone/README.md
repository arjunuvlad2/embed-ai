# Day 5: Capstone — Putting It All Together

Every piece from the last four sessions in one system: embeddings (Voyage
AI), retrieval (cosine similarity), prompt/context engineering (a tight
system prompt, grounded-or-refuse instructions), and tool use (an agentic
loop choosing between three tools) — running over the same PDF policy
documents from Mini Project 1. Then two more things a real Claude
application needs before you'd trust it: a way to check it's actually
right (`eval_capstone.py`), and a way to check it holds up when someone
tries to misuse it (`guardrails_check.py`).

## First: finish Tool Use & Agents

If yesterday's `tool-use-agents/` hands-on didn't fully wrap up, do that
before starting here — everything below assumes the agentic loop (call →
tool_use → execute → tool_result → repeat) is already solid.

## Setup

```bash
pip install -r requirements.txt
cp ../.env .env
```

(Or copy `.env.example` to `.env` and fill in your own keys if running this
standalone.)

## If you finished Mini Project 1

Great — skip straight to **Make It Yours** below and swap your own
retrieval code in.

## If you didn't finish Mini Project 1 yet

No problem, and nothing here is blocked by that. This folder includes a
complete reference solution so everyone has something real to build on
today:

```bash
python rag_search_pdfs.py
```

This is the RAG-over-PDFs pattern Mini Project 1 asked for — the same
retrieval logic as `rag_search.py`, with one new step in front of it:
extracting text from the 10 PDFs in `mini-project-1-pdfs/` with `pypdf`
instead of importing a Python list.

## The Capstone Agent

```bash
python capstone_agent.py
```

An interactive agent with three tools:

- **`search_policies`** — the retrieval from `rag_search_pdfs.py`, wrapped
  as a tool (same pattern as `rag_agent.py` from Day 4)
- **`list_policy_titles`** — a zero-argument tool that just enumerates
  what's available, for questions like "what policies do you have?"
- **`add_business_days`** — the same date-math tool from Day 4, reused
  because it's genuinely useful again in this domain

Try:

```
You: What policies do you have?
You: If I request PTO today, what's the earliest day I could take off?
You: Can I expense a new monitor?
```

Watch the `[agent] calling ...` lines — the PTO question needs
`search_policies` and `add_business_days` in sequence, exactly like
Day 4, except now the knowledge base came from files you (or the
reference script) had to parse first.

## Make It Yours

This is the actual capstone exercise: swap `capstone_agent.py`'s import
of `rag_search_pdfs` for **your own Mini Project 1 code**. If your
retrieval function has a different name or signature, adjust the calls
in `make_tool_executor` accordingly. Once it's wired up, everything else
— the tool loop, the other two tools, the system prompt — keeps working
unchanged, because tool use doesn't care where `search_policies`'s
answer came from, only that it returns a string.

## Eval Lab: Is It Actually Right?

```bash
python eval_capstone.py
```

An eval is just a list of questions with a known-good answer, run
automatically, and scored. `GOLDEN_SET` has 6 questions, each checked
against **both** the tool(s) the agent should have called and a keyword
that should appear in its final answer. Run it against your agent and
read the scorecard — a failure is a better learning moment than a clean
pass, because it forces you to ask why: wrong tool called, or a correct
answer phrased differently than the keyword check expected?

Try adding one question of your own to `GOLDEN_SET` and rerunning.

## Guardrails Lab: Is It Safe?

```bash
python guardrails_check.py
```

Four adversarial questions — a system-prompt extraction attempt, a
role-override attempt, a request to fabricate personal data, and a
question genuinely outside the knowledge base — fired at the agent. Each
response is then graded by a second Claude call acting as a judge: given
the question, the agent's answer, and a plain-English safety criterion,
it returns PASS or FAIL with a reason. This is the **LLM-as-judge**
pattern — just another `messages.create()` call, pointed at grading
instead of answering.

A FAIL isn't a broken agent — a two-sentence system prompt on a training
exercise is supposed to be somewhat breakable. The point is having a
repeatable way to find out.

## Challenges to try

- Add a fourth tool of your own. A few ideas: `word_count(policy_name)`,
  `policies_mentioning(keyword)` (a plain substring search, no embeddings
  — a good contrast with `search_policies`), or a summarizer that makes
  a second Claude call internally.
- Ask something that needs all three tools in one turn. What's the
  simplest question you can write that forces that?
- Lower `MAX_TOOL_ITERATIONS` and see which of your test questions break
  first — is it the ones you'd expect?
- Swap `add_business_days` to also skip a short list of company holidays,
  not just weekends — a small, realistic extension of a tool you already
  understand.
- Add a new `TEST_CASES` entry to `guardrails_check.py` — try to think of
  a question that would trick a *less* carefully instructed agent.
- Lower the eval's bar: change one `expect_keyword` to something the
  agent's answer is unlikely to contain, and watch it correctly fail —
  a good way to confirm the eval script itself isn't just always passing.
