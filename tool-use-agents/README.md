# Day 4: Tool Use & Agents

Two hands-on scripts. The first isolates the tool-use mechanics — Claude
asking to call a function, you running it, handing the result back — one
piece at a time. The second connects it back to Session 2: the RAG search
from `rag_search.py` becomes a *tool* Claude decides to call, instead of a
fixed pipeline stage that always runs, alongside a second tool for date
math.

## Setup

```bash
pip install -r requirements.txt
```

Same API keys as previous sessions. Copy your already-configured `.env`
from the parent folder:

```bash
cp ../.env .env
```

(Or copy `.env.example` to `.env` and fill in your own keys if running this
standalone.)

## Part 1 — Tool Use Fundamentals

```bash
python tool_use_basics.py
```

Runs 4 demos against the real API, back to back:

1. **A single tool call** — Claude asks to call `calculate`, and we stop
   right there to look at what that actually looks like (`stop_reason`,
   the `tool_use` content block) before running anything.
2. **The full agentic loop** — same question, but this time we actually
   execute the tool and send the result back, showing the whole
   call → tool → call-again pattern end to end.
3. **Multiple tools** — Claude has both `calculate` and
   `get_current_time` available and picks the right one per question.
4. **A failing tool** — dividing by zero, and how `is_error: true` in a
   `tool_result` lets Claude react sensibly instead of the script
   crashing or Claude reporting a raw exception as fact.

Run just one demo instead of all four:

```bash
python tool_use_basics.py 2    # the full agentic loop only
```

## Part 2 — A Small Agent (RAG as a Tool)

```bash
python rag_agent.py
```

An interactive agent with two tools:

- **`search_policies`** — the same Voyage AI + cosine similarity retrieval
  from `rag_search.py`, now wrapped as something Claude can choose to
  call rather than something that always runs
- **`add_business_days`** — adds N business days (skipping weekends) to a
  date

Try a question that only needs one tool:

```
You: How many vacation days do I get?
```

Then try one that genuinely needs both, in sequence:

```
You: If I request PTO today, what's the earliest day I could take off?
```

Watch the printed `[agent] calling ...` lines — Claude has to first look
up the PTO policy (5 business days' notice) *before* it can compute the
date, and it does that on its own without being told the exact steps.

## Challenges to try

- **Part 1:** add a third tool of your own (e.g. `word_count(text)`) and
  ask a question that needs it — does Claude pick it correctly among
  three options?
- **Part 1, demo 4:** make `get_current_time` raise an exception on
  purpose and see how Claude's final answer changes compared to a
  successful call.
- **Part 2:** ask a question that needs neither tool (e.g. "what's the
  capital of France?") — does the agent still answer, or does the system
  prompt make it try to search policies anyway? What does that tell you
  about how tightly to scope a tool's `description`?
- **Part 2:** lower `MAX_TOOL_ITERATIONS` to 1 and ask the two-tool PTO
  question again — what happens, and why is a hard cap like this a good
  idea even though it can cut off a legitimate multi-step answer?
