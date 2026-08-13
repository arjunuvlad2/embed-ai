"""
Day 3, Part 1: Prompt Engineering Fundamentals.

Five runnable demos, each isolating ONE technique so you can see its effect
in isolation: being clear and direct, few-shot examples, system prompts,
structured outputs, and chain-of-thought. Every demo calls the real Claude
API twice (before/after) so you're comparing actual model output, not a
canned example.

Setup:
    pip install -r requirements.txt
    cp ../.env .env      # reuse Session 1's ANTHROPIC_API_KEY

Run:
    python prompt_engineering_basics.py          # all 5 demos
    python prompt_engineering_basics.py 3         # just demo 3
"""

import sys

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"

client = anthropic.Anthropic()


def ask(prompt, system=None, max_tokens=400):
    kwargs = {"model": MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return "".join(block.text for block in response.content if block.type == "text")


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------ demo 1


def demo_1_clear_and_direct():
    header("DEMO 1 — Be Clear and Direct")

    vague = "Write something about our product."
    clear = (
        "Write a 3-sentence product description for a noise-cancelling "
        "headphone aimed at remote workers. Emphasize comfort during "
        "8-hour video calls. Do not mention price."
    )

    print("VAGUE PROMPT:\n  " + vague)
    print("\nResponse:\n" + ask(vague))

    print("\n" + "-" * 70)

    print("CLEAR & DIRECT PROMPT:\n  " + clear)
    print("\nResponse:\n" + ask(clear))

    print(
        "\nNotice: the vague prompt forces Claude to guess your product, "
        "audience, and tone. The clear prompt gets a usable answer on the "
        "first try — no back-and-forth needed."
    )


# ------------------------------------------------------------------ demo 2


def demo_2_few_shot():
    header("DEMO 2 — Zero-Shot vs. Few-Shot (Consistency)")

    test_sentences = [
        "The app crashes every time I try to upload a photo.",
        "Would be great if I could export my data as a CSV.",
        "Your support team fixed my issue in five minutes, amazing.",
    ]

    zero_shot = (
        "Classify this piece of customer feedback as bug_report, "
        "feature_request, or praise:\n\n\"" + test_sentences[0] + "\""
    )
    print("ZERO-SHOT PROMPT (no examples):\n  " + zero_shot)
    print("\nResponse:\n" + ask(zero_shot))

    print("\n" + "-" * 70)

    few_shot = (
        "Classify customer feedback into exactly one label: bug_report, "
        "feature_request, or praise. Respond with ONLY the label, nothing "
        "else.\n\n"
        "Feedback: \"It keeps logging me out every few minutes.\"\n"
        "Label: bug_report\n\n"
        "Feedback: \"Could you add dark mode?\"\n"
        "Label: feature_request\n\n"
        "Feedback: \"This saved me hours of work, thank you!\"\n"
        "Label: praise\n\n"
        f"Feedback: \"{test_sentences[0]}\"\n"
        "Label:"
    )
    print("FEW-SHOT PROMPT (3 examples, exact format shown):\n  [see code for full prompt]")
    print("\nResponse:\n" + ask(few_shot, max_tokens=20))

    print(
        "\nNotice: zero-shot often adds explanation, punctuation, or "
        "different casing — fine for a human, brittle if your code parses "
        "the output. Few-shot examples pin down the exact output format."
    )


# ------------------------------------------------------------------ demo 3


def demo_3_system_prompts():
    header("DEMO 3 — System Prompts (Setting Role and Tone)")

    question = "How do I fix a bug where my API returns a 401 error?"

    print("NO SYSTEM PROMPT:")
    print("\nResponse:\n" + ask(question))

    print("\n" + "-" * 70)

    system = (
        "You are a terse senior backend engineer reviewing a junior "
        "developer's question in Slack. Give a short, direct answer: at "
        "most 3 bullet points, no pleasantries, no explanations unless "
        "explicitly asked."
    )
    print(f"SYSTEM PROMPT:\n  \"{system}\"")
    print("\nResponse:\n" + ask(question, system=system))

    print(
        "\nNotice: same question, same model — the system prompt alone "
        "changed length, tone, and structure. This is the cheapest lever "
        "you have for making Claude's default behavior match your product."
    )


# ------------------------------------------------------------------ demo 4


def demo_4_structured_output():
    header("DEMO 4 — Structured Outputs (Reliable JSON)")

    email = (
        "Subject: Urgent - can't log in!!\n\n"
        "Hi, this is Priya Patel. I've been locked out of my account since "
        "yesterday and I have a client demo in 2 hours. Please help ASAP. "
        "My account email is priya.p@example.com."
    )

    print("SOURCE TEXT:\n" + email)

    schema_prompt = (
        "Extract the following from this support email as JSON with keys "
        "customer_name, email, urgency (low/medium/high), and summary "
        "(one sentence):\n\n" + email
    )
    print("\nWithout output_config (asking nicely for JSON in the prompt):")
    print(ask(schema_prompt))

    print("\n" + "-" * 70)
    print("With output_config.format (schema-enforced — always valid JSON):")

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": "Extract the customer's name, email, urgency, and a one-sentence summary from this support email:\n\n" + email}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "customer_name": {"type": "string"},
                        "email": {"type": "string"},
                        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
                        "summary": {"type": "string"},
                    },
                    "required": ["customer_name", "email", "urgency", "summary"],
                    "additionalProperties": False,
                },
            }
        },
    )
    print("".join(block.text for block in response.content if block.type == "text"))

    print(
        "\nNotice: asking nicely for JSON usually works, but 'usually' "
        "isn't good enough for code that calls json.loads() on the "
        "result. output_config.format guarantees the shape — no markdown "
        "fences to strip, no missing fields to handle."
    )


# ------------------------------------------------------------------ demo 5


def demo_5_chain_of_thought():
    header("DEMO 5 — Chain of Thought (Letting Claude Reason)")

    problem = (
        "A cafe sells coffee for $4 and pastries for $3. On Monday they "
        "sold 3 times as many coffees as pastries, and total revenue was "
        "$285. How many pastries did they sell?"
    )

    direct = problem + "\n\nRespond with ONLY the final number, nothing else."
    print("DIRECT-ANSWER PROMPT:")
    print("\nResponse:\n" + ask(direct, max_tokens=20))

    print("\n" + "-" * 70)

    cot = problem + "\n\nThink through this step by step before giving your final answer."
    print("CHAIN-OF-THOUGHT PROMPT (same problem, one instruction added):")
    print("\nResponse:\n" + ask(cot, max_tokens=400))

    print(
        "\nNotice: for multi-step problems, forcing an instant final "
        "answer skips the reasoning that would have caught a mistake. "
        "Asking Claude to show its work first tends to be more accurate, "
        "at the cost of a longer response. (Claude also has a native "
        "'thinking' mode for this on top of the prompting technique — "
        "see the slides.)"
    )


DEMOS = {
    1: demo_1_clear_and_direct,
    2: demo_2_few_shot,
    3: demo_3_system_prompts,
    4: demo_4_structured_output,
    5: demo_5_chain_of_thought,
}


def main():
    if len(sys.argv) > 1:
        demo_num = int(sys.argv[1])
        DEMOS[demo_num]()
    else:
        for demo in DEMOS.values():
            demo()


if __name__ == "__main__":
    main()
