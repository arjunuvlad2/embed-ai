"""
Day 4, Part 1: Tool Use Fundamentals.

Four runnable demos building up the agentic loop one piece at a time:
Claude asking to use a tool, actually completing the loop, choosing
between multiple tools, and handling a tool that fails. Every demo calls
the real Claude API so you're watching the actual mechanics, not a
diagram of them.

Setup:
    pip install -r requirements.txt
    cp ../.env .env      # reuse the same ANTHROPIC_API_KEY

Run:
    python tool_use_basics.py          # all 4 demos
    python tool_use_basics.py 2         # just demo 2
"""

import re
import sys
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"

client = anthropic.Anthropic()


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------- tool defs

CALCULATE_TOOL = {
    "name": "calculate",
    "description": (
        "Evaluate a basic arithmetic expression (+, -, *, /, parentheses). "
        "Use this for any math instead of computing it yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A math expression, e.g. '12 * (4 + 3)'",
            }
        },
        "required": ["expression"],
    },
}

GET_TIME_TOOL = {
    "name": "get_current_time",
    "description": (
        "Get the current date and time. Use this whenever the question "
        "involves 'today', 'now', or the current date — you cannot know "
        "this on your own."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}


# --------------------------------------------------------- tool execution


def safe_calculate(expression):
    if not re.fullmatch(r"[\d\s.+\-*/()]+", expression):
        raise ValueError(f"Refusing to evaluate unsafe expression: {expression!r}")
    return eval(expression, {"__builtins__": {}}, {})  # noqa: S307 — chars whitelisted above


def execute_tool(name, tool_input):
    if name == "calculate":
        return safe_calculate(tool_input["expression"])
    if name == "get_current_time":
        return datetime.now().strftime("%A, %B %d, %Y at %H:%M")
    raise ValueError(f"Unknown tool: {name}")


def run_agentic_loop(messages, tools, max_iterations=5):
    """The core pattern: call Claude, and while it wants a tool, run the
    tool locally and hand the result back — until it produces a final
    answer instead of another tool_use block."""
    response = client.messages.create(model=MODEL, max_tokens=400, tools=tools, messages=messages)

    iterations = 0
    while response.stop_reason == "tool_use":
        iterations += 1
        if iterations > max_iterations:
            return "[stopped: too many tool calls in a row — likely stuck in a loop]"

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  Claude calls {block.name}({block.input})")
                try:
                    result = execute_tool(block.name, block.input)
                    print(f"    -> {result}")
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(result)})
                except Exception as e:
                    print(f"    -> ERROR: {e}")
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True})
        messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(model=MODEL, max_tokens=400, tools=tools, messages=messages)

    return "".join(b.text for b in response.content if b.type == "text")


# ------------------------------------------------------------------ demo 1


def demo_1_single_tool():
    header("DEMO 1 — Claude Asking to Use a Tool")

    question = "What is 847 * 23, minus 100?"
    print(f"Question: {question}\n")

    response = client.messages.create(
        model=MODEL, max_tokens=300, tools=[CALCULATE_TOOL],
        messages=[{"role": "user", "content": question}],
    )

    print(f"stop_reason: {response.stop_reason}\n")
    for block in response.content:
        if block.type == "text":
            print(f"[text block] {block.text}")
        elif block.type == "tool_use":
            print(f"[tool_use block] Claude wants to call '{block.name}' with input: {block.input}")

    print(
        "\nNotice: Claude didn't compute the answer itself — it stopped and "
        "asked to call a tool. stop_reason is 'tool_use', not 'end_turn'. "
        "Nothing has actually been calculated yet; that part is on us, "
        "which is exactly what Demo 2 adds."
    )


# ------------------------------------------------------------------ demo 2


def demo_2_agentic_loop():
    header("DEMO 2 — The Full Agentic Loop")

    question = "What is 847 * 23, minus 100?"
    print(f"Question: {question}\n")

    messages = [{"role": "user", "content": question}]
    final_answer = run_agentic_loop(messages, tools=[CALCULATE_TOOL])

    print(f"\nFinal answer: {final_answer}")
    print(
        "\nNotice: this is the whole pattern. Call Claude -> if it wants a "
        "tool, run the tool locally and send the result back as a new "
        "message -> repeat until stop_reason is 'end_turn'. Claude never "
        "ran any code itself — we did, and handed back the result."
    )


# ------------------------------------------------------------------ demo 3


def demo_3_multi_tool():
    header("DEMO 3 — Multiple Tools, Claude Picks")

    tools = [CALCULATE_TOOL, GET_TIME_TOOL]
    questions = ["What day is it today?", "What's 15% of 340?"]

    for question in questions:
        print(f"Question: {question}")
        messages = [{"role": "user", "content": question}]
        final_answer = run_agentic_loop(messages, tools=tools)
        print(f"Final answer: {final_answer}\n" + "-" * 70)

    print(
        "\nNotice: both questions had access to the exact same two tools. "
        "Claude decided which one (if either) the question actually "
        "needed — nothing in our code told it which tool to use."
    )


# ------------------------------------------------------------------ demo 4


def demo_4_error_handling():
    header("DEMO 4 — When a Tool Call Fails")

    question = "What is 10 divided by 0?"
    print(f"Question: {question}\n")

    messages = [{"role": "user", "content": question}]
    final_answer = run_agentic_loop(messages, tools=[CALCULATE_TOOL])

    print(f"\nFinal answer: {final_answer}")
    print(
        "\nNotice: dividing by zero makes our calculate() function raise "
        "an exception. We caught it and sent it back with is_error=True "
        "instead of letting the script crash. That flag tells Claude the "
        "TOOL failed, not that the answer is 'error' — so it can react "
        "sensibly (explain the problem, try something else) instead of "
        "quietly reporting the raw exception as if it were a real answer."
    )


DEMOS = {
    1: demo_1_single_tool,
    2: demo_2_agentic_loop,
    3: demo_3_multi_tool,
    4: demo_4_error_handling,
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
