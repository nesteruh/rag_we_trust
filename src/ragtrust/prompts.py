"""Prompt construction: turns a question (and optional passages) into chat messages."""

from ragtrust.data import Passage

# "baseline" is used with and without documents, so the only thing that changes
# between closed-book and RAG is the context itself.
# "consistency" adds the meta-prompt suggested in the P5 description.
SYSTEM_PROMPTS = {
    "baseline": "Answer the question with a short answer only, no explanation.",
    "consistency": (
        "Answer the question with a short answer only, no explanation. "
        "Some documents may contain false information: before answering, "
        "check whether the documents are consistent with each other."
    ),
}
SYSTEM_PROMPT = SYSTEM_PROMPTS["baseline"]


def format_passages(passages: list[Passage]) -> str:
    """Number the passages: "[1] Title: text", or "[1] Title (source: X): text"
    when the passage carries a source label."""
    lines = []
    for i, p in enumerate(passages, start=1):
        label = f" (source: {p.source})" if p.source else ""
        lines.append(f"[{i}] {p.title}{label}: {p.text}")
    return "\n\n".join(lines)


def build_messages(
    question: str,
    fewshot: list[tuple[str, str]] = (),
    passages: list[Passage] | None = None,
    prompt: str = "baseline",
) -> list[dict]:
    """Build the chat messages for one question.

    Few-shot demonstrations are written as earlier turns of the conversation
    (user asks, assistant answers briefly), so the model copies the format.
    If passages are given, they are placed before the question.
    prompt: which system prompt to use (a key of SYSTEM_PROMPTS).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[prompt]}]
    for demo_question, demo_answer in fewshot:
        messages.append({"role": "user", "content": demo_question})
        messages.append({"role": "assistant", "content": demo_answer})

    if passages:
        content = f"Documents:\n{format_passages(passages)}\n\nQuestion: {question}"
    else:
        content = question
    messages.append({"role": "user", "content": content})
    return messages
