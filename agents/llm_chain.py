import re
import json
from typing import Any, Dict

import anthropic

from .form_scraper import FormField, GoogleForm


def _build_prompt(form: GoogleForm, context: str) -> str:
    lines = []
    for i, f in enumerate(form.fields, 1):
        line = f"Q{i}: {f.question}  [type: {f.field_type}]"
        if f.required:
            line += "  (required)"
        if f.options:
            line += f"\n     Options: {', '.join(f.options)}"
        lines.append(line)

    questions_block = "\n".join(lines)

    return f"""\
You are filling out a Google Form on behalf of a user.

Form title: {form.title}
Form description: {form.description or "N/A"}

--- Context to base your answers on ---
{context}
--- End of context ---

Answer every question below using ONLY information from the context above.
Keep answers concise and consistent with each other.

Rules:
- For "multiple_choice" or "dropdown": reply with EXACTLY one of the listed options, verbatim.
- For "checkboxes": reply with a JSON list of one or more options, e.g. ["Option A", "Option B"].
- For "short_text" / "paragraph": reply with a plain string.
- For "date": use YYYY-MM-DD format.
- For "time": use HH:MM format (24 h).
- For "linear_scale": reply with a single integer within the scale range.

{questions_block}

Respond ONLY with a valid JSON object. Keys are Q1, Q2, … matching the question numbers above.
Example format: {{"Q1": "Answer here", "Q2": ["Choice A"], "Q3": 4}}"""


def _extract_json(text: str) -> dict:
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")
    return json.loads(match.group())


def synthesize_answers(
    form: GoogleForm,
    context: str,
    *,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> Dict[str, Any]:
    """
    Ask Claude to generate answers for every field in the form.

    Returns a dict mapping entry_id → answer value (str or list[str]).
    """
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": _build_prompt(form, context)}],
    )

    raw = message.content[0].text
    answers_by_q = _extract_json(raw)

    entry_answers: Dict[str, Any] = {}
    for i, field in enumerate(form.fields, 1):
        q_key = f"Q{i}"
        if q_key not in answers_by_q:
            continue
        answer = answers_by_q[q_key]
        # Normalise to str for non-list types
        if not isinstance(answer, list):
            answer = str(answer)
        entry_answers[field.entry_id] = answer

    return entry_answers
