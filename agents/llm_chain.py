import re
import json
from typing import Any, Dict

import anthropic

from .form_scraper import GoogleForm


def _build_prompt(
    form: GoogleForm,
    persona: str,
    variation_index: int,
    total: int,
) -> str:
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
You are simulating a real person filling out a Google Form.

Form title: {form.title}
Form description: {form.description or "N/A"}

--- Persona ---
{persona}
--- End of persona ---

This is submission {variation_index} of {total}. You must respond as a distinct \
individual who genuinely matches the persona above. Introduce natural human variation:
- Vary phrasing and sentence structure in open-ended answers
- Slightly adjust numeric scores or ratings within what the persona would realistically choose
- Vary which optional items you emphasize or omit
- Do NOT make answers contradict the core persona — keep the identity consistent, \
just make each response feel like a different day or mood for the same type of person

Strict rules:
- For "multiple_choice" or "dropdown": reply with EXACTLY one of the listed options, verbatim.
- For "checkboxes": reply with a JSON list of selected options, e.g. ["Option A", "Option B"].
- For "short_text" / "paragraph": reply with a plain string.
- For "date": use YYYY-MM-DD format.
- For "time": use HH:MM 24 h format.
- For "linear_scale": reply with a single integer within the scale range.

{questions_block}

Respond ONLY with a valid JSON object. Keys are Q1, Q2, … matching the question numbers.
Example: {{"Q1": "Some answer", "Q2": ["Choice A"], "Q3": 7}}"""


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")
    return json.loads(match.group())


def synthesize_answers(
    form: GoogleForm,
    persona: str,
    *,
    variation_index: int = 1,
    total: int = 1,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> Dict[str, Any]:
    """
    Ask Claude to generate one set of form answers for the given persona.

    variation_index / total drive the natural-variation instruction so each
    call produces subtly different but persona-consistent answers.

    Returns a dict mapping entry_id → answer value (str or list[str]).
    """
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=1,  # max variation within the persona
        messages=[
            {
                "role": "user",
                "content": _build_prompt(form, persona, variation_index, total),
            }
        ],
    )

    raw = message.content[0].text
    answers_by_q = _extract_json(raw)

    entry_answers: Dict[str, Any] = {}
    for i, field in enumerate(form.fields, 1):
        q_key = f"Q{i}"
        if q_key not in answers_by_q:
            continue
        answer = answers_by_q[q_key]
        if not isinstance(answer, list):
            answer = str(answer)
        entry_answers[field.entry_id] = answer

    return entry_answers
