import re
import json
import httpx
from dataclasses import dataclass, field
from typing import List, Optional

FIELD_TYPES = {
    0: "short_text",
    1: "paragraph",
    2: "multiple_choice",
    3: "checkboxes",
    4: "dropdown",
    5: "linear_scale",
    7: "grid",
    9: "date",
    10: "time",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass
class FormField:
    entry_id: str
    question: str
    field_type: str
    required: bool
    options: List[str] = field(default_factory=list)


@dataclass
class GoogleForm:
    title: str
    description: str
    form_id: str
    fields: List[FormField]
    submit_url: str


def _extract_form_id(url: str) -> str:
    match = re.search(r"/forms/d/e/([^/]+)/", url)
    if not match:
        # Try the edit/non-published URL format
        match = re.search(r"/forms/d/([^/]+)/", url)
    if not match:
        raise ValueError(f"Cannot extract form ID from URL: {url}")
    return match.group(1)


def _load_form_data(html: str) -> list:
    marker = "FB_PUBLIC_LOAD_DATA_ = "
    start = html.find(marker)
    if start == -1:
        raise ValueError(
            "Could not find form data on the page. "
            "The form may be private, require sign-in, or the URL is incorrect."
        )
    start += len(marker)
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(html[start:])
    return data


def _parse_fields(form_data: list) -> List[FormField]:
    fields = []
    questions_block = form_data[1][1] if len(form_data[1]) > 1 else []

    for q in questions_block:
        if not q or len(q) < 5 or not isinstance(q[4], list) or not q[4]:
            continue

        question_text = q[1] or ""
        field_type_code = q[3] if len(q) > 3 else 0
        field_type = FIELD_TYPES.get(field_type_code, "text")

        entry_block = q[4][0]
        if not entry_block:
            continue

        entry_id = f"entry.{entry_block[0]}"

        # required flag lives at entry_block[2]
        required = bool(entry_block[2]) if len(entry_block) > 2 else False

        # options for choice-based fields live at entry_block[1]
        options: List[str] = []
        if field_type_code in (2, 3, 4) and len(entry_block) > 1 and entry_block[1]:
            options = [opt[0] for opt in entry_block[1] if opt and opt[0]]

        fields.append(
            FormField(
                entry_id=entry_id,
                question=question_text,
                field_type=field_type,
                required=required,
                options=options,
            )
        )

    return fields


def scrape_form(url: str) -> GoogleForm:
    """Fetch and parse a public Google Form, returning structured field metadata."""
    form_id = _extract_form_id(url)

    response = httpx.get(url, headers=HEADERS, follow_redirects=True, timeout=15)
    response.raise_for_status()

    data = _load_form_data(response.text)
    form_meta = data[1]

    title = form_meta[8] if len(form_meta) > 8 and form_meta[8] else "Untitled Form"
    description = form_meta[0] if form_meta[0] else ""

    fields = _parse_fields(data)

    return GoogleForm(
        title=title,
        description=description,
        form_id=form_id,
        fields=fields,
        submit_url=f"https://docs.google.com/forms/d/e/{form_id}/formResponse",
    )
