from typing import Any, Dict

import httpx

from .form_scraper import GoogleForm, scrape_form
from .llm_chain import synthesize_answers

_SUBMIT_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _submit(form: GoogleForm, answers: Dict[str, Any]) -> bool:
    """POST form answers to Google Forms. Returns True on success."""
    # Build flat list of (key, value) tuples to support multi-value checkboxes
    pairs = []
    for entry_id, value in answers.items():
        if isinstance(value, list):
            for item in value:
                pairs.append((entry_id, item))
        else:
            pairs.append((entry_id, value))

    headers = {
        **_SUBMIT_HEADERS,
        "Referer": f"https://docs.google.com/forms/d/e/{form.form_id}/viewform",
    }

    response = httpx.post(
        form.submit_url,
        data=pairs,
        headers=headers,
        follow_redirects=False,
        timeout=15,
    )
    # Google Forms redirects to a "thank you" page (302) on success
    return response.status_code in (200, 302)


def run_form_agent(
    form_url: str,
    context: str,
    *,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline: scrape → synthesize → submit.

    Args:
        form_url: Public Google Form viewform URL.
        context:  Text the LLM uses to derive consistent answers.
        api_key:  Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        model:    Claude model identifier.
        dry_run:  When True, skip the actual form submission.

    Returns:
        A result dict with keys: form_title, form_url, answers, submitted.
    """
    print(f"[1/3] Fetching form …")
    form = scrape_form(form_url)
    print(f"      Title  : {form.title}")
    print(f"      Fields : {len(form.fields)} question(s) found")

    print(f"\n[2/3] Synthesizing answers with Claude ({model}) …")
    answers = synthesize_answers(form, context, api_key=api_key, model=model)

    print("\n      Generated answers:")
    for i, field in enumerate(form.fields, 1):
        value = answers.get(field.entry_id, "(skipped)")
        q_preview = field.question[:70] + ("…" if len(field.question) > 70 else "")
        print(f"      Q{i}: {q_preview}")
        print(f"           → {value}")

    human_answers = {f.question: answers.get(f.entry_id) for f in form.fields}

    result: Dict[str, Any] = {
        "form_title": form.title,
        "form_url": form_url,
        "answers": human_answers,
        "submitted": False,
    }

    if dry_run:
        print("\n[3/3] DRY RUN — submission skipped.")
        return result

    print("\n[3/3] Submitting form …")
    success = _submit(form, answers)
    result["submitted"] = success

    if success:
        print("      ✓ Submitted successfully.")
    else:
        print("      ✗ Submission returned an unexpected status. Check the form manually.")

    return result
