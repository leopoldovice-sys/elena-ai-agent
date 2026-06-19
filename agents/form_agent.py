import time
from typing import Any, Dict, List

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
    """POST one set of answers to Google Forms. Returns True on success."""
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
    # Google Forms redirects (302) to a thank-you page on success
    return response.status_code in (200, 302)


def _print_answers(form: GoogleForm, answers: Dict[str, Any]) -> None:
    for i, field in enumerate(form.fields, 1):
        value = answers.get(field.entry_id, "(skipped)")
        q_preview = field.question[:65] + ("…" if len(field.question) > 65 else "")
        print(f"         Q{i}: {q_preview}")
        print(f"              → {value}")


def run_form_agent(
    form_url: str,
    persona: str,
    *,
    count: int = 1,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
    delay: float = 1.5,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline: scrape once → synthesize N times with natural variation → submit N times.

    Args:
        form_url: Public Google Form viewform URL.
        persona:  Description of the persona/behavior Claude uses to answer.
        count:    Number of submissions to generate and send.
        api_key:  Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        model:    Claude model identifier.
        delay:    Seconds to wait between submissions (avoids rate-limiting).
        dry_run:  When True, synthesize answers but skip actual submission.

    Returns:
        A summary dict with per-submission results and overall stats.
    """
    print(f"[1/3] Fetching form …")
    form = scrape_form(form_url)
    print(f"      Title  : {form.title}")
    print(f"      Fields : {len(form.fields)} question(s) found")
    print(f"      Target : {count} submission(s)")

    submissions: List[Dict[str, Any]] = []
    success_count = 0

    for n in range(1, count + 1):
        print(f"\n[2/3] Submission {n}/{count} — synthesizing answers …")

        answers = synthesize_answers(
            form,
            persona,
            variation_index=n,
            total=count,
            api_key=api_key,
            model=model,
        )

        _print_answers(form, answers)

        human_answers = {f.question: answers.get(f.entry_id) for f in form.fields}
        entry = {"index": n, "answers": human_answers, "submitted": False}

        if dry_run:
            print(f"      [DRY RUN] skipping submission {n}")
        else:
            print(f"\n[3/3] Submitting {n}/{count} …")
            success = _submit(form, answers)
            entry["submitted"] = success

            if success:
                success_count += 1
                print(f"      ✓ Submitted.")
            else:
                print(f"      ✗ Unexpected response — check form manually.")

            if n < count:
                time.sleep(delay)

        submissions.append(entry)

    total_submitted = 0 if dry_run else success_count
    print(f"\n{'─' * 40}")
    if dry_run:
        print(f"DRY RUN complete — {count} response(s) generated, none submitted.")
    else:
        print(f"Done — {total_submitted}/{count} submitted successfully.")

    return {
        "form_title": form.title,
        "form_url": form_url,
        "persona": persona,
        "count": count,
        "submitted_count": total_submitted,
        "dry_run": dry_run,
        "submissions": submissions,
    }
