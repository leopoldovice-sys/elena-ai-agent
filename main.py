#!/usr/bin/env python3
"""
elena-ai-agent — Google Form submission agent powered by Claude.

Usage examples:
  python main.py <FORM_URL> --persona "..." --count 10
  python main.py <FORM_URL> --persona-file persona.txt --count 5 --dry-run
  python main.py <FORM_URL> --persona "..." --count 20 --output results.json
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agents.form_agent import run_form_agent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scrape a Google Form, synthesize N persona-consistent answers with Claude, "
            "and submit them automatically with natural variation between each response."
        ),
    )
    p.add_argument("form_url", help="Public Google Form viewform URL")

    persona_group = p.add_mutually_exclusive_group(required=True)
    persona_group.add_argument(
        "--persona", "-p",
        metavar="TEXT",
        help='Persona description, e.g. "35-year-old UX designer, enthusiastic about AI"',
    )
    persona_group.add_argument(
        "--persona-file", "-f",
        metavar="PATH",
        help="Path to a file containing the persona description",
    )

    p.add_argument(
        "--count", "-N",
        type=int,
        default=1,
        metavar="N",
        help="Number of form submissions to generate (default: 1)",
    )
    p.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    p.add_argument(
        "--delay", "-d",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help="Seconds to wait between submissions (default: 1.5)",
    )
    p.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Synthesize answers but do NOT submit the form",
    )
    p.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Write the full result JSON to this file",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    if args.count < 1:
        print("Error: --count must be at least 1.", file=sys.stderr)
        sys.exit(1)

    if args.persona_file:
        with open(args.persona_file, encoding="utf-8") as fh:
            persona = fh.read().strip()
    else:
        persona = args.persona

    result = run_form_agent(
        form_url=args.form_url,
        persona=persona,
        count=args.count,
        api_key=api_key,
        model=args.model,
        delay=args.delay,
        dry_run=args.dry_run,
    )

    print("\n--- Summary ---")
    summary = {k: v for k, v in result.items() if k != "submissions"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\nFull results written to: {args.output}")


if __name__ == "__main__":
    main()
