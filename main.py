#!/usr/bin/env python3
"""
elena-ai-agent — Google Form submission agent powered by Claude.

Usage:
  python main.py <FORM_URL> --context "..." [--dry-run] [--output result.json]
  python main.py <FORM_URL> --context-file data.txt
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
        description="Synthesize Google Form answers with Claude and submit them automatically.",
    )
    p.add_argument("form_url", help="Public Google Form viewform URL")

    ctx = p.add_mutually_exclusive_group(required=True)
    ctx.add_argument(
        "--context", "-c",
        metavar="TEXT",
        help="Context text the LLM will use to generate answers",
    )
    ctx.add_argument(
        "--context-file", "-f",
        metavar="PATH",
        help="Path to a file whose contents are used as context",
    )

    p.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    p.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Generate answers but do NOT submit the form",
    )
    p.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Write the result JSON to this file",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    if args.context_file:
        with open(args.context_file, encoding="utf-8") as fh:
            context = fh.read()
    else:
        context = args.context

    result = run_form_agent(
        form_url=args.form_url,
        context=context,
        api_key=api_key,
        model=args.model,
        dry_run=args.dry_run,
    )

    print("\n--- Result ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\nResult written to: {args.output}")


if __name__ == "__main__":
    main()
