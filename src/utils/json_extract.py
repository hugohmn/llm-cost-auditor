"""Extract JSON from LLM responses that may wrap it in markdown."""

import json
import re


def extract_json(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks.

    LLMs often wrap JSON in ```json ... ``` blocks. This function
    strips those wrappers before parsing.
    """
    # Try raw text first
    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return stripped

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: find the first [ or { and extract to matching ] or }
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        if end > start:
            return text[start : end + 1]

    return stripped


def parse_json_response(text: str) -> list[dict[str, object]] | dict[str, object]:
    """Parse JSON from an LLM response, handling code blocks."""
    cleaned = extract_json(text)
    return json.loads(cleaned)
