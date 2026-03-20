"""Output type detection logic."""

from __future__ import annotations

import json
import re

from rosettastone.core.types import OutputType

# Matches ```json ... ``` or ``` ... ``` fenced blocks
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

_CLASSIFY_KEYWORDS = frozenset({"classify", "label", "categorize", "categorise", "tag"})


def _strip_code_fence(text: str) -> str:
    """Remove optional ```json ... ``` or ``` ... ``` markdown wrapper."""
    m = _FENCE_RE.match(text.strip())
    if m:
        return m.group(1).strip()
    return text


def detect_output_type(response: str, *, prompt: str | None = None) -> OutputType:
    """Auto-detect the output type from response content.

    Args:
        response: The model's response text.
        prompt: Optional original prompt. Used to apply classification overrides
            when the prompt contains task keywords like "classify" or "label".
    """
    response = response.strip()

    # Empty or whitespace-only responses default to SHORT_TEXT
    # (not CLASSIFICATION — empty string is not a label)
    if not response:
        return OutputType.SHORT_TEXT

    # Try JSON (including markdown-fenced JSON)
    candidate = _strip_code_fence(response)
    try:
        parsed = json.loads(candidate)
        # Numeric-primitive JSON + classify/label prompt → CLASSIFICATION override
        if isinstance(parsed, (int, float)) and prompt is not None:
            prompt_lower = prompt.lower()
            if any(kw in prompt_lower for kw in _CLASSIFY_KEYWORDS):
                return OutputType.CLASSIFICATION
        return OutputType.JSON
    except (json.JSONDecodeError, ValueError):
        pass

    # Classification: short, single-line, few words
    words = response.split()
    if len(words) <= 5 and "\n" not in response:
        return OutputType.CLASSIFICATION

    # Short text vs long text
    if len(words) <= 50:
        return OutputType.SHORT_TEXT

    return OutputType.LONG_TEXT
