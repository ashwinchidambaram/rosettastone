"""Output type detection logic."""

from __future__ import annotations

import json

from rosettastone.core.types import OutputType


def detect_output_type(response: str) -> OutputType:
    """Auto-detect the output type from response content."""
    response = response.strip()

    # Try JSON
    try:
        json.loads(response)
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
