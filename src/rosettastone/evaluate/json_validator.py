"""JSON schema validation and field-level comparison."""

from __future__ import annotations

import json
import re
from typing import Any

from rosettastone.evaluate.base import Evaluator


def _strip_fences(text: str) -> str:
    """Strip markdown code fences and thinking prefixes before JSON parsing.

    Handles:
    - Plain JSON response
    - JSON wrapped in ```json ... ``` at start of response
    - Thinking prefix + JSON fence block (e.g. Qwen CoT mode)
    - Thinking prefix + bare JSON object at end
    """
    text = text.strip()
    # 1. Whole response is a fence block (simple case)
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 2. Find any JSON fence block in the text (handles thinking prefix)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # 3. Find the last bare JSON object or array in the text
    for bracket in ("{", "["):
        last_pos = text.rfind(bracket)
        if last_pos != -1:
            candidate = text[last_pos:]
            if _looks_like_json(candidate):
                return candidate
    return text


def _looks_like_json(text: str) -> bool:
    """Return True if text can be parsed as JSON."""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


class JSONEvaluator(Evaluator):
    def score(self, expected: str, actual: str, **kwargs: Any) -> dict[str, float]:
        try:
            expected_obj = json.loads(_strip_fences(expected))
            actual_obj = json.loads(_strip_fences(actual))
        except json.JSONDecodeError:
            return {"json_valid": 0.0, "json_field_match": 0.0}

        # Both are valid JSON
        valid_score = 1.0

        # Field-level comparison (for dicts)
        if isinstance(expected_obj, dict) and isinstance(actual_obj, dict):
            expected_keys = set(expected_obj.keys())
            actual_keys = set(actual_obj.keys())
            if expected_keys:
                key_overlap = len(expected_keys & actual_keys) / len(expected_keys)
                # Check value matches for shared keys
                shared = expected_keys & actual_keys
                value_matches = sum(1 for k in shared if expected_obj[k] == actual_obj[k])
                value_score = value_matches / max(len(shared), 1)
                field_match = (key_overlap + value_score) / 2
            else:
                field_match = 1.0 if not actual_keys else 0.5
        else:
            # Non-dict JSON (arrays, primitives) — check equality
            field_match = 1.0 if expected_obj == actual_obj else 0.0

        return {"json_valid": valid_score, "json_field_match": field_match}
