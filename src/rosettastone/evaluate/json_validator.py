"""JSON schema validation and field-level comparison."""

from __future__ import annotations

import json

from rosettastone.evaluate.base import Evaluator


class JSONEvaluator(Evaluator):
    def score(self, expected: str, actual: str) -> dict[str, float]:
        try:
            expected_obj = json.loads(expected)
            actual_obj = json.loads(actual)
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
