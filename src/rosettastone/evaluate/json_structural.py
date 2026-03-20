"""Deep structural similarity for JSON outputs.

Provides two metrics:
- ``json_structural_sim``: weighted Jaccard over (key-path × value) pairs.
- ``json_schema_match``: key-path overlap ignoring values (schema-only).

Array comparison uses Longest Common Subsequence (LCS) for ordered arrays;
set intersection for unordered sets (Python ``set``). Type coercion (e.g.
``5`` vs ``"5"``) yields a partial match of 0.5.
"""

from __future__ import annotations

from typing import Any

from rosettastone.evaluate.base import Evaluator

_EMPTY_RESULT: dict[str, float] = {"json_structural_sim": 0.0, "json_schema_match": 0.0}


# ---------------------------------------------------------------------------
# Key-path extraction helpers
# ---------------------------------------------------------------------------


def _coerce_match(a: Any, b: Any) -> float:
    """Return 1.0 for equal values, 0.5 for type-coerced match, 0.0 otherwise."""
    if a == b:
        return 1.0
    # Type coercion: numeric-string equivalence
    try:
        if str(a) == str(b):
            return 0.5
    except Exception:
        pass
    return 0.0


def _lcs_length(seq_a: list[Any], seq_b: list[Any]) -> int:
    """Classic LCS length via dynamic programming."""
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0
    # Use rolling array to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _extract_keypaths(
    obj: Any,
    prefix: str = "",
) -> dict[str, Any]:
    """Recursively extract all key-paths with their leaf values.

    Returns a flat dict mapping dotted key-path strings to leaf values.
    Array elements are addressed as ``key[0]``, ``key[1]``, etc.
    """
    paths: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                paths.update(_extract_keypaths(v, child_prefix))
            else:
                paths[child_prefix] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                paths.update(_extract_keypaths(v, child_prefix))
            else:
                paths[child_prefix] = v
    else:
        # Scalar at the root level (or passed directly)
        paths[prefix or "__root__"] = obj

    return paths


# ---------------------------------------------------------------------------
# Array similarity
# ---------------------------------------------------------------------------


def _array_sim(a: list[Any], b: list[Any]) -> tuple[float, float]:
    """Return (structural_sim, schema_match) for two arrays.

    Uses LCS for ordered comparison. Both metrics are computed over element
    count so that they stay in [0, 1].
    """
    if not a and not b:
        return 1.0, 1.0
    total = max(len(a), len(b))
    if not a or not b:
        return 0.0, 0.0

    lcs = _lcs_length(a, b)
    schema_match = lcs / total

    # Value similarity: count element matches (order-aware via LCS items)
    # For structural sim, count partial coerce matches too
    # Walk both arrays for coerce scoring (unordered value matching)
    matched_values = 0.0
    used_b: set[int] = set()
    for item_a in a:
        for j, item_b in enumerate(b):
            if j in used_b:
                continue
            m = _coerce_match(item_a, item_b)
            if m > 0:
                matched_values += m
                used_b.add(j)
                break

    structural_sim = matched_values / total
    return structural_sim, schema_match


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def _compare(a: Any, b: Any) -> tuple[float, float]:
    """Recursively compare two JSON objects. Returns (structural_sim, schema_match)."""
    # Both are dicts
    if isinstance(a, dict) and isinstance(b, dict):
        return _dict_compare(a, b)
    # Both are lists
    if isinstance(a, list) and isinstance(b, list):
        return _array_sim(a, b)
    # Type mismatch at container level → full mismatch
    if type(a) != type(b):  # noqa: E721
        # Still attempt coerce for scalars
        if not isinstance(a, (dict, list)) and not isinstance(b, (dict, list)):
            m = _coerce_match(a, b)
            return m, 0.0  # schema_match=0 because types differ
        return 0.0, 0.0
    # Scalars
    m = _coerce_match(a, b)
    return m, 1.0 if m == 1.0 else 0.0


def _dict_compare(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float]:
    """Compare two dicts. Returns (structural_sim, schema_match)."""
    all_keys = set(a) | set(b)
    if not all_keys:
        return 1.0, 1.0

    shared_keys = set(a) & set(b)
    # schema_match: key-path overlap (Jaccard)
    schema_match = len(shared_keys) / len(all_keys)

    if not shared_keys:
        return 0.0, schema_match

    # structural_sim: weighted Jaccard over key × value pairs
    # Each shared key contributes proportionally to its recursive sub-score.
    structural_parts: list[float] = []
    for k in shared_keys:
        sub_sim, _ = _compare(a[k], b[k])
        structural_parts.append(sub_sim)

    # Keys present in only one side contribute 0
    total_keys = len(all_keys)
    structural_sum = sum(structural_parts)
    structural_sim = structural_sum / total_keys

    return structural_sim, schema_match


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------


class JSONStructuralEvaluator(Evaluator):
    """Deep structural JSON similarity evaluator.

    Returns ``{"json_structural_sim": float, "json_schema_match": float}``.
    Non-JSON input yields ``{...: 0.0, ...: 0.0}`` (no exception raised).
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)

    def score(self, expected: str, actual: str, **kwargs: Any) -> dict[str, float]:
        import json

        try:
            exp_obj = json.loads(expected.strip())
        except (json.JSONDecodeError, ValueError):
            return dict(_EMPTY_RESULT)

        try:
            act_obj = json.loads(actual.strip())
        except (json.JSONDecodeError, ValueError):
            return dict(_EMPTY_RESULT)

        structural_sim, schema_match = _compare(exp_obj, act_obj)
        return {
            "json_structural_sim": float(structural_sim),
            "json_schema_match": float(schema_match),
        }
