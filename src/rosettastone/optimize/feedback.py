"""Feedback utilities for prompt optimization.

Provides:
- build_feedback_map: index known-issue feedback from a training set for O(1) lookup.
- prepend_feedback: prepend a known-issue string to base metric feedback without
  altering scores (preserves the [0, 1] score range).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rosettastone.core.types import PromptPair


def build_feedback_map(train_set: list[PromptPair]) -> dict[str, str]:
    """Build a prompt-keyed map of known-issue feedback strings.

    Only pairs that have a non-None feedback value are included. The map is
    intended to be built once before optimization and used for O(1) lookup
    inside the metric function.

    Key encoding:
    - str prompt  → the prompt string as-is
    - list prompt → json.dumps(prompt, sort_keys=True)

    Args:
        train_set: Training prompt/response pairs, some of which may carry feedback.

    Returns:
        A dict mapping encoded prompt keys to feedback strings.
    """
    feedback_map: dict[str, str] = {}
    for pair in train_set:
        if pair.feedback is None:
            continue
        if isinstance(pair.prompt, str):
            key = pair.prompt
        else:
            key = json.dumps(pair.prompt, sort_keys=True)
        feedback_map[key] = pair.feedback
    return feedback_map


def prepend_feedback(base_feedback: str, known_issue: str | None) -> str:
    """Prepend a known-issue string to the base metric feedback.

    This is pure text prepending — it does not touch scores, so the [0, 1]
    score range is preserved.  The KNOWN ISSUE prefix makes the source of the
    extra context visible to the optimizer's reflection step.

    Args:
        base_feedback: The feedback string produced by the metric function.
        known_issue: An optional known-issue string sourced from the feedback map.
                     When None the base_feedback is returned unchanged.

    Returns:
        The (possibly augmented) feedback string.
    """
    if known_issue is not None:
        return f"KNOWN ISSUE: {known_issue}\n{base_feedback}"
    return base_feedback
