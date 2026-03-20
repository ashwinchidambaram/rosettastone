"""Prompt Auditor: detect verbatim substrings from training data in optimized prompts."""

from __future__ import annotations

from dataclasses import dataclass

from rosettastone.core.types import PromptPair

MIN_SUBSTRING_LENGTH = 30
BOILERPLATE_THRESHOLD = 0.1  # 10% of training data
MAX_BOILERPLATE_LENGTH = 50


@dataclass
class AuditFinding:
    """A verbatim substring match from training data in optimized prompt."""

    substring: str
    source_count: int
    is_boilerplate: bool


def audit_prompt(optimized_prompt: str, training_pairs: list[PromptPair]) -> list[AuditFinding]:
    """
    Audit an optimized prompt for verbatim substrings from training data.

    Finds all substrings of length >= MIN_SUBSTRING_LENGTH (30 chars) from
    training responses that appear in the optimized prompt. Filters out
    boilerplate (substrings appearing in >10% of training data AND <50 chars).

    Args:
        optimized_prompt: The optimized prompt to audit
        training_pairs: List of training PromptPair objects

    Returns:
        List of AuditFinding objects for non-boilerplate matches
    """
    if not training_pairs:
        return []

    findings: dict[str, int] = {}  # substring -> count in training data

    # Extract all substrings from training responses
    for pair in training_pairs:
        response = pair.response
        # Generate all substrings of length >= MIN_SUBSTRING_LENGTH
        for i in range(len(response) - MIN_SUBSTRING_LENGTH + 1):
            substring = response[i : i + MIN_SUBSTRING_LENGTH]
            findings[substring] = findings.get(substring, 0) + 1

    # Check which substrings appear in optimized prompt
    matched_findings = {}
    for substring, source_count in findings.items():
        if substring in optimized_prompt:
            matched_findings[substring] = source_count

    # Filter boilerplate: if substring appears in >10% of training data AND <50 chars
    boilerplate_threshold_count = len(training_pairs) * BOILERPLATE_THRESHOLD
    audit_results = []

    for substring, source_count in matched_findings.items():
        is_boilerplate = (
            source_count > boilerplate_threshold_count and len(substring) < MAX_BOILERPLATE_LENGTH
        )
        if not is_boilerplate:
            audit_results.append(
                AuditFinding(
                    substring=substring,
                    source_count=source_count,
                    is_boilerplate=False,
                )
            )

    return audit_results
