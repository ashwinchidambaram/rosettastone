"""PII Scanner: detect and flag personally identifiable information in prompts/responses."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rosettastone.core.types import PromptPair


@dataclass
class PIIWarning:
    """A PII finding for a single prompt pair."""

    pair_index: int
    pii_type: str
    severity: str
    count: int


# Regex patterns for PII detection
_PII_PATTERNS = {
    "email": (
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "MEDIUM",
    ),
    "us_phone": (
        r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "MEDIUM",
    ),
    "ssn": (
        r"\b\d{3}-\d{2}-\d{4}\b",
        "HIGH",
    ),
    "credit_card": (
        # NOTE: Matches structural 16-digit patterns only. No Luhn checksum validation.
        # False positive rate is high on technical data (order IDs, version strings, etc.).
        # Matches should be treated as candidate detections requiring manual review.
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "HIGH",
    ),
    "ipv4": (
        r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]"
        r"[0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?"
        r"[0-9][0-9]?)\b",
        "LOW",
    ),
}


def scan_text(text: str) -> list[tuple[str, str]]:
    """
    Scan a single text string for PII.

    Args:
        text: String to scan (prompt or response content)

    Returns:
        List of (pii_type, severity) tuples found in the text
    """
    findings = []
    for pii_type, (pattern, severity) in _PII_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            findings.append((pii_type, severity))
    return findings


def scan_pairs(pairs: list[PromptPair]) -> list[PIIWarning]:
    """
    Scan a list of PromptPair objects for PII in prompts and responses.

    Args:
        pairs: List of PromptPair objects to scan

    Returns:
        List of PIIWarning objects, one per (pair_index, pii_type) combination found
    """
    warnings = []

    for pair_idx, pair in enumerate(pairs):
        # Handle prompt: can be string or list of dicts
        prompt_text = ""
        if isinstance(pair.prompt, str):
            prompt_text = pair.prompt
        elif isinstance(pair.prompt, list):
            # Join all content from dict messages
            parts = []
            for item in pair.prompt:
                if isinstance(item, dict):
                    if "content" in item:
                        parts.append(str(item["content"]))
                    if "text" in item:
                        parts.append(str(item["text"]))
            prompt_text = " ".join(parts)

        # Scan prompt and response
        all_findings: dict[str, int] = {}

        for pii_type, severity in scan_text(prompt_text):
            key = pii_type
            all_findings[key] = all_findings.get(key, 0) + 1

        for pii_type, severity in scan_text(pair.response):
            key = pii_type
            all_findings[key] = all_findings.get(key, 0) + 1

        # Create one warning per unique pii_type found
        for pii_type, count in all_findings.items():
            severity = _PII_PATTERNS[pii_type][1]
            warnings.append(
                PIIWarning(
                    pair_index=pair_idx,
                    pii_type=pii_type,
                    severity=severity,
                    count=count,
                )
            )

    return warnings
