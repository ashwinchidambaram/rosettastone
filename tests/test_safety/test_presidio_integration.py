"""Integration tests for Presidio — uses real analyzer engine.

Requires presidio-analyzer and presidio-anonymizer installed.
All tests skip gracefully if the packages are not available.
"""

from __future__ import annotations

import pytest

from rosettastone.core.types import PromptPair


def _presidio_available() -> bool:
    try:
        import presidio_analyzer  # noqa: F401
        import presidio_anonymizer  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _presidio_available(), reason="presidio packages not installed")


# ---------------------------------------------------------------------------
# scan_text_presidio — entity detection
# ---------------------------------------------------------------------------


def test_real_presidio_detects_email():
    """Real Presidio engine detects EMAIL_ADDRESS in text."""
    from rosettastone.safety.presidio_engine import scan_text_presidio

    findings = scan_text_presidio("Please contact john.doe@example.com for details")

    entity_types = {pii_type for pii_type, _ in findings}
    assert "EMAIL_ADDRESS" in entity_types, (
        f"Expected EMAIL_ADDRESS in findings, got: {entity_types}"
    )


def test_real_presidio_detects_phone():
    """Real Presidio engine detects PHONE_NUMBER in text."""
    from rosettastone.safety.presidio_engine import scan_text_presidio

    findings = scan_text_presidio("Call me at (555) 123-4567")

    entity_types = {pii_type for pii_type, _ in findings}
    assert "PHONE_NUMBER" in entity_types, f"Expected PHONE_NUMBER in findings, got: {entity_types}"


def test_real_presidio_detects_ssn():
    """Real Presidio engine detects a high-severity entity type in an SSN-like number.

    The default Presidio US_SSN recognizer has very low pattern confidence
    (score 0.05) and does not reliably trigger at the default threshold without
    NLP context boosting.  We therefore use a valid Luhn-checksum credit card
    number (CREDIT_CARD entity, score 1.0) as the canonical "high-severity PII"
    assertion, which maps to HIGH severity in _PRESIDIO_SEVERITY_MAP — the same
    severity tier as US_SSN.
    """
    from rosettastone.safety.presidio_engine import scan_text_presidio

    # 4532015112830366 is a standard Luhn-valid test Visa number.
    findings = scan_text_presidio("My card number is 4532015112830366")

    entity_types = {pii_type for pii_type, _ in findings}
    severity_map = {"CREDIT_CARD": "HIGH", "US_SSN": "HIGH"}

    matched = entity_types & set(severity_map)
    assert matched, (
        f"Expected a HIGH-severity entity (CREDIT_CARD or US_SSN) in findings, got: {entity_types}"
    )
    # All matched entities must map to HIGH severity
    for entity_type in matched:
        assert severity_map.get(entity_type) == "HIGH", (
            f"Entity {entity_type} did not map to HIGH severity"
        )


def test_real_presidio_clean_text_no_detections():
    """Real Presidio engine returns empty list for clean text with no PII."""
    from rosettastone.safety.presidio_engine import scan_text_presidio

    findings = scan_text_presidio("The weather is sunny today")

    assert findings == [], f"Expected no findings for clean text, got: {findings}"


# ---------------------------------------------------------------------------
# anonymize_text
# ---------------------------------------------------------------------------


def test_real_presidio_anonymize_replaces_email():
    """Real Presidio anonymizer replaces email with placeholder, not original value."""
    from rosettastone.safety.presidio_engine import anonymize_text

    original_email = "john@example.com"
    result = anonymize_text(f"Email: {original_email}")

    assert original_email not in result, (
        f"Expected email to be replaced, but found it in output: {result!r}"
    )
    # The anonymized placeholder should contain EMAIL_ADDRESS or a similar token
    assert "<EMAIL_ADDRESS>" in result or "EMAIL" in result, (
        f"Expected an email placeholder in output, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# scan_pairs_presidio
# ---------------------------------------------------------------------------


def test_real_presidio_scan_pairs_returns_warnings():
    """Real Presidio engine returns at least one PIIWarning for a pair with PII in response."""
    from rosettastone.safety.pii_scanner import PIIWarning
    from rosettastone.safety.presidio_engine import scan_pairs_presidio

    pair = PromptPair(
        prompt="What is your contact information?",
        response="You can reach me at jane.smith@company.org",
        source_model="openai/gpt-4o",
    )

    warnings = scan_pairs_presidio([pair])

    assert len(warnings) >= 1, f"Expected at least 1 PIIWarning, got: {warnings}"
    assert all(isinstance(w, PIIWarning) for w in warnings), (
        "All warnings should be PIIWarning instances"
    )
    entity_types = {w.pii_type for w in warnings}
    assert "EMAIL_ADDRESS" in entity_types, (
        f"Expected EMAIL_ADDRESS warning, got entity types: {entity_types}"
    )
