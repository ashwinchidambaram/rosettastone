"""Property-based tests for PII scanner robustness."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from rosettastone.safety.pii_scanner import _PII_PATTERNS, scan_text

_KNOWN_PII_TYPES = frozenset(_PII_PATTERNS.keys())


@st.composite
def st_email(draw) -> str:
    user = draw(st.from_regex(r"[a-z]{3,8}", fullmatch=True))
    domain = draw(st.from_regex(r"[a-z]{3,8}", fullmatch=True))
    tld = draw(st.sampled_from(["com", "org", "net"]))
    return f"{user}@{domain}.{tld}"


@st.composite
def st_ssn(draw) -> str:
    area = draw(st.integers(min_value=100, max_value=999))
    group = draw(st.integers(min_value=10, max_value=99))
    serial = draw(st.integers(min_value=1000, max_value=9999))
    return f"{area}-{group}-{serial}"


@settings(max_examples=200)
@given(st.text(max_size=1000))
def test_scan_text_never_crashes(text):
    """scan_text must not raise for any string input and must return a list."""
    result = scan_text(text)
    assert isinstance(result, list)


@settings(max_examples=200)
@given(st.text(max_size=50), st_email(), st.text(max_size=50))
def test_scan_text_valid_emails_detected(prefix, email, suffix):
    """Emails embedded in text are detected and reported as type 'email'."""
    text = prefix + " " + email + " " + suffix
    results = scan_text(text)
    pii_types = {r[0] for r in results}
    assert "email" in pii_types, f"Expected 'email' in {pii_types!r} for text={text!r}"


@settings(max_examples=200)
@given(st.text(max_size=50), st_ssn(), st.text(max_size=50))
def test_scan_text_valid_ssns_detected(prefix, ssn, suffix):
    """SSNs embedded in text are detected and reported as type 'ssn'."""
    text = prefix + " " + ssn + " " + suffix
    results = scan_text(text)
    pii_types = {r[0] for r in results}
    assert "ssn" in pii_types, f"Expected 'ssn' in {pii_types!r} for text={text!r}"


@settings(max_examples=200)
@given(st.text(max_size=500))
def test_scan_text_counts_nonnegative(text):
    """All match counts in scan_text results are non-negative."""
    for _pii_type, _severity, match_count in scan_text(text):
        assert match_count >= 0


@settings(max_examples=200)
@given(st.text(max_size=500))
def test_scan_text_types_from_known_set(text):
    """All returned pii_type values are members of the known _PII_PATTERNS keys."""
    for pii_type, _severity, _count in scan_text(text):
        assert pii_type in _KNOWN_PII_TYPES, (
            f"Unexpected pii_type {pii_type!r} not in {_KNOWN_PII_TYPES}"
        )
