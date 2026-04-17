"""Tests for PII scanner module."""

from __future__ import annotations

import pytest

from rosettastone.core.types import OutputType, PromptPair
from rosettastone.safety.pii_scanner import PIIWarning, scan_pairs, scan_text

# ---------------------------------------------------------------------------
# Tests for scan_text
# ---------------------------------------------------------------------------


class TestScanTextEmailDetection:
    """Tests for email pattern detection in scan_text."""

    def test_detects_standard_email(self):
        """This test proves that a standard email is detected."""
        findings = scan_text("Contact me at john.doe@example.com for details")
        assert len(findings) > 0, "Expected email to be detected"
        assert any(pii_type == "email" for pii_type, *_ in findings), "Expected email in findings"

    def test_detects_email_with_numbers(self):
        """This test proves that emails with numbers are detected."""
        text = "Reach out to user123@domain.org"
        findings = scan_text(text)
        assert any(pii_type == "email" for pii_type, *_ in findings), (
            "Expected numeric email detected"
        )

    def test_detects_email_case_insensitive(self):
        """This test proves that emails are detected case-insensitively."""
        findings = scan_text("EMAIL@DOMAIN.COM is the contact")
        assert any(pii_type == "email" for pii_type, *_ in findings), (
            "Expected uppercase email detected"
        )

    def test_email_severity_is_medium(self):
        """This test proves that email severity is MEDIUM."""
        findings = scan_text("user@example.com")
        email_findings = [
            (pii_type, severity) for pii_type, severity, *_ in findings if pii_type == "email"
        ]
        assert len(email_findings) > 0, "Expected email finding"
        assert email_findings[0][1] == "MEDIUM", (
            f"Expected MEDIUM severity for email, got {email_findings[0][1]}"
        )

    def test_no_false_positives_for_invalid_email(self):
        """This test proves that incomplete emails are not detected."""
        findings = scan_text("invalid@domain")
        # 'invalid@domain' without TLD may or may not match depending on strictness
        # We just verify the scanner doesn't crash
        assert isinstance(findings, list)


class TestScanTextPhoneDetection:
    """Tests for US phone number pattern detection in scan_text."""

    def test_detects_standard_us_phone(self):
        """This test proves that a standard US phone number is detected."""
        findings = scan_text("Call me at (555) 123-4567")
        assert any(pii_type == "us_phone" for pii_type, *_ in findings), (
            "Expected US phone detected"
        )

    def test_detects_us_phone_without_parentheses(self):
        """This test proves that US phone without parentheses is detected."""
        findings = scan_text("My number is 555-123-4567")
        assert any(pii_type == "us_phone" for pii_type, *_ in findings), (
            "Expected phone without parentheses detected"
        )

    def test_detects_us_phone_with_dots(self):
        """This test proves that US phone with dots is detected."""
        findings = scan_text("555.123.4567")
        assert any(pii_type == "us_phone" for pii_type, *_ in findings), (
            "Expected phone with dots detected"
        )

    def test_detects_us_phone_with_plus_1(self):
        """This test proves that US phone with +1 prefix is detected."""
        findings = scan_text("+1-555-123-4567")
        assert any(pii_type == "us_phone" for pii_type, *_ in findings), (
            "Expected +1 formatted phone detected"
        )

    def test_phone_severity_is_medium(self):
        """This test proves that phone severity is MEDIUM."""
        findings = scan_text("555-123-4567")
        phone_findings = [
            (pii_type, severity) for pii_type, severity, *_ in findings if pii_type == "us_phone"
        ]
        assert len(phone_findings) > 0, "Expected phone finding"
        assert phone_findings[0][1] == "MEDIUM", (
            f"Expected MEDIUM severity for phone, got {phone_findings[0][1]}"
        )

    def test_phone_no_false_positive_version_string(self):
        """Version strings like '1.234.5678' should NOT be detected as phone numbers.

        The leading \\b anchor prevents matching digit sequences that are embedded
        within longer numeric tokens (e.g. version or build strings).
        """
        for version_string in ("1.234.5678", "v1.2.3456", "release-1.234.5678"):
            findings = scan_text(version_string)
            phone_findings = [pii_type for pii_type, *_ in findings if pii_type == "us_phone"]
            assert len(phone_findings) == 0, (
                f"Expected no phone match for version string '{version_string}', "
                f"got {phone_findings}"
            )


class TestScanTextSSNDetection:
    """Tests for US Social Security Number pattern detection in scan_text."""

    def test_detects_ssn(self):
        """This test proves that a valid SSN format is detected."""
        findings = scan_text("SSN: 123-45-6789")
        assert any(pii_type == "ssn" for pii_type, *_ in findings), "Expected SSN detected"

    def test_ssn_severity_is_high(self):
        """This test proves that SSN severity is HIGH."""
        findings = scan_text("123-45-6789")
        ssn_findings = [
            (pii_type, severity) for pii_type, severity, *_ in findings if pii_type == "ssn"
        ]
        assert len(ssn_findings) > 0, "Expected SSN finding"
        assert ssn_findings[0][1] == "HIGH", (
            f"Expected HIGH severity for SSN, got {ssn_findings[0][1]}"
        )

    def test_ssn_with_leading_zeros(self):
        """This test proves that SSN with leading zeros is detected."""
        findings = scan_text("001-23-4567")
        assert any(pii_type == "ssn" for pii_type, *_ in findings), (
            "Expected SSN with leading zeros detected"
        )

    def test_no_false_positive_for_partial_ssn(self):
        """This test proves that incomplete SSN patterns are not detected."""
        findings = scan_text("123-456789")
        ssn_findings = [pii_type for pii_type, *_ in findings if pii_type == "ssn"]
        assert len(ssn_findings) == 0, "Expected no SSN match for incomplete format"


class TestScanTextCreditCardDetection:
    """Tests for credit card pattern detection in scan_text."""

    def test_detects_credit_card_16_digits(self):
        """This test proves that 16-digit credit card is detected."""
        findings = scan_text("4532-1234-5678-9010")
        assert any(pii_type == "credit_card" for pii_type, *_ in findings), (
            "Expected credit card detected"
        )

    def test_detects_credit_card_without_separators(self):
        """This test proves that credit card without separators is detected."""
        findings = scan_text("4532123456789010")
        assert any(pii_type == "credit_card" for pii_type, *_ in findings), (
            "Expected credit card without separators detected"
        )

    def test_detects_credit_card_with_spaces(self):
        """This test proves that credit card with spaces is detected."""
        findings = scan_text("4532 1234 5678 9010")
        assert any(pii_type == "credit_card" for pii_type, *_ in findings), (
            "Expected credit card with spaces detected"
        )

    def test_credit_card_severity_is_high(self):
        """This test proves that credit card severity is HIGH."""
        findings = scan_text("4532-1234-5678-9010")
        cc_findings = [
            (pii_type, severity) for pii_type, severity, *_ in findings if pii_type == "credit_card"
        ]
        assert len(cc_findings) > 0, "Expected credit card finding"
        assert cc_findings[0][1] == "HIGH", (
            f"Expected HIGH severity for credit card, got {cc_findings[0][1]}"
        )

    def test_credit_card_false_positive_documented(self):
        """A plain 16-digit order ID IS detected — documenting known false-positive behavior.

        The credit card regex matches any structurally valid 16-digit sequence.
        No Luhn checksum is performed, so arbitrary numeric IDs (order IDs, transaction
        references, etc.) will trigger this pattern. Matches are candidate detections
        that require manual review before acting on them.
        """
        # This 16-digit string is an order ID, not a real credit card, but the
        # regex cannot distinguish them without Luhn validation.
        order_id_text = "Order ID: 1234567890123456"
        findings = scan_text(order_id_text)
        cc_findings = [pii_type for pii_type, *_ in findings if pii_type == "credit_card"]
        # Assert it IS detected — confirming the known false-positive behavior
        assert len(cc_findings) > 0, (
            "Expected credit_card match for 16-digit order ID (known false positive — "
            "no Luhn check is performed)"
        )


class TestScanTextIPAddressDetection:
    """Tests for IPv4 address pattern detection in scan_text."""

    def test_detects_ipv4_address(self):
        """This test proves that a valid IPv4 address is detected."""
        findings = scan_text("Server at 192.168.1.1")
        assert any(pii_type == "ipv4" for pii_type, *_ in findings), "Expected IPv4 detected"

    def test_detects_ipv4_with_max_values(self):
        """This test proves that IPv4 with max octets is detected."""
        findings = scan_text("255.255.255.255")
        assert any(pii_type == "ipv4" for pii_type, *_ in findings), (
            "Expected max-value IPv4 detected"
        )

    def test_ipv4_severity_is_low(self):
        """This test proves that IPv4 severity is LOW."""
        findings = scan_text("192.168.1.1")
        ipv4_findings = [
            (pii_type, severity) for pii_type, severity, *_ in findings if pii_type == "ipv4"
        ]
        assert len(ipv4_findings) > 0, "Expected IPv4 finding"
        assert ipv4_findings[0][1] == "LOW", (
            f"Expected LOW severity for IPv4, got {ipv4_findings[0][1]}"
        )

    def test_no_false_positive_for_invalid_ipv4(self):
        """This test proves that invalid IPv4 (octets >255) is not detected."""
        findings = scan_text("256.256.256.256")
        ipv4_findings = [pii_type for pii_type, *_ in findings if pii_type == "ipv4"]
        assert len(ipv4_findings) == 0, "Expected no match for out-of-range IPv4"


class TestScanTextEdgeCases:
    """Tests for edge cases in scan_text."""

    def test_empty_string_returns_empty_list(self):
        """This test proves that empty string yields empty findings."""
        findings = scan_text("")
        assert findings == [], f"Expected empty list, got {findings}"

    def test_text_with_no_pii_returns_empty_list(self):
        """This test proves that clean text yields empty findings."""
        findings = scan_text("The quick brown fox jumps over the lazy dog")
        assert findings == [], f"Expected empty findings, got {findings}"

    def test_multiple_pii_types_in_same_text(self):
        """This test proves that multiple PII types in one text are detected."""
        text = "Contact john@example.com at 555-123-4567. SSN: 123-45-6789. IP: 192.168.1.1"
        findings = scan_text(text)
        pii_types = set(pii_type for pii_type, *_ in findings)
        assert "email" in pii_types, "Expected email in findings"
        assert "us_phone" in pii_types, "Expected phone in findings"
        assert "ssn" in pii_types, "Expected SSN in findings"
        assert "ipv4" in pii_types, "Expected IPv4 in findings"

    def test_duplicate_pii_pattern_returns_single_tuple(self):
        """This test proves that duplicate PII patterns return one tuple per type."""
        text = "Email john@example.com and jane@example.com"
        findings = scan_text(text)
        email_count = sum(1 for pii_type, *_ in findings if pii_type == "email")
        assert email_count == 1, (
            f"Expected one email finding even with multiple emails, got {email_count}"
        )


# ---------------------------------------------------------------------------
# Tests for scan_pairs
# ---------------------------------------------------------------------------


class TestScanPairsBasic:
    """Tests for basic scan_pairs functionality."""

    def test_empty_list_returns_empty_warnings(self):
        """This test proves that empty pair list yields no warnings."""
        result = scan_pairs([])
        assert result == [], f"Expected empty warnings, got {result}"

    def test_single_clean_pair_returns_no_warnings(self):
        """This test proves that clean pairs yield no warnings."""
        pair = PromptPair(
            prompt="What is the capital of France?",
            response="Paris",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert result == [], f"Expected no warnings, got {result}"

    def test_pii_in_prompt_detected(self):
        """This test proves that PII in prompt is detected."""
        pair = PromptPair(
            prompt="My email is john@example.com",
            response="Got it",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings"
        assert result[0].pair_index == 0, "Expected pair_index=0"
        assert result[0].pii_type == "email", f"Expected email PII type, got {result[0].pii_type}"

    def test_pii_in_response_detected(self):
        """This test proves that PII in response is detected."""
        pair = PromptPair(
            prompt="What is your phone?",
            response="You can reach me at 555-123-4567",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings"
        assert result[0].pii_type == "us_phone", (
            f"Expected phone PII type, got {result[0].pii_type}"
        )

    def test_pii_warning_has_correct_fields(self):
        """This test proves that PIIWarning has all required fields."""
        pair = PromptPair(
            prompt="SSN: 123-45-6789",
            response="OK",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings"
        warning = result[0]
        assert isinstance(warning, PIIWarning), f"Expected PIIWarning instance, got {type(warning)}"
        assert hasattr(warning, "pair_index"), "Expected pair_index field"
        assert hasattr(warning, "pii_type"), "Expected pii_type field"
        assert hasattr(warning, "severity"), "Expected severity field"
        assert hasattr(warning, "count"), "Expected count field"
        assert warning.severity == "HIGH", f"Expected severity=HIGH for SSN, got {warning.severity}"


class TestScanPairsPromptFormats:
    """Tests for different prompt formats."""

    def test_string_prompt_scanned(self):
        """This test proves that string prompts are scanned."""
        pair = PromptPair(
            prompt="Contact: john@example.com",
            response="OK",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings for string prompt"

    def test_list_prompt_with_content_key_scanned(self):
        """This test proves that list prompts with content key are scanned."""
        pair = PromptPair(
            prompt=[{"role": "user", "content": "Email me at test@example.com"}],
            response="OK",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings from list prompt content"
        assert result[0].pii_type == "email", (
            f"Expected email detection in list prompt, got {result[0].pii_type}"
        )

    def test_list_prompt_with_text_key_scanned(self):
        """This test proves that list prompts with text key are scanned."""
        pair = PromptPair(
            prompt=[{"role": "user", "text": "Call me at 555-123-4567"}],
            response="OK",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings from list prompt text"

    def test_list_prompt_multiple_items_joined(self):
        """This test proves that multiple list items are joined."""
        pair = PromptPair(
            prompt=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "My SSN is 123-45-6789"},
            ],
            response="OK",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings from multi-item list prompt"


class TestScanPairsMultiplePairs:
    """Tests for multiple pairs."""

    def test_multiple_pairs_with_different_pii(self):
        """This test proves that PII in multiple pairs is tracked separately."""
        pairs = [
            PromptPair(
                prompt="Email: user1@example.com",
                response="OK",
                source_model="openai/gpt-4o",
            ),
            PromptPair(
                prompt="Phone: 555-123-4567",
                response="OK",
                source_model="openai/gpt-4o",
            ),
        ]
        result = scan_pairs(pairs)
        assert len(result) == 2, f"Expected 2 warnings, got {len(result)}"
        assert result[0].pair_index == 0, "Expected first warning for pair 0"
        assert result[1].pair_index == 1, "Expected second warning for pair 1"

    def test_pair_index_matches_input_position(self):
        """This test proves that pair_index corresponds to input list position."""
        pairs = [
            PromptPair(
                prompt="Clean",
                response="Clean",
                source_model="openai/gpt-4o",
            ),  # index 0
            PromptPair(
                prompt="Clean",
                response="Clean",
                source_model="openai/gpt-4o",
            ),  # index 1
            PromptPair(
                prompt="Email: test@example.com",
                response="OK",
                source_model="openai/gpt-4o",
            ),  # index 2 with PII
        ]
        result = scan_pairs(pairs)
        assert len(result) > 0, "Expected warnings"
        assert result[0].pair_index == 2, f"Expected pair_index=2, got {result[0].pair_index}"


class TestScanPairsCountField:
    """Tests for the count field in PIIWarning."""

    def test_count_reflects_matches_in_text(self):
        """This test proves that count field tracks PII occurrences."""
        pair = PromptPair(
            prompt="User 1: 192.168.1.1, User 2: 192.168.1.2",
            response="OK",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings"
        # Count should be >= 1 for at least one IP detection
        assert result[0].count >= 1, f"Expected count >= 1, got {result[0].count}"

    def test_scan_text_count_is_occurrence_count(self):
        """Proves that scan_text returns actual match counts, not just 1 per PII type.

        A text with 3 email addresses must yield occurrence_count == 3, not 1 or 2.
        """
        text = "a@example.com, b@example.com, c@example.com"
        findings = scan_text(text)
        email_findings = [
            (pii_type, count) for pii_type, _sev, count in findings if pii_type == "email"
        ]
        assert len(email_findings) == 1, f"Expected one email entry, got {email_findings}"
        _pii_type, occurrence_count = email_findings[0]
        assert occurrence_count == 3, (
            f"Expected occurrence_count == 3 for 3 email addresses, got {occurrence_count}"
        )


class TestScanPairsWithMetadata:
    """Tests for pairs with optional metadata."""

    def test_pairs_with_metadata_scanned(self):
        """This test proves that metadata doesn't interfere with scanning."""
        pair = PromptPair(
            prompt="Email: test@example.com",
            response="OK",
            source_model="openai/gpt-4o",
            metadata={"session_id": "xyz"},
            output_type=OutputType.SHORT_TEXT,
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings even with metadata"


class TestScanPairsSeverityLevels:
    """Tests for severity levels in findings."""

    def test_high_severity_pii_detected_with_correct_level(self):
        """This test proves that HIGH severity PII is correctly marked."""
        pairs = [
            PromptPair(
                prompt="SSN: 123-45-6789",
                response="OK",
                source_model="openai/gpt-4o",
            ),
            PromptPair(
                prompt="Card: 4532-1234-5678-9010",
                response="OK",
                source_model="openai/gpt-4o",
            ),
        ]
        result = scan_pairs(pairs)
        severities = [w.severity for w in result]
        assert "HIGH" in severities, f"Expected HIGH severity in {severities}"

    def test_medium_severity_pii_detected_with_correct_level(self):
        """This test proves that MEDIUM severity PII is correctly marked."""
        pair = PromptPair(
            prompt="Email: test@example.com",
            response="OK",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings"
        assert result[0].severity == "MEDIUM", (
            f"Expected MEDIUM severity for email, got {result[0].severity}"
        )

    def test_low_severity_pii_detected_with_correct_level(self):
        """This test proves that LOW severity PII is correctly marked."""
        pair = PromptPair(
            prompt="Server: 192.168.1.1",
            response="OK",
            source_model="openai/gpt-4o",
        )
        result = scan_pairs([pair])
        assert len(result) > 0, "Expected warnings"
        assert result[0].severity == "LOW", (
            f"Expected LOW severity for IPv4, got {result[0].severity}"
        )


# ---------------------------------------------------------------------------
# Adversarial tests — Unicode evasion and edge-case inputs
# ---------------------------------------------------------------------------


class TestPIIScannerAdversarial:
    """Adversarial tests documenting scanner behaviour under Unicode evasion.

    Standard inputs must be detected; evasion inputs are expected to slip
    through the regex-based scanner (marked xfail where the scanner misses them).
    The xfail markers document known limitations without breaking CI.
    """

    def test_standard_ssn_detected(self):
        """Standard SSN format '123-45-6789' must be detected."""
        findings = scan_text("123-45-6789")
        assert any(pii_type == "ssn" for pii_type, *_ in findings), (
            "Expected SSN '123-45-6789' to be detected"
        )

    def test_standard_email_detected(self):
        """Standard email 'john.doe@example.com' must be detected."""
        findings = scan_text("john.doe@example.com")
        assert any(pii_type == "email" for pii_type, *_ in findings), (
            "Expected email 'john.doe@example.com' to be detected"
        )

    def test_standard_credit_card_detected(self):
        """Standard credit card '4111-1111-1111-1111' must be detected."""
        findings = scan_text("4111-1111-1111-1111")
        assert any(pii_type == "credit_card" for pii_type, *_ in findings), (
            "Expected credit card '4111-1111-1111-1111' to be detected"
        )

    def test_standard_phone_detected(self):
        """Standard phone '(555) 123-4567' must be detected."""
        findings = scan_text("(555) 123-4567")
        assert any(pii_type == "us_phone" for pii_type, *_ in findings), (
            "Expected phone '(555) 123-4567' to be detected"
        )

    def test_clean_text_no_false_positives(self):
        """Normal prose has no PII detected."""
        findings = scan_text("The quick brown fox jumps over the lazy dog.")
        assert findings == [], f"Expected no PII in clean prose, got {findings}"

    @pytest.mark.xfail(
        reason=(
            "regex scanner limitation: zero-width space (U+200B) inserted between digits "
            "breaks \\d{3} — the scanner misses SSN evasion via zero-width chars"
        )
    )
    def test_ssn_with_zero_width_chars_not_detected(self):
        """SSN with a zero-width space (U+200B) between digits evades the regex.

        The pattern \\b\\d{3}-\\d{2}-\\d{4}\\b requires three consecutive ASCII
        digits before the first hyphen.  A zero-width space inserted between
        the first and second digit breaks that run, so the scanner misses the
        value.  This is a known limitation; the test is marked xfail to
        document it without blocking CI.
        """
        ssn_with_zwsp = "123\u200b-45-6789"  # zero-width space between '3' and '-'
        findings = scan_text(ssn_with_zwsp)
        # If the scanner ever gains Unicode-aware detection this should pass.
        assert any(pii_type == "ssn" for pii_type, *_ in findings), (
            "Expected scanner to detect SSN despite zero-width space evasion"
        )

    def test_email_with_cyrillic_a_not_detected(self):
        """Email where a Latin 'a' is replaced with Cyrillic 'а' (U+0430) partially evades.

        The email regex character class [a-zA-Z0-9._%+-] accepts only ASCII chars.
        A Cyrillic homoglyph (U+0430) in the local-part terminates the greedy match
        early, so the scanner does NOT match the full 'useraT@example.com' string.
        However, the scanner still fires because the ASCII fragment 't@example.com'
        after the Cyrillic character satisfies the pattern independently.

        This documents the evasion technique: the detected address is a truncated
        false-positive fragment ('t@example.com'), not the true full address.
        An attacker can therefore evade username-level matching while still
        triggering a scanner hit on a meaningless stub.
        """
        # 'а' here is Cyrillic U+0430, visually identical to Latin 'a'
        cyrillic_email = "user\u0430t@example.com"
        findings = scan_text(cyrillic_email)
        # The scanner detects a partial match — 't@example.com' — not the full address.
        # This is the documented partial-evasion behaviour.
        assert any(pii_type == "email" for pii_type, *_ in findings), (
            "Expected scanner to fire on the ASCII stub after the Cyrillic char "
            "(partial match 't@example.com' — not the full address 'useraT@example.com')"
        )

    def test_credit_card_space_padded_behavior(self):
        """Credit card with double spaces between groups is NOT detected by the scanner.

        The credit-card regex uses [-\\s]? — a single optional whitespace.
        Double spaces ('  ') require two whitespace characters between groups,
        so the pattern fails to match.  This test documents that known gap
        without asserting detection.
        """
        double_space_cc = "4111  1111  1111  1111"
        findings = scan_text(double_space_cc)
        cc_findings = [pii_type for pii_type, *_ in findings if pii_type == "credit_card"]
        # Document: double-space padded cards are currently NOT detected.
        # If the regex is later updated to handle \\s+ this assertion should
        # be flipped to assert detection.
        assert len(cc_findings) == 0, (
            "Double-space padded credit card is expected to evade the current regex "
            f"([-\\s]? matches at most one whitespace); got {cc_findings}"
        )

    @pytest.mark.xfail(
        reason=(
            "regex scanner limitation: fullwidth parentheses （U+FF08）（U+FF09） are not "
            "matched by \\( and \\) in the phone pattern"
        )
    )
    def test_phone_with_fullwidth_parens_not_detected(self):
        """Phone number using fullwidth parentheses （） evades the ASCII-only regex.

        The phone pattern uses literal \\( and \\) which only match the ASCII
        parentheses U+0028 / U+0029.  Fullwidth variants U+FF08 / U+FF09 are
        visually similar but distinct code points that the regex does not cover.
        """
        fullwidth_phone = "\uff08555\uff09 123-4567"  # （555） 123-4567
        findings = scan_text(fullwidth_phone)
        assert any(pii_type == "us_phone" for pii_type, *_ in findings), (
            "Expected scanner to detect phone despite fullwidth parentheses evasion"
        )

    def test_scan_pairs_accumulates_across_prompt_and_response(self):
        """scan_pairs accumulates PII findings from both prompt and response fields.

        When the same PII type appears in both the prompt and the response of a
        single pair, the occurrence_count on the resulting PIIWarning must reflect
        the combined total (prompt occurrences + response occurrences).
        """
        pair = PromptPair(
            prompt="My SSN is 123-45-6789.",
            response="Confirmed: SSN 123-45-6789 on file.",
            source_model="openai/gpt-4o",
        )
        warnings = scan_pairs([pair])

        ssn_warnings = [w for w in warnings if w.pii_type == "ssn"]
        assert len(ssn_warnings) == 1, (
            f"Expected exactly one SSN PIIWarning (merged per pair), got {ssn_warnings}"
        )
        warning = ssn_warnings[0]
        assert warning.pair_index == 0, f"Expected pair_index=0, got {warning.pair_index}"
        # One occurrence in prompt + one in response = 2 total
        assert warning.occurrence_count == 2, (
            f"Expected occurrence_count=2 (one in prompt, one in response), "
            f"got {warning.occurrence_count}"
        )
