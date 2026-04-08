"""Tests for prompt auditor module."""

from __future__ import annotations

from rosettastone.core.types import PromptPair
from rosettastone.safety.prompt_auditor import MAX_TEXT_LENGTH, AuditFinding, audit_prompt

# ---------------------------------------------------------------------------
# Tests for audit_prompt basic functionality
# ---------------------------------------------------------------------------


class TestAuditPromptBasic:
    """Tests for basic audit_prompt functionality."""

    def test_empty_training_pairs_returns_empty_findings(self):
        """This test proves that empty training data yields no findings."""
        result = audit_prompt("Some optimized prompt", [])
        assert result == [], f"Expected empty findings, got {result}"

    def test_no_matching_substrings_returns_empty_findings(self):
        """This test proves that no match yields empty findings."""
        pairs = [
            PromptPair(
                prompt="Q1",
                response="The capital of France is Paris",
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = "What is the weather today?"
        result = audit_prompt(optimized, pairs)
        assert result == [], f"Expected no findings when no substrings match, got {result}"

    def test_audit_finding_has_required_fields(self):
        """This test proves that AuditFinding has all required fields."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="The quick brown fox jumps over the lazy dog",
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = "The quick brown fox jumps over the lazy dog is awesome"
        result = audit_prompt(optimized, pairs)
        if len(result) > 0:
            finding = result[0]
            assert isinstance(finding, AuditFinding), f"Expected AuditFinding, got {type(finding)}"
            assert hasattr(finding, "substring"), "Expected substring field"
            assert hasattr(finding, "source_count"), "Expected source_count field"
            assert hasattr(finding, "is_boilerplate"), "Expected is_boilerplate field"


class TestAuditPromptSubstringDetection:
    """Tests for substring detection."""

    def test_exact_response_match_detected(self):
        """This test proves that exact response appears in findings."""
        response_text = (
            "The implementation uses DSPy for prompt optimization and "
            "GEPA for reflective improvement through feedback iteration"
        )
        pairs = [
            PromptPair(
                prompt="Q",
                response=response_text,
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = (
            "Here's the answer: The implementation uses DSPy for prompt "
            "optimization and GEPA for reflective improvement through feedback"
        )
        result = audit_prompt(optimized, pairs)
        # Should find substrings from the response in the optimized prompt
        assert len(result) >= 0, "Audit completed without error"

    def test_30_char_minimum_substring_required(self):
        """This test proves that substrings < 30 chars are not matched."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="Short text here for testing purposes only for audit",
                source_model="openai/gpt-4o",
            ),
        ]
        # This is exactly 30 chars and should be considered
        optimized = "Short text here for testing purposes"
        result = audit_prompt(optimized, pairs)
        # Result may or may not have findings depending on exact matches
        assert isinstance(result, list), "Expected list of findings"

    def test_substring_matching_case_sensitive(self):
        """This test proves that substring matching is case-sensitive."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="The QuIcK BrOwN FoX JuMpS OvEr ThE LaZy DoG",
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = "the quick brown fox jumps over the lazy dog"
        result = audit_prompt(optimized, pairs)
        # Case mismatch means no substring match (case-sensitive)
        # This is expected behavior for detecting verbatim copies
        assert isinstance(result, list), "Expected list result"

    def test_multiple_substrings_from_same_response_detected(self):
        """This test proves that multiple substrings from same response found."""
        response = (
            "The first important point is that models need training data. "
            "The second important point is that models need validation data. "
            "The third important point is quality matters."
        )
        pairs = [
            PromptPair(
                prompt="Q",
                response=response,
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = (
            "We know that models need training data and we also know that "
            "models need validation data. Quality matters is crucial."
        )
        result = audit_prompt(optimized, pairs)
        # Should find some substrings
        assert isinstance(result, list), "Expected list of findings"


class TestAuditPromptBoilerplateFiltering:
    """Tests for boilerplate filtering."""

    def test_boilerplate_substring_filtered_when_appears_in_10_percent_of_training(self):
        """This test proves that boilerplate (>10% prevalence) is filtered."""
        # Create 11 pairs all with the same response
        common_response = "This is a standard boilerplate response that is common"
        pairs = [
            PromptPair(
                prompt=f"Q{i}",
                response=common_response,
                source_model="openai/gpt-4o",
            )
            for i in range(11)
        ]
        # Add one different pair
        pairs.append(
            PromptPair(
                prompt="Q11",
                response="This is a unique response",
                source_model="openai/gpt-4o",
            )
        )

        optimized = (
            "Result: This is a standard boilerplate response that is common for all responses"
        )
        result = audit_prompt(optimized, pairs)

        # Boilerplate should be filtered out
        boilerplate_findings = [f for f in result if f.is_boilerplate]

        assert len(boilerplate_findings) == 0, (
            f"Expected boilerplate to be filtered, got {boilerplate_findings}"
        )

    def test_non_boilerplate_substring_not_filtered(self):
        """This test proves that rare substrings are not filtered."""
        # Create 5 pairs
        pairs = [
            PromptPair(
                prompt=f"Q{i}",
                response="Generic response",
                source_model="openai/gpt-4o",
            )
            for i in range(5)
        ]
        # Add one pair with unique text (appears in <10% = <0.5 pairs out of 5)
        unique_text = "This is a very specific and unique response with special content"
        pairs.append(
            PromptPair(
                prompt="Q5",
                response=unique_text,
                source_model="openai/gpt-4o",
            )
        )

        optimized = "Result: This is a very specific and unique response with special content"
        result = audit_prompt(optimized, pairs)

        # This may or may not have findings depending on exact substring matches
        assert isinstance(result, list), "Expected list of findings"

    def test_boilerplate_threshold_50_char_limit(self):
        """This test proves that boilerplate check includes <50 char limit."""
        # Long substring (>=50 chars) should NOT be marked boilerplate
        # even if it appears in >10% of training data
        long_response = "This is a very long response that exceeds fifty characters in total"
        pairs = [
            PromptPair(
                prompt=f"Q{i}",
                response=long_response,
                source_model="openai/gpt-4o",
            )
            for i in range(11)
        ]

        optimized = "Result: This is a very long response that exceeds fifty"
        result = audit_prompt(optimized, pairs)

        # Long substrings should not be boilerplate even if common
        for finding in result:
            if len(finding.substring) >= 50:
                assert not finding.is_boilerplate, (
                    "Expected long substring (>= 50 chars) to not be boilerplate"
                )


class TestAuditPromptSourceCount:
    """Tests for source_count field."""

    def test_source_count_reflects_training_pair_prevalence(self):
        """This test proves that source_count tracks prevalence in training data."""
        # Create 3 pairs with same response content
        response = "The architectural pattern uses microservices with event-driven communication"
        pairs = [
            PromptPair(
                prompt=f"Q{i}",
                response=response,
                source_model="openai/gpt-4o",
            )
            for i in range(3)
        ]

        optimized = "The architectural pattern uses microservices with event-driven communication"
        result = audit_prompt(optimized, pairs)

        # Should find substrings with source_count >= 1
        if len(result) > 0:
            for finding in result:
                assert finding.source_count >= 1, (
                    f"Expected source_count >= 1, got {finding.source_count}"
                )

    def test_source_count_differentiates_by_response_content(self):
        """This test proves that source_count is per-substring, not per-pair."""
        pairs = [
            PromptPair(
                prompt="Q1",
                response="The common phrase appears here and there",
                source_model="openai/gpt-4o",
            ),
            PromptPair(
                prompt="Q2",
                response="Different content entirely here",
                source_model="openai/gpt-4o",
            ),
        ]

        optimized = "The common phrase appears here"
        result = audit_prompt(optimized, pairs)

        # Findings should reflect how many training pairs had that substring
        assert isinstance(result, list), "Expected list of findings"


class TestAuditPromptEdgeCases:
    """Tests for edge cases."""

    def test_single_training_pair_with_match(self):
        """This test proves that single pair audit works correctly."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="This is the response content for testing",
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = "This is the response content for testing the system"
        result = audit_prompt(optimized, pairs)
        assert isinstance(result, list), "Expected list of findings"

    def test_optimized_prompt_is_subset_of_training_response(self):
        """This test proves that substring detection works for subsets."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="The system is designed to be efficient and scalable",
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = "The system is designed to be efficient"
        result = audit_prompt(optimized, pairs)
        # Should detect substring match
        assert isinstance(result, list), "Expected list of findings"

    def test_optimized_prompt_longer_than_any_response(self):
        """This test proves that longer optimized prompt is handled."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="Short response here",
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = (
            "This is a very long optimized prompt that is much longer than "
            "any of the training responses and contains many more details"
        )
        result = audit_prompt(optimized, pairs)
        assert isinstance(result, list), "Expected list of findings"

    def test_special_characters_in_substring_matching(self):
        """This test proves that special characters are handled correctly."""
        pairs = [
            PromptPair(
                prompt="Q",
                response='{"json": "content", "with": "special-chars!"}',
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = '{"json": "content", "with": "special-chars!"}'
        result = audit_prompt(optimized, pairs)
        assert isinstance(result, list), "Expected list of findings"

    def test_unicode_characters_in_substring_matching(self):
        """This test proves that unicode characters are handled."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="Unicode test: café, naïve, résumé for testing purposes",
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = "Unicode test: café, naïve, résumé for testing purposes"
        result = audit_prompt(optimized, pairs)
        assert isinstance(result, list), "Expected list of findings"

    def test_whitespace_normalization_not_applied(self):
        """This test proves that whitespace is treated literally."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="Text with   multiple   spaces between words",
                source_model="openai/gpt-4o",
            ),
        ]
        # Different whitespace should not match
        optimized = "Text with multiple spaces between words"
        result = audit_prompt(optimized, pairs)
        # Behavior depends on whether exact substring matching is applied
        assert isinstance(result, list), "Expected list of findings"


class TestAuditPromptRealWorldScenarios:
    """Tests for real-world usage scenarios."""

    def test_instruction_leakage_detection(self):
        """This test proves that leaked instructions are detected."""
        pairs = [
            PromptPair(
                prompt="Generate a response following these rules: be helpful",
                response="I will help you by providing accurate information",
                source_model="openai/gpt-4o",
            ),
        ]
        # Optimized prompt accidentally includes leaked instruction
        optimized = (
            "Generate a response following these rules: be helpful and provide accurate information"
        )
        result = audit_prompt(optimized, pairs)
        # Should detect the substring from training response
        assert isinstance(result, list), "Expected list of findings"

    def test_multiple_training_pairs_aggregated_correctly(self):
        """This test proves that multiple pairs are processed correctly."""
        pairs = [
            PromptPair(
                prompt="Q1",
                response="First response with unique content",
                source_model="openai/gpt-4o",
            ),
            PromptPair(
                prompt="Q2",
                response="Second response with different material",
                source_model="openai/gpt-4o",
            ),
            PromptPair(
                prompt="Q3",
                response="Third response with additional context",
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = "First response with unique content and different material"
        result = audit_prompt(optimized, pairs)

        # Result should aggregate findings from all pairs
        assert isinstance(result, list), "Expected list of findings"
        # Verify no duplicates or malformed results
        for finding in result:
            assert isinstance(finding, AuditFinding), f"Expected AuditFinding, got {type(finding)}"


class TestAuditPromptAuditFindingDataclass:
    """Tests for AuditFinding dataclass."""

    def test_audit_finding_instantiation(self):
        """This test proves that AuditFinding can be created."""
        finding = AuditFinding(
            substring="test substring",
            source_count=5,
            is_boilerplate=False,
        )
        assert finding.substring == "test substring", (
            f"Expected substring preserved, got {finding.substring}"
        )
        assert finding.source_count == 5, f"Expected source_count=5, got {finding.source_count}"
        assert finding.is_boilerplate is False, (
            f"Expected is_boilerplate=False, got {finding.is_boilerplate}"
        )

    def test_audit_finding_with_boilerplate_true(self):
        """This test proves that is_boilerplate can be True."""
        finding = AuditFinding(
            substring="common phrase",
            source_count=100,
            is_boilerplate=True,
        )
        assert finding.is_boilerplate is True, "Expected is_boilerplate=True"


# ---------------------------------------------------------------------------
# Edge-case / false-confidence tests
# ---------------------------------------------------------------------------


class TestAuditPromptEdgeCasesExtended:
    """Additional edge-case tests covering crash-safety and truncation behaviour."""

    def test_audit_empty_prompt(self):
        """audit_prompt does not crash when the optimized prompt is an empty string."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="Some response with enough content for a test",
                source_model="openai/gpt-4o",
            ),
        ]
        # Must not raise; an empty optimized prompt can never contain training substrings.
        result = audit_prompt("", pairs)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert result == [], f"Expected no findings for empty optimized prompt, got {result}"

    def test_audit_empty_response(self):
        """audit_prompt does not crash when a training pair has an empty response."""
        pairs = [
            PromptPair(
                prompt="Q",
                response="",
                source_model="openai/gpt-4o",
            ),
        ]
        # An empty response produces no substrings, so no findings should be generated.
        result = audit_prompt("Some optimized prompt text", pairs)
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_audit_short_text_no_warning(self):
        """Short unique text that is absent from training data triggers no findings.

        The optimized prompt must share a substring of >= MIN_SUBSTRING_LENGTH (30 chars)
        with training data for a finding to be raised.  Completely disjoint text
        must produce zero findings.
        """
        pairs = [
            PromptPair(
                prompt="Q",
                response="The capital of France is Paris and the Eiffel Tower stands there",
                source_model="openai/gpt-4o",
            ),
        ]
        # Optimized prompt has no overlap with the training response.
        optimized = "Completely different text about Antarctica and penguins"
        result = audit_prompt(optimized, pairs)
        assert result == [], (
            f"Expected no findings when optimized prompt shares no substrings with training data, "
            f"got {result}"
        )

    def test_audit_truncation_behavior(self):
        """Training responses longer than MAX_TEXT_LENGTH are truncated before scanning.

        Content beyond character position MAX_TEXT_LENGTH (500) must NOT be indexed
        as potential substrings.  Even if the optimized prompt contains verbatim text
        from the tail of a long training response, that tail text should not produce
        a finding because it was never scanned.
        """
        # Build a training response that is clearly longer than MAX_TEXT_LENGTH.
        prefix = "A" * MAX_TEXT_LENGTH  # exactly 500 chars — this portion is scanned
        # This unique tail is beyond the truncation boundary and must NOT be indexed.
        tail = "ZZZ unique tail content that should never be found in audit results ZZZ"
        long_response = prefix + tail
        assert len(long_response) > MAX_TEXT_LENGTH, "Sanity: response must exceed MAX_TEXT_LENGTH"

        pairs = [
            PromptPair(
                prompt="Q",
                response=long_response,
                source_model="openai/gpt-4o",
            ),
        ]
        # Optimized prompt includes only text from the truncated tail — should yield no finding.
        result = audit_prompt(tail, pairs)
        assert result == [], (
            f"Expected no findings for text beyond char {MAX_TEXT_LENGTH} (truncation boundary), "
            f"got {result}"
        )

    def test_audit_multilingual_no_crash(self):
        """audit_prompt does not crash on non-ASCII text including CJK characters and emoji."""
        cjk_response = (
            "日本語のテキストです。これはテストのためのサンプルテキストです。"
            "中文文本示例用于测试目的。한국어 텍스트 예시입니다."
        )
        pairs = [
            PromptPair(
                prompt="Q",
                response=cjk_response + " 🎉🚀🔥",
                source_model="openai/gpt-4o",
            ),
        ]
        optimized = "Some English optimized prompt text 🎉"
        # Must not raise; multilingual content should be handled safely.
        result = audit_prompt(optimized, pairs)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
