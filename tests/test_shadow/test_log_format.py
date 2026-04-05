"""Tests for shadow log format serialization."""

from __future__ import annotations

from rosettastone.shadow.log_format import ShadowLogEntry, read_log_entries, write_log_entry


class TestShadowLogEntry:
    def test_round_trip_jsonl(self):
        entry = ShadowLogEntry(
            prompt="Hello",
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            source_response="Hi there",
            target_response="Hello! How can I help?",
            source_latency_ms=150.0,
            target_latency_ms=200.0,
        )
        json_str = entry.model_dump_json()
        restored = ShadowLogEntry.from_jsonl_line(json_str)
        assert restored.prompt == entry.prompt
        assert restored.source_model == entry.source_model
        assert restored.target_response == entry.target_response
        assert restored.source_latency_ms == entry.source_latency_ms

    def test_to_prompt_pair_dict(self):
        entry = ShadowLogEntry(
            prompt="What is 2+2?",
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            source_response="4",
            target_response="The answer is 4.",
        )
        d = entry.to_prompt_pair_dict()
        assert d["prompt"] == "What is 2+2?"
        assert d["response"] == "4"  # source_response is the baseline
        assert d["source_model"] == "openai/gpt-4o"
        assert "shadow_request_id" in d["metadata"]
        assert d["metadata"]["target_response"] == "The answer is 4."

    def test_write_and_read_entries(self, tmp_path):
        entries = [
            ShadowLogEntry(
                prompt=f"q{i}",
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                source_response=f"a{i}",
                target_response=f"b{i}",
            )
            for i in range(3)
        ]
        for e in entries:
            write_log_entry(e, tmp_path)

        read_back = read_log_entries(tmp_path)
        assert len(read_back) == 3
        prompts = {e.prompt for e in read_back}
        assert prompts == {"q0", "q1", "q2"}

    def test_valid_rosettastone_ingestion_input(self):
        """to_prompt_pair_dict() produces a valid PromptPair-compatible dict."""
        from rosettastone.core.types import PromptPair

        entry = ShadowLogEntry(
            prompt="Test prompt",
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            source_response="Test response",
            target_response="Alternative response",
        )
        d = entry.to_prompt_pair_dict()
        pair = PromptPair(**d)
        assert pair.prompt == "Test prompt"
        assert pair.source_model == "openai/gpt-4o"
