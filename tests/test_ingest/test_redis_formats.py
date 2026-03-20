"""Tests for redis_formats parsers."""

from __future__ import annotations

import json

from rosettastone.core.types import PromptPair
from rosettastone.ingest.redis_formats import parse_litellm_entry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SOURCE_MODEL = "openai/gpt-4o"


def _litellm_value(
    messages: list[dict],
    assistant_content: str = "The answer is 42.",
) -> bytes:
    """Build a minimal LiteLLM cache entry as bytes."""
    data = {
        "messages": messages,
        "response": {"choices": [{"message": {"role": "assistant", "content": assistant_content}}]},
    }
    return json.dumps(data).encode()


def _key(name: str = "litellm:abc123") -> bytes:
    return name.encode()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_parse_litellm_returns_prompt_pair_for_valid_entry():
    """This test proves that a valid LiteLLM cache entry produces a PromptPair."""
    messages = [{"role": "user", "content": "What is 6 * 7?"}]
    value = _litellm_value(messages, assistant_content="42")

    result = parse_litellm_entry(_key(), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "What is 6 * 7?", f"Unexpected prompt: {result.prompt!r}"
    assert result.response == "42", f"Unexpected response: {result.response!r}"
    assert result.source_model == _SOURCE_MODEL, f"Unexpected source_model: {result.source_model!r}"


def test_parse_litellm_uses_last_user_message_as_prompt():
    """This test proves that when multiple user messages exist, the last one is used as prompt."""
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up question"},
    ]
    value = _litellm_value(messages, assistant_content="Follow-up answer")

    result = parse_litellm_entry(_key(), value, _SOURCE_MODEL)

    assert result is not None, "Expected a result, got None"
    assert result.prompt == "Follow-up question", (
        f"Expected last user message as prompt, got: {result.prompt!r}"
    )


def test_parse_litellm_system_messages_not_used_as_prompt():
    """This test proves that system messages are not selected as the prompt."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    value = _litellm_value(messages)

    result = parse_litellm_entry(_key(), value, _SOURCE_MODEL)

    assert result is not None, "Expected a result"
    assert result.prompt == "Hello!", f"Expected user message as prompt, got: {result.prompt!r}"


def test_parse_litellm_string_response_field():
    """This test proves that a plain string in the 'response' field is accepted."""
    data = {
        "messages": [{"role": "user", "content": "Ping"}],
        "response": "Pong",
    }
    value = json.dumps(data).encode()

    result = parse_litellm_entry(_key(), value, _SOURCE_MODEL)

    assert result is not None, "Expected a result for string response"
    assert result.response == "Pong", f"Unexpected response: {result.response!r}"


# ---------------------------------------------------------------------------
# None / failure cases
# ---------------------------------------------------------------------------


def test_parse_litellm_returns_none_for_invalid_json():
    """This test proves that malformed JSON bytes return None without raising."""
    result = parse_litellm_entry(_key(), b"{not valid json", _SOURCE_MODEL)

    assert result is None, f"Expected None for invalid JSON, got: {result!r}"


def test_parse_litellm_returns_none_for_missing_messages():
    """This test proves that an entry with no 'messages' key returns None."""
    data = {"response": {"choices": [{"message": {"role": "assistant", "content": "Hi"}}]}}
    value = json.dumps(data).encode()

    result = parse_litellm_entry(_key(), value, _SOURCE_MODEL)

    assert result is None, f"Expected None for missing messages, got: {result!r}"


def test_parse_litellm_returns_none_when_no_user_message():
    """This test proves that entries with only system/assistant messages return None."""
    messages = [
        {"role": "system", "content": "System prompt only"},
        {"role": "assistant", "content": "No user turn"},
    ]
    value = _litellm_value(messages)

    result = parse_litellm_entry(_key(), value, _SOURCE_MODEL)

    assert result is None, f"Expected None when no user message, got: {result!r}"


def test_parse_litellm_returns_none_for_missing_response():
    """This test proves that an entry with no 'response' key returns None."""
    data = {"messages": [{"role": "user", "content": "Hello?"}]}
    value = json.dumps(data).encode()

    result = parse_litellm_entry(_key(), value, _SOURCE_MODEL)

    assert result is None, f"Expected None for missing response, got: {result!r}"


def test_parse_litellm_returns_none_for_empty_choices():
    """This test proves that a response with an empty choices list returns None."""
    data = {
        "messages": [{"role": "user", "content": "Hello?"}],
        "response": {"choices": []},
    }
    value = json.dumps(data).encode()

    result = parse_litellm_entry(_key(), value, _SOURCE_MODEL)

    assert result is None, f"Expected None for empty choices, got: {result!r}"


def test_parse_litellm_returns_none_for_empty_bytes():
    """This test proves that empty bytes input returns None without raising."""
    result = parse_litellm_entry(_key(), b"", _SOURCE_MODEL)

    assert result is None, f"Expected None for empty bytes, got: {result!r}"


def test_parse_litellm_returns_none_for_non_dict_value():
    """This test proves that a valid JSON array (not dict) as value returns None."""
    result = parse_litellm_entry(_key(), b"[1, 2, 3]", _SOURCE_MODEL)

    assert result is None, f"Expected None for JSON array value, got: {result!r}"
