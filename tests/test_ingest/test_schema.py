"""Tests for the PromptPairInput Pydantic schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rosettastone.ingest.schema import PromptPairInput

# ---------------------------------------------------------------------------
# Happy path: valid inputs
# ---------------------------------------------------------------------------


def test_valid_string_prompt_parses_correctly():
    """This test proves that a string prompt is accepted and stored as-is."""
    data = {
        "prompt": "What is the capital of France?",
        "response": "Paris",
        "source_model": "openai/gpt-4o",
    }
    result = PromptPairInput.model_validate(data)

    assert result.prompt == "What is the capital of France?", (
        f"Expected prompt to be preserved verbatim, got: {result.prompt!r}"
    )
    assert result.response == "Paris", f"Expected response 'Paris', got: {result.response!r}"
    assert result.source_model == "openai/gpt-4o", (
        f"Expected source_model 'openai/gpt-4o', got: {result.source_model!r}"
    )


def test_valid_list_prompt_messages_format_parses():
    """This test proves that a list-of-dicts prompt (messages format) is accepted."""
    data = {
        "prompt": [{"role": "user", "content": "What is the capital of France?"}],
        "response": "Paris",
        "source_model": "openai/gpt-4o",
    }
    result = PromptPairInput.model_validate(data)

    assert isinstance(result.prompt, list), (
        f"Expected prompt to be a list, got: {type(result.prompt)}"
    )
    assert len(result.prompt) == 1, f"Expected 1 message, got: {len(result.prompt)}"
    assert result.prompt[0]["role"] == "user", (
        f"Expected role 'user', got: {result.prompt[0].get('role')!r}"
    )
    assert result.prompt[0]["content"] == "What is the capital of France?", (
        f"Expected correct content, got: {result.prompt[0].get('content')!r}"
    )


def test_dict_response_normalized_to_content_string():
    """This test proves that a dict response with 'content' key is normalized to that string."""
    data = {
        "prompt": "Say hello",
        "response": {"content": "Hello, world!"},
        "source_model": "anthropic/claude-sonnet-4",
    }
    result = PromptPairInput.model_validate(data)

    assert isinstance(result.response, str), (
        f"Expected response to be a string after normalization, got: {type(result.response)}"
    )
    assert result.response == "Hello, world!", (
        f"Expected 'Hello, world!' (from 'content' key), got: {result.response!r}"
    )


def test_dict_response_without_content_key_normalized_to_str_repr():
    """This test proves that a dict response without 'content' key falls back to str(dict)."""
    data = {
        "prompt": "Classify this",
        "response": {"label": "positive", "confidence": 0.95},
        "source_model": "openai/gpt-4o",
    }
    result = PromptPairInput.model_validate(data)

    assert isinstance(result.response, str), (
        f"Expected response to be a string after normalization, got: {type(result.response)}"
    )
    # The fallback is str(dict) — verify the original data is represented
    assert "positive" in result.response, (
        f"Expected fallback string to contain dict contents, got: {result.response!r}"
    )


def test_optional_fields_default_correctly():
    """This test proves that optional fields have correct defaults when omitted."""
    data = {
        "prompt": "Minimal prompt",
        "response": "Minimal response",
        "source_model": "openai/gpt-4o",
    }
    result = PromptPairInput.model_validate(data)

    assert result.metadata == {}, (
        f"Expected metadata default to be empty dict, got: {result.metadata!r}"
    )
    assert result.feedback is None, (
        f"Expected feedback default to be None, got: {result.feedback!r}"
    )
    assert result.input_tokens is None, (
        f"Expected input_tokens default to be None, got: {result.input_tokens!r}"
    )
    assert result.output_tokens is None, (
        f"Expected output_tokens default to be None, got: {result.output_tokens!r}"
    )
    assert result.timestamp is None, (
        f"Expected timestamp default to be None, got: {result.timestamp!r}"
    )


def test_optional_fields_accept_provided_values():
    """This test proves that all optional fields are stored when explicitly provided."""
    data = {
        "prompt": "A prompt",
        "response": "A response",
        "source_model": "openai/gpt-4o",
        "input_tokens": 12,
        "output_tokens": 5,
        "timestamp": "2024-01-01T00:00:00Z",
        "metadata": {"session_id": "abc123"},
        "feedback": "thumbs_up",
    }
    result = PromptPairInput.model_validate(data)

    assert result.input_tokens == 12, f"Expected input_tokens=12, got: {result.input_tokens}"
    assert result.output_tokens == 5, f"Expected output_tokens=5, got: {result.output_tokens}"
    assert result.timestamp == "2024-01-01T00:00:00Z", (
        f"Expected timestamp preserved, got: {result.timestamp!r}"
    )
    assert result.metadata == {"session_id": "abc123"}, (
        f"Expected metadata preserved, got: {result.metadata!r}"
    )
    assert result.feedback == "thumbs_up", (
        f"Expected feedback='thumbs_up', got: {result.feedback!r}"
    )


def test_multi_turn_messages_prompt_accepted():
    """This test proves that multi-turn conversation prompts are accepted."""
    data = {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "response": "6",
        "source_model": "openai/gpt-4o",
    }
    result = PromptPairInput.model_validate(data)

    assert len(result.prompt) == 4, (
        f"Expected 4 messages in conversation, got: {len(result.prompt)}"
    )


# ---------------------------------------------------------------------------
# Error path: missing required fields
# ---------------------------------------------------------------------------


def test_missing_prompt_raises_validation_error():
    """This test proves that omitting 'prompt' triggers a Pydantic ValidationError."""
    data = {
        "response": "Paris",
        "source_model": "openai/gpt-4o",
    }
    with pytest.raises(ValidationError, match="prompt"):
        PromptPairInput.model_validate(data)


def test_missing_response_raises_validation_error():
    """This test proves that omitting 'response' triggers a Pydantic ValidationError."""
    data = {
        "prompt": "What is the capital of France?",
        "source_model": "openai/gpt-4o",
    }
    with pytest.raises(ValidationError, match="response"):
        PromptPairInput.model_validate(data)


def test_missing_source_model_raises_validation_error():
    """This test proves that omitting 'source_model' triggers a Pydantic ValidationError."""
    data = {
        "prompt": "What is the capital of France?",
        "response": "Paris",
    }
    with pytest.raises(ValidationError, match="source_model"):
        PromptPairInput.model_validate(data)


def test_empty_dict_raises_validation_error_for_all_required_fields():
    """This test proves that all three required fields are truly required."""
    with pytest.raises(ValidationError) as exc_info:
        PromptPairInput.model_validate({})

    error_str = str(exc_info.value)
    # All three required fields should appear in the error
    assert "prompt" in error_str, "Expected 'prompt' in ValidationError message"
    assert "response" in error_str, "Expected 'response' in ValidationError message"
    assert "source_model" in error_str, "Expected 'source_model' in ValidationError message"


def test_none_prompt_raises_validation_error():
    """This test proves that None is not accepted as a prompt value."""
    data = {
        "prompt": None,
        "response": "Paris",
        "source_model": "openai/gpt-4o",
    }
    with pytest.raises(ValidationError):
        PromptPairInput.model_validate(data)


def test_none_response_raises_validation_error():
    """This test proves that None is not accepted as a response value."""
    data = {
        "prompt": "What is the capital of France?",
        "response": None,
        "source_model": "openai/gpt-4o",
    }
    with pytest.raises(ValidationError):
        PromptPairInput.model_validate(data)
