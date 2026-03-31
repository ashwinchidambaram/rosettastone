"""Tests for redis_formats parsers."""

from __future__ import annotations

import json

from rosettastone.core.types import PromptPair
from rosettastone.ingest.redis_formats import (
    parse_gptcache_entry,
    parse_langchain_entry,
    parse_litellm_entry,
    parse_redisvl_entry,
)

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


# ---------------------------------------------------------------------------
# LangChain parser — happy paths
# ---------------------------------------------------------------------------


def test_parse_langchain_direct_input_output():
    """This test proves that a LangChain direct input/output dict produces a PromptPair."""
    data = {"input": "What is the capital of France?", "output": "Paris."}
    value = json.dumps(data).encode()

    result = parse_langchain_entry(_key("langchain:abc"), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "What is the capital of France?"
    assert result.response == "Paris."
    assert result.source_model == _SOURCE_MODEL


def test_parse_langchain_llmresult_with_message_content():
    """This test proves that a LangChain LLMResult generations entry produces a PromptPair."""
    data = {
        "type": "LLMResult",
        "llm_output": {},
        "generations": [
            [{"text": "The sky is blue.", "message": {"role": "user", "content": "Why is the sky blue?"}}]
        ],
    }
    value = json.dumps(data).encode()

    result = parse_langchain_entry(_key("langchain:xyz"), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "Why is the sky blue?"
    assert result.response == "The sky is blue."


# ---------------------------------------------------------------------------
# LangChain parser — None / failure cases
# ---------------------------------------------------------------------------


def test_parse_langchain_returns_none_for_invalid_json():
    """This test proves that malformed JSON bytes return None without raising."""
    result = parse_langchain_entry(_key(), b"{bad json", _SOURCE_MODEL)

    assert result is None, f"Expected None for invalid JSON, got: {result!r}"


def test_parse_langchain_returns_none_when_no_prompt_found():
    """This test proves that a generations entry with no prompt source returns None."""
    data = {
        "generations": [[{"text": "Some response text."}]],
    }
    value = json.dumps(data).encode()

    result = parse_langchain_entry(_key(), value, _SOURCE_MODEL)

    assert result is None, f"Expected None when no prompt found, got: {result!r}"


# ---------------------------------------------------------------------------
# RedisVL parser — happy paths
# ---------------------------------------------------------------------------


def test_parse_redisvl_prompt_response_format():
    """This test proves that a RedisVL prompt/response dict produces a PromptPair."""
    data = {"prompt": "Summarize quantum entanglement.", "response": "It is spooky action at a distance.", "metadata": {}}
    value = json.dumps(data).encode()

    result = parse_redisvl_entry(_key("redisvl:abc"), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "Summarize quantum entanglement."
    assert result.response == "It is spooky action at a distance."
    assert result.source_model == _SOURCE_MODEL


def test_parse_redisvl_input_text_format():
    """This test proves that a RedisVL input_text/response dict produces a PromptPair."""
    data = {"input_text": "What is 2+2?", "response": "4", "vector_score": 0.99}
    value = json.dumps(data).encode()

    result = parse_redisvl_entry(_key("redisvl:def"), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "What is 2+2?"
    assert result.response == "4"


def test_parse_redisvl_output_key_variant():
    """This test proves that a RedisVL entry with 'output' key (not 'response') produces a PromptPair."""
    data = {"prompt": "Test prompt", "output": "Test output"}
    value = json.dumps(data).encode()

    result = parse_redisvl_entry(_key("redisvl:key"), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "Test prompt", f"Unexpected prompt: {result.prompt!r}"
    assert result.response == "Test output", f"Unexpected response: {result.response!r}"


# ---------------------------------------------------------------------------
# RedisVL parser — None / failure cases
# ---------------------------------------------------------------------------


def test_parse_redisvl_returns_none_for_invalid_json():
    """This test proves that malformed JSON bytes return None without raising."""
    result = parse_redisvl_entry(_key(), b"not json at all", _SOURCE_MODEL)

    assert result is None, f"Expected None for invalid JSON, got: {result!r}"


def test_parse_redisvl_returns_none_when_missing_prompt_key():
    """This test proves that an entry with no recognized prompt key returns None."""
    data = {"response": "Some answer"}
    value = json.dumps(data).encode()

    result = parse_redisvl_entry(_key(), value, _SOURCE_MODEL)

    assert result is None, f"Expected None for missing prompt key, got: {result!r}"


def test_parse_redisvl_returns_none_when_no_response_key():
    """This test proves that a prompt without response/output key returns None."""
    data = {"prompt": "Test prompt", "something_else": "value"}
    value = json.dumps(data).encode()

    result = parse_redisvl_entry(_key("redisvl:key"), value, _SOURCE_MODEL)

    assert result is None, f"Expected None when no response key, got: {result!r}"


# ---------------------------------------------------------------------------
# GPTCache parser — happy paths
# ---------------------------------------------------------------------------


def test_parse_gptcache_query_answer_format():
    """This test proves that a GPTCache query/answer dict produces a PromptPair."""
    data = {"type": "openai_chat", "query": "Translate 'hello' to Spanish.", "answer": "hola"}
    value = json.dumps(data).encode()

    result = parse_gptcache_entry(_key("gptcache:abc"), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "Translate 'hello' to Spanish."
    assert result.response == "hola"
    assert result.source_model == _SOURCE_MODEL


def test_parse_gptcache_question_answer_format():
    """This test proves that a GPTCache question/answer dict produces a PromptPair."""
    data = {"question": "What is the boiling point of water?", "answer": "100°C at sea level."}
    value = json.dumps(data).encode()

    result = parse_gptcache_entry(_key("gptcache:def"), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "What is the boiling point of water?"
    assert result.response == "100°C at sea level."


def test_parse_gptcache_prompt_response_format():
    """This test proves that a GPTCache prompt/response dict produces a PromptPair."""
    data = {"prompt": "List three planets.", "response": "Mars, Venus, Jupiter."}
    value = json.dumps(data).encode()

    result = parse_gptcache_entry(_key("gptcache:ghi"), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "List three planets."
    assert result.response == "Mars, Venus, Jupiter."


# ---------------------------------------------------------------------------
# GPTCache parser — None / failure cases
# ---------------------------------------------------------------------------


def test_parse_gptcache_returns_none_for_invalid_json():
    """This test proves that malformed JSON bytes return None without raising."""
    result = parse_gptcache_entry(_key(), b"{{broken", _SOURCE_MODEL)

    assert result is None, f"Expected None for invalid JSON, got: {result!r}"


def test_parse_gptcache_returns_none_when_no_recognized_format():
    """This test proves that an entry with no recognized keys returns None."""
    data = {"foo": "bar", "baz": 42}
    value = json.dumps(data).encode()

    result = parse_gptcache_entry(_key(), value, _SOURCE_MODEL)

    assert result is None, f"Expected None for unrecognized format, got: {result!r}"


def test_parse_gptcache_falls_back_to_litellm_format():
    """This test proves that a GPTCache entry in LiteLLM format falls back correctly."""
    data = {
        "messages": [{"role": "user", "content": "Fallback query"}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "Fallback answer"}}]},
    }
    value = json.dumps(data).encode()

    result = parse_gptcache_entry(_key("gptcache:key"), value, _SOURCE_MODEL)

    assert isinstance(result, PromptPair), f"Expected PromptPair, got: {type(result)}"
    assert result.prompt == "Fallback query", f"Unexpected prompt: {result.prompt!r}"
    assert result.response == "Fallback answer", f"Unexpected response: {result.response!r}"
