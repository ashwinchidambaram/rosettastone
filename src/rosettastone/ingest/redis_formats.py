"""Parsers for Redis cache entry formats used by LLM proxies."""

from __future__ import annotations

import json
from typing import Any

from rosettastone.core.types import PromptPair


def parse_litellm_entry(key: bytes, value: bytes, source_model: str) -> PromptPair | None:
    """Parse a LiteLLM cache entry into a PromptPair.

    LiteLLM cache format stores:
    - ``messages``: list of role/content dicts (the request messages)
    - ``response``: the completion response object

    Returns None on any parse failure — callers should log and skip.
    """
    try:
        raw = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(raw, dict):
        return None

    data: dict[str, Any] = raw
    messages: list[dict[str, Any]] | None = data.get("messages")
    if not messages or not isinstance(messages, list):
        return None

    # Extract prompt from the last user message
    prompt: str | None = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content:
                prompt = content
                break

    if prompt is None:
        return None

    # Extract assistant response text
    response_text: str | None = None
    response_obj = data.get("response")
    if isinstance(response_obj, dict):
        # OpenAI-style: response.choices[0].message.content
        choices = response_obj.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        response_text = content
    elif isinstance(response_obj, str):
        response_text = response_obj

    if not response_text:
        return None

    return PromptPair(
        prompt=prompt,
        response=response_text,
        source_model=source_model,
    )


def parse_langchain_entry(key: bytes, value: bytes, source_model: str) -> PromptPair | None:
    """Parse a LangChain RedisCache / RedisSemanticCache entry into a PromptPair.

    Tries two known LangChain cache value structures:
    1. Direct: ``{"input": "...", "output": "..."}``
    2. LLMResult: ``{"generations": [[{"text": "...", "message": {...}}]], ...}``
       where prompt is embedded in ``generations[0][0]["message"]["content"]``.

    Returns None on any parse failure — callers should log and skip.
    """
    try:
        raw = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(raw, dict):
        return None

    data: dict[str, Any] = raw

    # Structure 1: simple input/output dict
    direct_input = data.get("input")
    direct_output = data.get("output")
    if (
        isinstance(direct_input, str)
        and direct_input
        and isinstance(direct_output, str)
        and direct_output
    ):  # noqa: E501
        return PromptPair(prompt=direct_input, response=direct_output, source_model=source_model)

    # Structure 2: LLMResult with generations list
    generations = data.get("generations")
    if not isinstance(generations, list) or not generations:
        return None

    first_list = generations[0]
    if not isinstance(first_list, list) or not first_list:
        return None

    first_gen = first_list[0]
    if not isinstance(first_gen, dict):
        return None

    response_text: str | None = first_gen.get("text")
    if not isinstance(response_text, str) or not response_text:
        return None

    # Try to find prompt in message.content (SemanticCache stores it there)
    prompt: str | None = None
    message = first_gen.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content:
            prompt = content

    # Also check top-level "prompt" or "input" as fallback
    if prompt is None:
        for key_name in ("prompt", "input"):
            candidate = data.get(key_name)
            if isinstance(candidate, str) and candidate:
                prompt = candidate
                break

    if prompt is None:
        return None

    return PromptPair(prompt=prompt, response=response_text, source_model=source_model)


def parse_redisvl_entry(key: bytes, value: bytes, source_model: str) -> PromptPair | None:
    """Parse a RedisVL SemanticCache entry into a PromptPair.

    RedisVL stores entries as Redis Hashes (HSET). When fetched with GET the
    value is a JSON string with common shapes:
    - ``{"prompt": "...", "response": "...", "metadata": {...}}``
    - ``{"input_text": "...", "response": "...", "vector_score": 0.95}``

    Returns None on any parse failure — callers should log and skip.
    """
    try:
        raw = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(raw, dict):
        return None

    data: dict[str, Any] = raw

    prompt: str | None = None
    for key_name in ("prompt", "input_text"):
        candidate = data.get(key_name)
        if isinstance(candidate, str) and candidate:
            prompt = candidate
            break

    if prompt is None:
        return None

    response_text: str | None = None
    for key_name in ("response", "output"):
        candidate = data.get(key_name)
        if isinstance(candidate, str) and candidate:
            response_text = candidate
            break

    if not response_text:
        return None

    return PromptPair(prompt=prompt, response=response_text, source_model=source_model)


def parse_gptcache_entry(key: bytes, value: bytes, source_model: str) -> PromptPair | None:
    """Parse a GPTCache entry into a PromptPair.

    GPTCache stores semantic cache entries in several formats:
    1. ``{"query": "...", "answer": "..."}`` (with optional ``"type": "openai_chat"``)
    2. ``{"question": "...", "answer": "..."}``
    3. ``{"prompt": "...", "response": "..."}``
    4. LiteLLM-style OpenAI compat: ``{"messages": [...], "response": {...}}``

    Returns None on any parse failure — callers should log and skip.
    """
    try:
        raw = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(raw, dict):
        return None

    data: dict[str, Any] = raw

    answer: str | None = data.get("answer") if isinstance(data.get("answer"), str) else None

    # Try query/answer
    query = data.get("query")
    if isinstance(query, str) and query and isinstance(answer, str) and answer:
        return PromptPair(prompt=query, response=answer, source_model=source_model)

    # Try question/answer
    question = data.get("question")
    if isinstance(question, str) and question and isinstance(answer, str) and answer:
        return PromptPair(prompt=question, response=answer, source_model=source_model)

    # Try prompt/response
    prompt = data.get("prompt")
    response_val = data.get("response")
    if isinstance(prompt, str) and prompt and isinstance(response_val, str) and response_val:
        return PromptPair(prompt=prompt, response=response_val, source_model=source_model)

    # Fall back to LiteLLM-style messages + response object
    return parse_litellm_entry(key, value, source_model)
