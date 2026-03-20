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
