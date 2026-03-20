"""LiteLLM convenience wrappers and model info lookups."""

from __future__ import annotations

from typing import Any

import litellm


def get_model_info(model: str) -> dict[str, Any]:
    """Get model metadata from LiteLLM."""
    try:
        return dict(litellm.get_model_info(model))
    except Exception:
        return {}


def count_tokens(model: str, text: str) -> int:
    """Count tokens for a given model."""
    try:
        return int(litellm.token_counter(model=model, text=text))
    except Exception:
        # Rough fallback: ~4 chars per token
        return len(text) // 4
