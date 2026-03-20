"""Tests for RedisAdapter."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from rosettastone.core.types import PromptPair
from rosettastone.ingest.redis_adapter import RedisAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SOURCE_MODEL = "openai/gpt-4o"
_REDIS_URL = "redis://localhost:6379/0"


def _litellm_value(prompt: str, response: str = "Answer") -> bytes:
    """Return a minimal LiteLLM cache entry as bytes."""
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "response": {"choices": [{"message": {"role": "assistant", "content": response}}]},
    }
    return json.dumps(data).encode()


def _make_mock_redis(entries: dict[bytes, bytes]) -> MagicMock:
    """Build a mock Redis client whose scan/get mirrors ``entries``.

    ``scan`` is set up as a callable that always returns ``(0, all_keys)`` so
    both the format-detection pass and the collection pass see the same data
    regardless of how many times ``_scan_keys`` is called internally.
    """
    keys = list(entries.keys())
    client = MagicMock()
    # Return (cursor=0, keys) on every call — idempotent for multiple passes.
    client.scan.side_effect = lambda cursor: (0, keys)
    client.get.side_effect = lambda k: entries.get(k)
    return client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_returns_prompt_pairs_for_litellm_entries():
    """This test proves that valid LiteLLM cache entries are loaded as PromptPairs."""
    entries = {
        b"key:1": _litellm_value("What is 2+2?", "4"),
        b"key:2": _litellm_value("Capital of France?", "Paris"),
    }
    mock_client = _make_mock_redis(entries)

    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 2, f"Expected 2 pairs, got {len(result)}"
    assert all(isinstance(p, PromptPair) for p in result), (
        "Expected all results to be PromptPair instances"
    )
    prompts = {p.prompt for p in result}
    assert "What is 2+2?" in prompts, f"Missing expected prompt; got: {prompts!r}"
    assert "Capital of France?" in prompts, f"Missing expected prompt; got: {prompts!r}"


def test_load_single_entry_has_correct_source_model():
    """This test proves that the source_model on each PromptPair matches the adapter config."""
    entries = {b"key:1": _litellm_value("Hello?")}
    mock_client = _make_mock_redis(entries)

    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert result[0].source_model == _SOURCE_MODEL, (
        f"Expected source_model={_SOURCE_MODEL!r}, got {result[0].source_model!r}"
    )


def test_load_uses_scan_not_keys():
    """This test proves that SCAN is used (client.scan called, client.keys never called)."""
    entries = {b"key:1": _litellm_value("Test?")}
    mock_client = _make_mock_redis(entries)

    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        adapter.load()

    mock_client.scan.assert_called()
    mock_client.keys.assert_not_called()  # type: ignore[attr-defined]


def test_scan_multiple_cursor_iterations():
    """This test proves that SCAN pagination (non-zero cursor) is followed until cursor=0."""
    batch1_keys = [b"key:1", b"key:2"]
    batch2_keys = [b"key:3"]
    all_entries = {
        b"key:1": _litellm_value("Q1", "A1"),
        b"key:2": _litellm_value("Q2", "A2"),
        b"key:3": _litellm_value("Q3", "A3"),
    }

    # Single pass: (cursor=42, batch1), (cursor=0, batch2)
    mock_client = MagicMock()
    mock_client.scan.side_effect = [
        (42, batch1_keys),
        (0, batch2_keys),
    ]
    mock_client.get.side_effect = lambda k: all_entries.get(k)

    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 3, f"Expected 3 pairs from 2 SCAN batches, got {len(result)}"
    # Single pass: 2 cursor iterations = 2 total scan calls
    assert mock_client.scan.call_count == 2, (
        f"Expected 2 scan calls (single pass, 2 cursor iterations), got {mock_client.scan.call_count}"
    )


# ---------------------------------------------------------------------------
# Mixed formats / unparseable entries
# ---------------------------------------------------------------------------


def test_unparseable_entries_are_skipped_not_raised():
    """This test proves that unparseable Redis values are skipped silently (no exception)."""
    entries = {
        b"key:good": _litellm_value("Good prompt", "Good response"),
        b"key:bad": b"this is not valid json at all",
        b"key:also-bad": b'{"no_messages": true}',
    }
    mock_client = _make_mock_redis(entries)

    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1, f"Expected 1 parseable pair (bad entries skipped), got {len(result)}"
    assert result[0].prompt == "Good prompt", f"Unexpected prompt: {result[0].prompt!r}"


def test_mixed_format_logs_warning_for_skipped(caplog):
    """This test proves that a warning is logged (structurally) when entries are skipped."""
    entries = {
        b"key:good": _litellm_value("Valid"),
        b"key:bad": b"garbage",
    }
    mock_client = _make_mock_redis(entries)

    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        with caplog.at_level(logging.WARNING, logger="rosettastone.ingest.redis_adapter"):
            adapter.load()

    assert any("skipped" in record.message.lower() for record in caplog.records), (
        "Expected a warning log mentioning 'skipped' for unparseable entries"
    )


# ---------------------------------------------------------------------------
# Zero matches
# ---------------------------------------------------------------------------


def test_zero_matches_raises_value_error():
    """This test proves that a Redis with no parseable entries raises ValueError."""
    entries = {
        b"key:1": b"not json",
        b"key:2": b'{"unrecognized": "format"}',
    }
    mock_client = _make_mock_redis(entries)

    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        with pytest.raises(ValueError, match="No recognizable cache format"):
            adapter.load()


def test_empty_redis_raises_value_error():
    """This test proves that an empty Redis instance raises ValueError."""
    mock_client = MagicMock()
    mock_client.scan.side_effect = lambda cursor: (0, [])

    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        with pytest.raises(ValueError, match="No recognizable cache format"):
            adapter.load()


# ---------------------------------------------------------------------------
# Import error handling
# ---------------------------------------------------------------------------


def test_import_error_raises_clear_message():
    """This test proves that a missing 'redis' package raises ImportError with guidance."""
    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)

    with patch.dict("sys.modules", {"redis": None}):
        with pytest.raises(ImportError, match="pip install redis"):
            adapter._make_client()


# ---------------------------------------------------------------------------
# get() returns None (key expired between scan and get)
# ---------------------------------------------------------------------------


def test_none_get_result_skipped_gracefully():
    """This test proves that keys returning None (expired after SCAN) are silently skipped."""
    good_key = b"key:good"
    expired_key = b"key:expired"
    entries = {good_key: _litellm_value("Still here", "Yes"), expired_key: None}  # type: ignore[dict-item]

    mock_client = MagicMock()
    mock_client.scan.side_effect = lambda cursor: (0, [good_key, expired_key])
    mock_client.get.side_effect = lambda k: entries.get(k)

    adapter = RedisAdapter(_REDIS_URL, _SOURCE_MODEL)
    with patch.object(adapter, "_make_client", return_value=mock_client):
        result = adapter.load()

    assert len(result) == 1, f"Expected 1 result (expired key skipped), got {len(result)}"
