"""Redis adapter integration tests — requires a running Redis instance.

All tests are marked @pytest.mark.e2e and will skip gracefully if
Redis is not available. Run with:
    docker compose --profile redis up -d
    uv run pytest tests/test_ingest/test_redis_integration.py -v -m e2e
"""

from __future__ import annotations

import json
import os
import time

import pytest

from rosettastone.ingest.redis_adapter import RedisAdapter

# Mark entire module as e2e
pytestmark = pytest.mark.e2e

_SOURCE_MODEL = "test-model"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_url():
    return os.environ.get("ROSETTASTONE_E2E_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
def redis_client(redis_url):
    """Connect to real Redis, skip if unavailable."""
    try:
        import redis as redis_lib
    except ImportError:
        pytest.skip("redis package not installed")

    client = redis_lib.from_url(redis_url)
    try:
        client.ping()
    except redis_lib.ConnectionError:
        pytest.skip("Redis not available — start with: docker compose --profile redis up -d")

    client.flushdb()  # clean state before test
    yield client
    client.flushdb()  # clean after test
    client.close()


# ---------------------------------------------------------------------------
# Helpers to populate Redis with each cache format
# ---------------------------------------------------------------------------


def _write_litellm_entries(client, count=5):
    """Write LiteLLM-format cache entries."""
    for i in range(count):
        key = f"litellm:test-{i}"
        value = json.dumps(
            {
                "messages": [{"role": "user", "content": f"Question {i}"}],
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": f"Answer {i}"}}]
                },
            }
        )
        client.set(key, value)


def _write_langchain_entries(client, count=3):
    """Write LangChain-format cache entries."""
    for i in range(count):
        key = f"langchain:test-{i}"
        value = json.dumps({"input": f"LC Question {i}", "output": f"LC Answer {i}"})
        client.set(key, value)


def _write_redisvl_entries(client, count=3):
    """Write RedisVL-format cache entries."""
    for i in range(count):
        key = f"redisvl:test-{i}"
        value = json.dumps(
            {
                "prompt": f"RVL Question {i}",
                "response": f"RVL Answer {i}",
                "metadata": {"source": "test"},
            }
        )
        client.set(key, value)


def _write_gptcache_entries(client, count=3):
    """Write GPTCache-format cache entries."""
    for i in range(count):
        key = f"gptcache:test-{i}"
        value = json.dumps({"query": f"GPC Question {i}", "answer": f"GPC Answer {i}"})
        client.set(key, value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_litellm_format_roundtrip(redis_client, redis_url):
    """Write 5 LiteLLM entries and verify all 5 are loaded as correct PromptPairs."""
    _write_litellm_entries(redis_client, count=5)

    adapter = RedisAdapter(redis_url=redis_url, source_model=_SOURCE_MODEL)
    pairs = adapter.load()

    assert len(pairs) == 5, f"Expected 5 pairs, got {len(pairs)}"
    prompts = {p.prompt for p in pairs}
    for i in range(5):
        assert f"Question {i}" in prompts, f"Missing prompt 'Question {i}' in {prompts!r}"
    responses = {p.response for p in pairs}
    for i in range(5):
        assert f"Answer {i}" in responses, f"Missing response 'Answer {i}' in {responses!r}"
    assert all(p.source_model == _SOURCE_MODEL for p in pairs), (
        "All pairs should have source_model matching the adapter config"
    )


def test_langchain_format_loads(redis_client, redis_url):
    """Write 3 LangChain entries and verify all 3 are loaded as PromptPairs."""
    _write_langchain_entries(redis_client, count=3)

    adapter = RedisAdapter(redis_url=redis_url, source_model=_SOURCE_MODEL)
    pairs = adapter.load()

    assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"
    prompts = {p.prompt for p in pairs}
    for i in range(3):
        assert f"LC Question {i}" in prompts, f"Missing 'LC Question {i}' in {prompts!r}"


def test_redisvl_format_loads(redis_client, redis_url):
    """Write 3 RedisVL entries and verify all 3 are loaded as PromptPairs."""
    _write_redisvl_entries(redis_client, count=3)

    adapter = RedisAdapter(redis_url=redis_url, source_model=_SOURCE_MODEL)
    pairs = adapter.load()

    assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"
    prompts = {p.prompt for p in pairs}
    for i in range(3):
        assert f"RVL Question {i}" in prompts, f"Missing 'RVL Question {i}' in {prompts!r}"


def test_gptcache_format_loads(redis_client, redis_url):
    """Write 3 GPTCache entries and verify all 3 are loaded as PromptPairs."""
    _write_gptcache_entries(redis_client, count=3)

    adapter = RedisAdapter(redis_url=redis_url, source_model=_SOURCE_MODEL)
    pairs = adapter.load()

    assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"
    prompts = {p.prompt for p in pairs}
    for i in range(3):
        assert f"GPC Question {i}" in prompts, f"Missing 'GPC Question {i}' in {prompts!r}"


def test_mixed_format_majority_wins(redis_client, redis_url):
    """LiteLLM (5 entries) should win format detection over LangChain (2 entries).

    The LiteLLM entries must all be returned. The LangChain entries may or may
    not parse depending on whether the winning parser can handle them, but at
    minimum the 5 LiteLLM pairs must be present.
    """
    _write_litellm_entries(redis_client, count=5)
    _write_langchain_entries(redis_client, count=2)

    adapter = RedisAdapter(redis_url=redis_url, source_model=_SOURCE_MODEL)
    pairs = adapter.load()

    # The LiteLLM entries must all be returned
    assert len(pairs) >= 5, f"Expected at least 5 pairs (LiteLLM wins), got {len(pairs)}"
    litellm_prompts = {f"Question {i}" for i in range(5)}
    loaded_prompts = {p.prompt for p in pairs}
    for expected in litellm_prompts:
        assert expected in loaded_prompts, (
            f"LiteLLM prompt {expected!r} missing from loaded pairs; got {loaded_prompts!r}"
        )


def test_empty_redis_raises_valueerror(redis_client, redis_url):
    """An empty Redis DB should raise ValueError (no recognizable format)."""
    # redis_client fixture already flushdb'd — nothing written
    adapter = RedisAdapter(redis_url=redis_url, source_model=_SOURCE_MODEL)
    with pytest.raises(ValueError, match="No recognizable cache format"):
        adapter.load()


def test_large_dataset_pagination(redis_client, redis_url):
    """Write 200 LiteLLM entries to verify SCAN pagination works with real Redis."""
    _write_litellm_entries(redis_client, count=200)

    adapter = RedisAdapter(redis_url=redis_url, source_model=_SOURCE_MODEL)
    pairs = adapter.load()

    assert len(pairs) == 200, f"Expected 200 pairs from paginated SCAN, got {len(pairs)}"


def test_expired_key_handled(redis_client, redis_url):
    """A key that expires between SCAN and GET should be silently skipped.

    Write 3 entries, set entry 1 to expire in 1 second, sleep 1.5 s, then
    load — only 2 pairs should be returned.
    """
    _write_litellm_entries(redis_client, count=3)

    # Set key 1 to expire in 1 second
    redis_client.expire("litellm:test-1", 1)

    time.sleep(1.5)

    adapter = RedisAdapter(redis_url=redis_url, source_model=_SOURCE_MODEL)
    pairs = adapter.load()

    assert len(pairs) == 2, f"Expected 2 pairs after one key expired, got {len(pairs)}"
    loaded_prompts = {p.prompt for p in pairs}
    assert "Question 1" not in loaded_prompts, "Expired key's prompt should not appear in results"
