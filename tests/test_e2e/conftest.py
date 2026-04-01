"""Fixtures for E2E tests requiring real APIs and Redis."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest


def pytest_configure(config):  # type: ignore[no-untyped-def]
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring external services")
    config.addinivalue_line("markers", "ollama: tests requiring local Ollama instance")
    config.addinivalue_line("markers", "ray: tests requiring Ray cluster")


@pytest.fixture(scope="session")
def redis_url() -> str:
    """Redis URL for E2E tests. Uses DB 15 to isolate from dev data."""
    return os.environ.get("ROSETTASTONE_E2E_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture(scope="session")
def redis_client(redis_url: str):
    """Session-scoped Redis client. Skips all E2E tests if Redis is unavailable."""
    try:
        import redis as redis_lib
    except ImportError:
        pytest.skip("redis package not installed")

    client = redis_lib.from_url(redis_url, decode_responses=False)
    try:
        client.ping()
    except redis_lib.ConnectionError:
        pytest.skip(f"Redis unavailable at {redis_url}")

    return client


@pytest.fixture
def clean_redis(redis_client) -> Generator[None, None, None]:
    """Flush the E2E Redis DB before and after each test."""
    redis_client.flushdb()
    yield
    redis_client.flushdb()


@pytest.fixture(scope="session")
def generated_data_cache() -> dict:
    """Session-scoped cache to avoid regenerating data for the same source model."""
    return {}
