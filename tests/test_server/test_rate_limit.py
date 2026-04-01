"""Tests for the sliding-window rate limiter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rosettastone.server.rate_limit import check_rate_limit, reset_for_testing


def _make_request(ip: str = "1.2.3.4") -> SimpleNamespace:
    """Create a mock request object for testing."""
    client = SimpleNamespace(host=ip)
    state = SimpleNamespace()
    return SimpleNamespace(client=client, state=state)


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """Reset rate limiter before each test."""
    reset_for_testing()
    yield
    reset_for_testing()


def test_rate_limit_allows_requests_within_window(monkeypatch):
    """Requests within the limit should not be rate-limited."""
    monkeypatch.setenv("ROSETTASTONE_RATE_LIMIT", "3")
    request = _make_request()

    # All 3 requests should pass
    is_limited_1, retry_after_1 = check_rate_limit(request, "test")
    assert not is_limited_1
    assert retry_after_1 == 0

    is_limited_2, retry_after_2 = check_rate_limit(request, "test")
    assert not is_limited_2
    assert retry_after_2 == 0

    is_limited_3, retry_after_3 = check_rate_limit(request, "test")
    assert not is_limited_3
    assert retry_after_3 == 0


def test_rate_limit_blocks_on_limit_exceeded(monkeypatch):
    """The 4th request with limit=3 should be rate-limited."""
    monkeypatch.setenv("ROSETTASTONE_RATE_LIMIT", "3")
    request = _make_request()

    # First 3 should pass
    for i in range(3):
        is_limited, retry_after = check_rate_limit(request, "test")
        assert not is_limited, f"Request {i + 1} should not be limited"
        assert retry_after == 0

    # 4th should be blocked
    is_limited, retry_after = check_rate_limit(request, "test")
    assert is_limited
    assert retry_after > 0


def test_rate_limit_retry_after_is_positive(monkeypatch):
    """When limited, retry_after should be a positive integer."""
    monkeypatch.setenv("ROSETTASTONE_RATE_LIMIT", "2")
    request = _make_request()

    # Fill the limit
    check_rate_limit(request, "test")
    check_rate_limit(request, "test")

    # Next request should be limited with positive retry_after
    is_limited, retry_after = check_rate_limit(request, "test")
    assert is_limited
    assert isinstance(retry_after, int)
    assert retry_after > 0


def test_rate_limit_different_users_have_independent_limits(monkeypatch):
    """Different IP addresses should have independent rate limits."""
    monkeypatch.setenv("ROSETTASTONE_RATE_LIMIT", "3")

    request_1 = _make_request(ip="1.2.3.4")
    request_2 = _make_request(ip="5.6.7.8")

    # Request 1: make 3 requests (limit reached)
    for i in range(3):
        is_limited, _ = check_rate_limit(request_1, "test")
        assert not is_limited

    # Request 2: should still be able to make 3 requests independently
    for i in range(3):
        is_limited, _ = check_rate_limit(request_2, "test")
        assert not is_limited, f"User 2 request {i + 1} should not be limited"

    # Request 1: next request should be blocked
    is_limited_1, _ = check_rate_limit(request_1, "test")
    assert is_limited_1

    # Request 2: next request should also be blocked
    is_limited_2, _ = check_rate_limit(request_2, "test")
    assert is_limited_2
