"""Sliding-window rate limiter for submission endpoints."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

# Default: 10 submissions per hour
_DEFAULT_LIMIT = 10
_DEFAULT_WINDOW_SECONDS = 3600


def _get_limit() -> int:
    """Read rate limit from env var. Default 10."""
    try:
        return int(os.environ.get("ROSETTASTONE_RATE_LIMIT", str(_DEFAULT_LIMIT)))
    except ValueError:
        return _DEFAULT_LIMIT


# keyed by (user_id_or_ip, endpoint) -> list of timestamps
_windows: dict[tuple[str, str], list[float]] = defaultdict(list)
_lock = threading.Lock()


def _get_key(request) -> str:
    """Return a stable key for the current requester.

    Multi-user mode: use user_id. Single-user: use client IP.
    """
    import os as _os

    multi_user = _os.environ.get("ROSETTASTONE_MULTI_USER", "").lower() in ("1", "true", "yes")
    if multi_user:
        user = getattr(request.state, "user", None)
        if user:
            if isinstance(user, dict):
                uid = user.get("user_id") or user.get("id")
            else:
                uid = getattr(user, "user_id", None) or getattr(user, "id", None)
            if uid:
                return f"user:{uid}"
    # Fall back to IP
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def check_rate_limit(request, endpoint: str = "submit") -> tuple[bool, int]:
    """Check if the request should be rate-limited.

    Returns (is_limited: bool, retry_after_seconds: int).
    If not limited, records the attempt.
    """
    limit = _get_limit()
    window = _DEFAULT_WINDOW_SECONDS
    now = time.time()
    key = (_get_key(request), endpoint)

    with _lock:
        # Prune expired entries
        cutoff = now - window
        _windows[key] = [t for t in _windows[key] if t > cutoff]

        if len(_windows[key]) >= limit:
            # Calculate retry-after: when the oldest entry expires
            oldest = min(_windows[key])
            retry_after = int(oldest + window - now) + 1
            return True, retry_after

        # Record this attempt
        _windows[key].append(now)
        return False, 0


def reset_for_testing() -> None:
    """Clear all windows — for test isolation only."""
    with _lock:
        _windows.clear()
