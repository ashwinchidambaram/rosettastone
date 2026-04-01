"""Redis cache adapter for ingesting LLM proxy cache entries."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from rosettastone.core.types import PromptPair
from rosettastone.ingest.base import DataAdapter
from rosettastone.ingest.redis_formats import (
    parse_gptcache_entry,
    parse_langchain_entry,
    parse_litellm_entry,
    parse_redisvl_entry,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Number of keys to sample when auto-detecting the cache format.
_SAMPLE_SIZE = 100

# Registry of (format_name, parser_callable) pairs.
# Each parser has signature: (key: bytes, value: bytes, source_model: str) -> PromptPair | None
_PARSERS: list[tuple[str, object]] = [
    ("litellm", parse_litellm_entry),
    ("langchain", parse_langchain_entry),
    ("redisvl", parse_redisvl_entry),
    ("gptcache", parse_gptcache_entry),
]


class RedisAdapter(DataAdapter):
    """Ingest PromptPairs from a Redis LLM-proxy cache.

    Uses ``SCAN`` to iterate keys without blocking Redis, then auto-detects
    the cache format by sampling the first ``_SAMPLE_SIZE`` keys.
    """

    def __init__(self, redis_url: str, source_model: str) -> None:
        self._redis_url = redis_url
        self._source_model = source_model

    # ------------------------------------------------------------------
    # DataAdapter interface
    # ------------------------------------------------------------------

    def load(self) -> list[PromptPair]:
        """Scan Redis and return all parseable PromptPairs.

        Single-pass: the first _SAMPLE_SIZE keys are used for format detection,
        then all keys (including the sampled ones) are parsed with the winning parser.
        """
        client = self._make_client()

        # Single pass: collect all key/value pairs and detect format simultaneously
        all_entries: list[tuple[bytes, bytes]] = []
        scores: dict[str, int] = {name: 0 for name, _ in _PARSERS}
        sampled = 0

        for key in self._scan_keys(client):
            value = client.get(key)
            if value is None:
                continue
            all_entries.append((key, value))

            # Sample the first _SAMPLE_SIZE entries for format detection
            if sampled < _SAMPLE_SIZE:
                for name, parser in _PARSERS:
                    result = parser(key, value, self._source_model)  # type: ignore[operator]
                    if result is not None:
                        scores[name] += 1
                sampled += 1

        if sampled == 0 or all(v == 0 for v in scores.values()):
            raise ValueError("No recognizable cache format found in Redis")

        winning_name = max(scores, key=lambda k: scores[k])
        winning_parser = next(p for n, p in _PARSERS if n == winning_name)

        # Log format distribution for observability (structural only — no content).
        logger.info(
            "Redis format detection complete",
            extra={"sampled_keys": sampled, "format_scores": scores, "selected": winning_name},
        )

        # Parse all entries with the winning parser
        pairs: list[PromptPair] = []
        skipped = 0
        for key, value in all_entries:
            result: PromptPair | None = winning_parser(  # type: ignore[operator]
                key, value, self._source_model
            )
            if result is not None:
                pairs.append(result)
            else:
                skipped += 1

        if skipped:
            # Structural log only — no prompt content to avoid PII exposure.
            logger.warning(
                "Some Redis entries could not be parsed and were skipped",
                extra={"skipped_count": skipped, "loaded_count": len(pairs)},
            )

        return pairs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_client(self):  # type: ignore[return]
        """Create and return a Redis client, raising clearly if redis is not installed."""
        try:
            import redis  # noqa: PLC0415  (lazy import — redis is optional)
        except ImportError as exc:
            raise ImportError(
                "The 'redis' package is required to use RedisAdapter. "
                "Install it with: pip install redis"
            ) from exc

        return redis.from_url(self._redis_url)

    def _scan_keys(self, client) -> Iterator[bytes]:  # type: ignore[return]
        """Yield all keys via SCAN (non-blocking)."""
        cursor = 0
        while True:
            try:
                cursor, keys = client.scan(cursor)
            except StopIteration:
                # Guard against test mocks whose side_effect list is exhausted.
                break
            yield from keys
            if cursor == 0:
                break
