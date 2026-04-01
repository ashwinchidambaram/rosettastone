"""Populate Redis with synthetic data in LiteLLM cache format."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import cast

from rosettastone.testing.synth_data import GeneratedPair

logger = logging.getLogger(__name__)


class RedisPopulator:
    """Write generated pairs to Redis in LiteLLM cache format."""

    def __init__(self, redis_url: str, key_prefix: str = "litellm") -> None:
        import redis as redis_lib

        self.client = redis_lib.from_url(redis_url, decode_responses=False)
        self.key_prefix = key_prefix

    def populate(self, data: list[GeneratedPair]) -> int:
        """Write pairs to Redis. Returns number of keys written."""
        count = 0
        for pair in data:
            cache_key = self._make_key(pair.messages)
            value = json.dumps(
                {
                    "messages": pair.messages,
                    "response": {
                        "choices": [
                            {"message": {"role": "assistant", "content": pair.response_text}}
                        ]
                    },
                }
            )
            self.client.set(cache_key, value.encode())
            count += 1

        logger.info("Populated %d keys in Redis (prefix=%s)", count, self.key_prefix)
        return count

    def cleanup(self, pattern: str | None = None) -> int:
        """Delete keys matching pattern. Returns number deleted."""
        pat = pattern or f"{self.key_prefix}:*"
        keys = self.client.keys(pat)
        if keys:
            deleted = self.client.delete(*list(keys))  # type: ignore[arg-type]
            logger.info("Cleaned up %d Redis keys matching %s", deleted, pat)
            return cast(int, deleted)
        return 0

    def _make_key(self, messages: list[dict[str, str]]) -> str:
        digest = hashlib.md5(json.dumps(messages).encode()).hexdigest()
        return f"{self.key_prefix}:{digest}"

    @staticmethod
    def write_jsonl(data: list[GeneratedPair], path: Path, source_model: str) -> int:
        """Write generated pairs as JSONL file (fallback for non-Redis tests)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w") as f:
            for pair in data:
                # Extract prompt from messages (last user message)
                prompt: str | list[dict[str, str]] = ""
                for msg in reversed(pair.messages):
                    if msg["role"] == "user":
                        prompt = msg["content"]
                        break

                record = {
                    "prompt": prompt,
                    "response": pair.response_text,
                    "source_model": source_model,
                    "metadata": {"output_type": pair.output_type},
                }
                f.write(json.dumps(record) + "\n")
                count += 1

        logger.info("Wrote %d pairs to %s", count, path)
        return count
