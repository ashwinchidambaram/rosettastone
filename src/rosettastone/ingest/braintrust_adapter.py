"""Braintrust data adapter for ingesting LLM experiment log entries."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from rosettastone.core.types import PromptPair
from rosettastone.ingest.base import DataAdapter

logger = logging.getLogger(__name__)


class BraintrustAdapter(DataAdapter):
    """Ingest PromptPairs from a Braintrust project's experiment logs.

    Fetches log entries from the specified Braintrust project, extracts
    ``input`` → prompt and ``output`` → response, and maps optional fields
    such as ``expected`` → feedback and metadata fields like model, tags,
    timestamps, and scores.

    Args:
        project_name: The Braintrust project name to query logs from.
        api_key: Braintrust API key. Falls back to the ``BRAINTRUST_API_KEY``
            environment variable when not provided.
        source_model: Fallback model name used when the log entry metadata
            does not contain a ``model`` field.
    """

    def __init__(
        self,
        project_name: str,
        api_key: str | None = None,
        source_model: str = "unknown",
    ) -> None:
        self._project_name = project_name
        self._api_key = api_key or os.environ.get("BRAINTRUST_API_KEY")
        self._source_model = source_model

    # ------------------------------------------------------------------
    # DataAdapter interface
    # ------------------------------------------------------------------

    def load(self) -> list[PromptPair]:
        """Fetch log entries from the Braintrust project and return PromptPairs.

        Entries missing either ``input`` or ``output`` fields are skipped.
        A structural warning (no content) is emitted when entries are skipped.
        """
        client = self._make_client()
        project = client.projects.retrieve(self._project_name)
        entries: list[dict[str, Any]] = project.logs.list()

        pairs: list[PromptPair] = []
        skipped = 0

        for entry in entries:
            pair = self._parse_log_entry(entry)
            if pair is not None:
                pairs.append(pair)
            else:
                skipped += 1

        if skipped:
            # Structural log only — no prompt/response content to avoid PII exposure.
            logger.warning(
                "Some Braintrust log entries could not be parsed and were skipped",
                extra={"skipped_count": skipped, "loaded_count": len(pairs)},
            )

        logger.info(
            "Braintrust load complete",
            extra={
                "project": self._project_name,
                "loaded_count": len(pairs),
                "skipped_count": skipped,
            },
        )

        return pairs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_client(self):  # type: ignore[return]
        """Create and return a Braintrust client, raising clearly if braintrust is not installed."""
        try:
            import braintrust  # noqa: PLC0415  (lazy import — braintrust is optional)
        except ImportError as exc:
            raise ImportError(
                "The 'braintrust' package is required to use BraintrustAdapter. "
                "Install it with: pip install braintrust"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key

        return braintrust.Braintrust(**kwargs)

    def _parse_log_entry(self, entry: dict[str, Any]) -> PromptPair | None:
        """Parse a single Braintrust log entry dict into a PromptPair.

        Returns None if the entry is missing required fields (input or output).
        Never logs prompt/response content — structural info only.
        """
        if "input" not in entry:
            return None
        if "output" not in entry:
            return None

        raw_input = entry["input"]
        raw_output = entry["output"]

        # Prompt: str or list[dict] (chat messages) — preserved as-is.
        prompt: str | list[dict[str, Any]] = raw_input

        # Response: must be a str; dicts are JSON-serialized.
        if isinstance(raw_output, dict):
            response: str = json.dumps(raw_output)
        else:
            response = str(raw_output)

        # Model: prefer entry metadata, fall back to adapter-level source_model.
        entry_metadata: dict[str, Any] = entry.get("metadata") or {}
        model: str = entry_metadata.get("model") or self._source_model

        # Build PromptPair metadata — structural fields only (no content).
        metadata: dict[str, Any] = {}

        # Entry ID for traceability.
        entry_id = entry.get("id")
        if entry_id is not None:
            metadata["entry_id"] = entry_id

        # Carry over known metadata sub-fields.
        for field in ("tags", "timestamp", "scores"):
            if field in entry_metadata:
                metadata[field] = entry_metadata[field]

        # expected → feedback (the "gold standard" reference output).
        feedback: str | None = None
        if "expected" in entry:
            expected_val = entry["expected"]
            feedback = expected_val if isinstance(expected_val, str) else json.dumps(expected_val)

        return PromptPair(
            prompt=prompt,
            response=response,
            source_model=model,
            metadata=metadata,
            feedback=feedback,
        )
