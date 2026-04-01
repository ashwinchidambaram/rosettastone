"""OpenTelemetry OTLP JSON export adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rosettastone.core.types import PromptPair
from rosettastone.ingest.base import DataAdapter

logger = logging.getLogger(__name__)


class OTelAdapter(DataAdapter):
    """Load prompt/response pairs from OTLP JSON export files.

    Supports a single .json file or a directory of .json files (non-recursive).
    Parses ``gen_ai.*`` semantic convention attributes from span attributes and
    events. Spans without both prompt and completion are silently skipped.
    """

    def __init__(self, export_path: Path, source_model: str = "unknown") -> None:
        self.export_path = Path(export_path)
        self.source_model = source_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[PromptPair]:
        """Load PromptPairs from the configured OTLP export path."""
        if not self.export_path.exists():
            raise FileNotFoundError(f"Path not found: {self.export_path}")

        files: list[Path]
        if self.export_path.is_dir():
            files = sorted(self.export_path.glob("*.json"))
        else:
            files = [self.export_path]

        pairs: list[PromptPair] = []
        total_skipped = 0

        for file in files:
            file_pairs, file_skipped = self._load_file(file)
            pairs.extend(file_pairs)
            total_skipped += file_skipped

        if total_skipped:
            logger.warning(
                "Skipped spans during OTel ingest",
                extra={"skipped_count": total_skipped, "loaded_count": len(pairs)},
            )

        return pairs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> tuple[list[PromptPair], int]:
        """Parse one OTLP JSON file. Returns (pairs, skipped_count)."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Skipped malformed JSON file during OTel ingest",
                extra={"file": str(path)},
            )
            return [], 0

        return self._parse_spans(data)

    def _parse_spans(self, data: dict[str, Any]) -> tuple[list[PromptPair], int]:
        """Traverse the full OTLP envelope and extract PromptPairs."""
        pairs: list[PromptPair] = []
        skipped = 0

        for resource_span in data.get("resourceSpans", []):
            for scope_span in resource_span.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    pair = self._extract_pair_from_span(span)
                    if pair is not None:
                        pairs.append(pair)
                    else:
                        skipped += 1

        return pairs, skipped

    def _extract_pair_from_span(self, span: dict[str, Any]) -> PromptPair | None:
        """Extract a PromptPair from a single OTLP span, or return None to skip."""
        attrs = self._attrs_to_dict(span.get("attributes", []))

        # Only process spans that contain at least one gen_ai.* attribute or event
        has_gen_ai_attrs = any(k.startswith("gen_ai.") for k in attrs)
        has_gen_ai_events = self._has_gen_ai_events(span.get("events", []))

        if not has_gen_ai_attrs and not has_gen_ai_events:
            # Not a gen_ai span — skip silently without counting as skipped
            return None

        # Resolve prompt: inline attribute takes precedence over events
        prompt_raw: str | None = attrs.get("gen_ai.prompt")
        if prompt_raw is None:
            prompt_raw = self._extract_from_events(span.get("events", []), "gen_ai.content.prompt")

        # Resolve completion: inline attribute takes precedence over events
        completion: str | None = attrs.get("gen_ai.completion")
        if completion is None:
            completion = self._extract_from_events(
                span.get("events", []), "gen_ai.content.completion"
            )

        if prompt_raw is None or completion is None:
            # gen_ai span but missing prompt or completion — skip and count
            return None

        # Parse prompt: if it's a JSON-encoded list of messages, deserialise it
        prompt: str | list[dict[str, Any]]
        try:
            parsed = json.loads(prompt_raw)
            if isinstance(parsed, list):
                prompt = parsed
            else:
                prompt = prompt_raw
        except (json.JSONDecodeError, ValueError):
            prompt = prompt_raw

        model = attrs.get("gen_ai.request.model", self.source_model)

        metadata: dict[str, Any] = {}
        span_name = span.get("name")
        trace_id = span.get("traceId")
        if span_name is not None:
            metadata["span_name"] = span_name
        if trace_id is not None:
            metadata["trace_id"] = trace_id

        return PromptPair(
            prompt=prompt,
            response=completion,
            source_model=model,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Attribute / event utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _attrs_to_dict(attributes: list[dict[str, Any]]) -> dict[str, str]:
        """Convert OTLP attribute array to a flat {key: stringValue} dict."""
        result: dict[str, str] = {}
        for entry in attributes:
            key = entry.get("key", "")
            value = entry.get("value", {})
            if isinstance(value, dict) and "stringValue" in value:
                result[key] = value["stringValue"]
        return result

    @staticmethod
    def _has_gen_ai_events(events: list[dict[str, Any]]) -> bool:
        """Return True if any event name starts with 'gen_ai.'."""
        return any(e.get("name", "").startswith("gen_ai.") for e in events)

    @classmethod
    def _extract_from_events(cls, events: list[dict[str, Any]], event_name: str) -> str | None:
        """Find the first event with matching name and return its gen_ai attribute value."""
        for event in events:
            if event.get("name") != event_name:
                continue
            event_attrs = cls._attrs_to_dict(event.get("attributes", []))
            # The payload attribute key matches the event name prefix pattern:
            # gen_ai.content.prompt -> gen_ai.prompt
            # gen_ai.content.completion -> gen_ai.completion
            for value in event_attrs.values():
                return value
        return None
