"""Shadow log entry format — JSONL-compatible with RosettaStone ingestion."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ShadowLogEntry(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    prompt: str
    source_model: str
    target_model: str
    source_response: str
    target_response: str
    source_latency_ms: float = 0.0
    target_latency_ms: float = 0.0
    source_tokens: int = 0
    target_tokens: int = 0
    source_cost: float = 0.0
    target_cost: float = 0.0
    metadata: dict[str, Any] = {}

    def to_prompt_pair_dict(self) -> dict[str, Any]:
        """Convert to PromptPair-compatible dict for RosettaStone ingestion."""
        return {
            "prompt": self.prompt,
            "response": self.source_response,  # baseline = source response
            "source_model": self.source_model,
            "metadata": {
                "shadow_request_id": self.request_id,
                "target_response": self.target_response,
                **self.metadata,
            },
        }

    @classmethod
    def from_jsonl_line(cls, line: str) -> ShadowLogEntry:
        return cls.model_validate(json.loads(line))


def write_log_entry(entry: ShadowLogEntry, log_dir: Path) -> None:
    """Append a log entry to today's JSONL log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    log_file = log_dir / f"shadow_{date_str}.jsonl"
    with log_file.open("a") as f:
        f.write(entry.model_dump_json() + "\n")


def read_log_entries(log_dir: Path) -> list[ShadowLogEntry]:
    """Read all shadow log entries from all JSONL files in log_dir."""
    entries: list[ShadowLogEntry] = []
    for log_file in sorted(log_dir.glob("shadow_*.jsonl")):
        for line in log_file.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(ShadowLogEntry.from_jsonl_line(line))
                except Exception:
                    pass  # Skip malformed lines
    return entries
