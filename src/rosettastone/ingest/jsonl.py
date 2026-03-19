"""JSONL file adapter."""

from __future__ import annotations

import json
from pathlib import Path

from rosettastone.core.types import PromptPair
from rosettastone.ingest.base import DataAdapter
from rosettastone.ingest.schema import PromptPairInput


class JSONLAdapter(DataAdapter):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> list[PromptPair]:
        pairs: list[PromptPair] = []
        with open(self.path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    validated = PromptPairInput.model_validate(raw)
                    pairs.append(
                        PromptPair(
                            prompt=validated.prompt,
                            response=str(validated.response),
                            source_model=validated.source_model,
                            metadata=validated.metadata,
                            feedback=validated.feedback,
                        )
                    )
                except Exception as e:
                    raise ValueError(f"Error parsing line {line_num}: {e}") from e
        return pairs
