"""CSV/TSV file adapter."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from rosettastone.core.types import PromptPair
from rosettastone.ingest.base import DataAdapter

logger = logging.getLogger(__name__)


@dataclass
class CSVColumnMapping:
    """Column name mapping for CSV/TSV files."""

    prompt_col: str = "prompt"
    response_col: str = "response"
    source_model_col: str = "source_model"
    metadata_cols: list[str] | None = None
    feedback_col: str | None = None


class CSVAdapter(DataAdapter):
    """Adapter for loading prompt/response pairs from CSV or TSV files."""

    def __init__(
        self,
        path: Path,
        column_mapping: CSVColumnMapping | None = None,
        delimiter: str | None = None,
        source_model: str = "unknown",
    ) -> None:
        self.path = Path(path)
        self.column_mapping = column_mapping or CSVColumnMapping()
        self._explicit_delimiter = delimiter
        self.source_model = source_model

    def _detect_delimiter(self) -> str:
        """Detect delimiter from file extension. .tsv/.tab -> tab, otherwise comma."""
        suffix = self.path.suffix.lower()
        if suffix in (".tsv", ".tab"):
            return "\t"
        return ","

    def load(self) -> list[PromptPair]:
        """Load prompt/response pairs from the CSV/TSV file."""
        if self._explicit_delimiter is not None:
            delimiter = self._explicit_delimiter
        else:
            delimiter = self._detect_delimiter()
        mapping = self.column_mapping
        pairs: list[PromptPair] = []

        with open(self.path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fieldnames = reader.fieldnames or []

            # Validate required columns exist
            if mapping.prompt_col not in fieldnames:
                raise ValueError(
                    f"Required column '{mapping.prompt_col}' not found in CSV. "
                    f"Available columns: {list(fieldnames)}"
                )
            if mapping.response_col not in fieldnames:
                raise ValueError(
                    f"Required column '{mapping.response_col}' not found in CSV. "
                    f"Available columns: {list(fieldnames)}"
                )

            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                prompt_val = row.get(mapping.prompt_col, "").strip()
                response_val = row.get(mapping.response_col, "").strip()

                # Skip rows where either prompt or response is empty
                if not prompt_val and not response_val:
                    logger.debug("Skipping blank row at row %d", row_num)
                    continue
                if not prompt_val or not response_val:
                    logger.debug(
                        "Skipping row %d with missing %s",
                        row_num,
                        "prompt" if not prompt_val else "response",
                    )
                    continue

                # Determine source_model
                source_model_val: str
                col_present = mapping.source_model_col in (fieldnames or [])
                col_value = row.get(mapping.source_model_col, "").strip() if col_present else ""
                if col_present and col_value:
                    source_model_val = col_value
                else:
                    source_model_val = self.source_model

                # Collect metadata columns
                metadata: dict[str, str] = {}
                if mapping.metadata_cols:
                    for col in mapping.metadata_cols:
                        if col in row:
                            metadata[col] = row[col].strip()

                # Collect feedback column
                feedback: str | None = None
                if mapping.feedback_col and mapping.feedback_col in row:
                    feedback_val = row[mapping.feedback_col].strip()
                    feedback = feedback_val if feedback_val else None

                pairs.append(
                    PromptPair(
                        prompt=prompt_val,
                        response=response_val,
                        source_model=source_model_val,
                        metadata=metadata,
                        feedback=feedback,
                    )
                )

        logger.info("Loaded %d prompt pairs from %s", len(pairs), self.path.name)
        return pairs
