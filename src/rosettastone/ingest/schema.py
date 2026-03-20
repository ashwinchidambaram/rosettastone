"""Pydantic models for the universal JSONL schema."""

from typing import Any, Optional

from pydantic import BaseModel, model_validator


class PromptPairInput(BaseModel):
    """Schema for a single line in the JSONL input file."""

    prompt: str | list[dict[str, Any]]
    response: str | dict[str, Any]
    source_model: str

    # Optional enrichment
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = {}
    feedback: Optional[str] = None

    @model_validator(mode="after")
    def normalize_response(self) -> "PromptPairInput":
        """Ensure response is always a string."""
        if isinstance(self.response, dict):
            self.response = self.response.get("content", str(self.response))
        return self
