"""LangSmith trace adapter for ingesting LLM run prompt/response pairs."""

from __future__ import annotations

import logging
import os
from typing import Any

from rosettastone.core.types import PromptPair
from rosettastone.ingest.base import DataAdapter

logger = logging.getLogger(__name__)


class LangSmithAdapter(DataAdapter):
    """Ingest PromptPairs from a LangSmith project's traced LLM runs.

    Uses ``client.list_runs`` with ``execution_order=1`` to retrieve only
    top-level (non-nested) runs, then parses each run's inputs/outputs into
    a :class:`~rosettastone.core.types.PromptPair`.

    Optional date filters (ISO-8601 strings) narrow the query window.
    The ``langsmith`` package is imported lazily so that the adapter can be
    constructed even when the package is not installed.
    """

    def __init__(
        self,
        project_name: str,
        api_key: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        source_model: str = "unknown",
    ) -> None:
        self._project_name = project_name
        self._api_key = api_key or os.environ.get("LANGCHAIN_API_KEY")
        self._start_date = start_date
        self._end_date = end_date
        self._source_model = source_model

    # ------------------------------------------------------------------
    # DataAdapter interface
    # ------------------------------------------------------------------

    def load(self) -> list[PromptPair]:
        """Fetch all top-level runs from the LangSmith project and return parsed PromptPairs.

        Runs that cannot be parsed (missing inputs, unrecognised format) are
        skipped. Only structural information is logged — never prompt content.
        """
        client = self._make_client()

        list_runs_kwargs: dict[str, Any] = {
            "project_name": self._project_name,
            "execution_order": 1,
        }
        if self._start_date is not None:
            list_runs_kwargs["start_time"] = self._start_date
        if self._end_date is not None:
            list_runs_kwargs["end_time"] = self._end_date

        runs = client.list_runs(**list_runs_kwargs)

        pairs: list[PromptPair] = []
        skipped = 0

        for run in runs:
            pair = self._parse_run(run)
            if pair is not None:
                pairs.append(pair)
            else:
                skipped += 1

        if skipped:
            # Structural log only — no run content to avoid PII exposure.
            logger.warning(
                "Some LangSmith runs could not be parsed and were skipped",
                extra={"skipped_count": skipped, "loaded_count": len(pairs)},
            )

        logger.info(
            "LangSmith load complete",
            extra={
                "project_name": self._project_name,
                "loaded_count": len(pairs),
                "skipped_count": skipped,
            },
        )

        return pairs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_client(self) -> Any:
        """Create and return a LangSmith client, raising clearly if langsmith is not installed."""
        try:
            import langsmith  # noqa: PLC0415  (lazy import — langsmith is optional)
        except ImportError as exc:
            raise ImportError(
                "The 'langsmith' package is required to use LangSmithAdapter. "
                "Install it with: pip install langsmith"
            ) from exc

        return langsmith.Client(api_key=self._api_key)

    def _parse_run(self, run: Any) -> PromptPair | None:
        """Parse a single LangSmith run into a PromptPair, or return None if unparseable.

        Extraction strategy:
        - **prompt**: ``inputs['messages']`` (list[dict]) → ``inputs['prompt']``
          (str) → ``inputs['input']`` (str)
        - **response**: ``outputs['output']`` → ``outputs['generations'][0][0]['text']``
          → ``outputs['response']``
        - **model**: ``extra['metadata']['ls_model_name']`` or fallback to
          ``self._source_model``
        """
        try:
            inputs: dict[str, Any] | None = run.inputs
            outputs: dict[str, Any] | None = run.outputs
            extra: dict[str, Any] = run.extra or {}

            # --- Extract prompt ------------------------------------------------
            if not inputs:
                return None

            prompt: str | list[dict[str, Any]] | None = None
            if "messages" in inputs:
                prompt = inputs["messages"]
            elif "prompt" in inputs:
                prompt = inputs["prompt"]
            elif "input" in inputs:
                prompt = inputs["input"]

            if prompt is None:
                return None

            # --- Extract response ----------------------------------------------
            if not outputs:
                return None

            raw_response: Any = None
            if "output" in outputs:
                raw_response = outputs["output"]
            elif "generations" in outputs:
                try:
                    raw_response = outputs["generations"][0][0]["text"]
                except (IndexError, KeyError, TypeError):
                    return None
            elif "response" in outputs:
                raw_response = outputs["response"]

            if raw_response is None:
                return None

            response: str = raw_response if isinstance(raw_response, str) else str(raw_response)

            # --- Extract model -------------------------------------------------
            metadata: dict[str, Any] = extra.get("metadata", {})
            source_model: str = metadata.get("ls_model_name") or self._source_model

            return PromptPair(
                prompt=prompt,
                response=response,
                source_model=source_model,
                metadata={"extra": extra} if extra else {},
            )

        except Exception:
            # Broad catch: any unexpected attribute access on a malformed run
            # object is treated as unparseable. No content is logged.
            return None
