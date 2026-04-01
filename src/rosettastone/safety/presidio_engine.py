"""Presidio-based PII scanner — drop-in replacement for the regex pii_scanner."""

from __future__ import annotations

import logging
import threading
from typing import Any

from rosettastone.core.types import PromptPair
from rosettastone.safety.pii_scanner import PIIWarning

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity mapping from Presidio entity types → HIGH / MEDIUM / LOW
# ---------------------------------------------------------------------------

_PRESIDIO_SEVERITY_MAP: dict[str, str] = {
    "CREDIT_CARD": "HIGH",
    "US_SSN": "HIGH",
    "EMAIL_ADDRESS": "MEDIUM",
    "PHONE_NUMBER": "MEDIUM",
    "IP_ADDRESS": "LOW",
    "PERSON": "MEDIUM",
    "LOCATION": "LOW",
    "DATE_TIME": "LOW",
    "NRP": "LOW",
    "MEDICAL_LICENSE": "HIGH",
    "US_BANK_NUMBER": "HIGH",
    "US_DRIVER_LICENSE": "HIGH",
    "US_PASSPORT": "HIGH",
}

# Module-level singletons — populated lazily on first use
_analyzer_instance: Any = None
_anonymizer_instance: Any = None
_init_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lazy engine helpers (easily patchable in tests)
# ---------------------------------------------------------------------------


def _get_analyzer() -> Any:
    """Return a cached AnalyzerEngine, importing presidio lazily.

    Uses double-checked locking to ensure thread-safe singleton initialization.

    Raises:
        ImportError: When presidio_analyzer is not installed.
    """
    global _analyzer_instance  # noqa: PLW0603
    if _analyzer_instance is None:
        with _init_lock:
            if _analyzer_instance is None:  # double-checked locking
                try:
                    from presidio_analyzer import AnalyzerEngine
                except ImportError as exc:
                    raise ImportError(
                        "presidio_analyzer is required for Presidio-based PII scanning. "
                        "Install it with: pip install presidio-analyzer"
                    ) from exc

                _analyzer_instance = AnalyzerEngine()
                logger.debug("AnalyzerEngine initialised (singleton)")
    return _analyzer_instance


def _get_anonymizer() -> Any:
    """Return a cached AnonymizerEngine, importing presidio lazily.

    Uses double-checked locking to ensure thread-safe singleton initialization.

    Raises:
        ImportError: When presidio_anonymizer is not installed.
    """
    global _anonymizer_instance  # noqa: PLW0603
    if _anonymizer_instance is None:
        with _init_lock:
            if _anonymizer_instance is None:  # double-checked locking
                try:
                    from presidio_anonymizer import AnonymizerEngine
                except ImportError as exc:
                    raise ImportError(
                        "presidio_anonymizer is required for Presidio-based PII anonymization. "
                        "Install it with: pip install presidio-anonymizer"
                    ) from exc

                _anonymizer_instance = AnonymizerEngine()  # type: ignore[no-untyped-call]
                logger.debug("AnonymizerEngine initialised (singleton)")
    return _anonymizer_instance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_prompt_text(prompt: str | list[dict[str, Any]]) -> str:
    """Flatten a prompt (string or chat-message list) into a plain string."""
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for item in prompt:
        if isinstance(item, dict):
            if "content" in item:
                parts.append(str(item["content"]))
            if "text" in item:
                parts.append(str(item["text"]))
    return " ".join(parts)


def _severity_for(entity_type: str) -> str:
    """Map a Presidio entity type to HIGH / MEDIUM / LOW. Defaults to MEDIUM."""
    return _PRESIDIO_SEVERITY_MAP.get(entity_type, "MEDIUM")


# ---------------------------------------------------------------------------
# Public API — mirrors pii_scanner.py exactly
# ---------------------------------------------------------------------------


def scan_text_presidio(text: str) -> list[tuple[str, str]]:
    """Scan a single text string for PII using Presidio.

    Same return signature as ``pii_scanner.scan_text`` — drop-in replacement.

    Args:
        text: String to scan (prompt or response content). Content is never logged.

    Returns:
        List of ``(pii_type, severity)`` tuples, one per unique entity type found.
    """
    analyzer = _get_analyzer()
    results = analyzer.analyze(text=text, language="en")

    seen: dict[str, str] = {}
    for result in results:
        entity_type: str = result.entity_type
        if entity_type not in seen:
            seen[entity_type] = _severity_for(entity_type)

    logger.debug("scan_text_presidio: found %d entity type(s)", len(seen))
    return list(seen.items())


def scan_pairs_presidio(pairs: list[PromptPair]) -> list[PIIWarning]:
    """Scan a list of PromptPair objects for PII using Presidio.

    Same return type as ``pii_scanner.scan_pairs`` — compatible interface.

    Args:
        pairs: List of PromptPair objects to scan. Content is never logged.

    Returns:
        List of :class:`~rosettastone.safety.pii_scanner.PIIWarning` objects.
    """
    warnings: list[PIIWarning] = []

    for pair_idx, pair in enumerate(pairs):
        prompt_text = _extract_prompt_text(pair.prompt)

        # Accumulate counts per entity type across both prompt and response
        counts: dict[str, int] = {}

        for entity_type, _ in scan_text_presidio(prompt_text):
            counts[entity_type] = counts.get(entity_type, 0) + 1

        for entity_type, _ in scan_text_presidio(pair.response):
            counts[entity_type] = counts.get(entity_type, 0) + 1

        for entity_type, count in counts.items():
            warnings.append(
                PIIWarning(
                    pair_index=pair_idx,
                    pii_type=entity_type,
                    severity=_severity_for(entity_type),
                    occurrence_count=count,
                )
            )

    logger.debug("scan_pairs_presidio: %d warning(s) across %d pair(s)", len(warnings), len(pairs))
    return warnings


def anonymize_text(text: str) -> str:
    """Replace PII in *text* with entity-type placeholders.

    For example, an email becomes ``<EMAIL_ADDRESS>``.

    Args:
        text: Input string. Content is never logged.

    Returns:
        New string with PII replaced by ``<ENTITY_TYPE>`` tokens.
    """
    analyzer = _get_analyzer()
    anonymizer = _get_anonymizer()

    analyzer_results = analyzer.analyze(text=text, language="en")

    if not analyzer_results:
        return text

    # Build per-entity Replace operators so the placeholder is the entity type
    try:
        from presidio_anonymizer.entities import OperatorConfig

        operators: dict[str, Any] = {
            result.entity_type: OperatorConfig("replace", {"new_value": f"<{result.entity_type}>"})
            for result in analyzer_results
        }
    except ImportError:
        # Fallback: let anonymizer use its default operators
        operators = {}

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators if operators else None,
    )
    logger.debug("anonymize_text: anonymized %d entity/entities", len(analyzer_results))
    return str(anonymized.text)


def anonymize_pairs(pairs: list[PromptPair]) -> list[PromptPair]:
    """Return new PromptPair objects with PII anonymized in both prompt and response.

    Original pairs are **not** mutated. When ``pair.prompt`` is a list of dicts,
    the structure is preserved — only the ``content`` / ``text`` string values
    inside each message are anonymized.

    Args:
        pairs: List of PromptPair objects to anonymize.

    Returns:
        New list of PromptPair objects with PII replaced by placeholders.
    """
    result: list[PromptPair] = []
    for pair in pairs:
        if isinstance(pair.prompt, list):
            anonymized_messages = []
            for msg in pair.prompt:
                new_msg = dict(msg)
                if "content" in new_msg and isinstance(new_msg["content"], str):
                    new_msg["content"] = anonymize_text(new_msg["content"])
                if "text" in new_msg and isinstance(new_msg["text"], str):
                    new_msg["text"] = anonymize_text(new_msg["text"])
                anonymized_messages.append(new_msg)
            anon_prompt: Any = anonymized_messages
        else:
            anon_prompt = anonymize_text(str(pair.prompt))

        anon_response = anonymize_text(pair.response)

        new_pair = pair.model_copy(update={"prompt": anon_prompt, "response": anon_response})
        result.append(new_pair)

    logger.debug("anonymize_pairs: processed %d pair(s)", len(pairs))
    return result
