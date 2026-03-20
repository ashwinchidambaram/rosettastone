"""LLM-as-judge evaluator using bidirectional pointwise Likert scoring."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from rosettastone.evaluate.base import Evaluator
from rosettastone.utils.logging import get_logger

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig

logger = get_logger("evaluate.llm_judge")

_RUBRIC = (
    "Rate the behavioral equivalence of the Response to the Expected output on a 1-5 scale: "
    "1=completely different, 2=same topic but wrong approach, "
    "3=similar intent different execution, "
    "4=mostly equivalent with minor differences, "
    "5=functionally equivalent"
)

_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing whether two LLM responses are behaviorally equivalent. "
    "Respond with only a single integer from 1 to 5."
)


def _build_messages(
    expected: str,
    actual: str,
    prompt: str | None,
    *,
    flip: bool = False,
) -> list[dict[str, str]]:
    """Build judge messages. When flip=True the order of Expected/Response is swapped."""
    if flip:
        label_a, text_a = "Response", actual
        label_b, text_b = "Expected", expected
    else:
        label_a, text_a = "Expected", expected
        label_b, text_b = "Response", actual

    user_content_parts = []
    if prompt:
        user_content_parts.append(f"Original prompt: {prompt}\n")
    user_content_parts.append(f"{label_a}: {text_a}\n{label_b}: {text_b}\n\n{_RUBRIC}")

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "".join(user_content_parts)},
    ]


def _parse_score(text: str) -> float | None:
    """Extract a 1-5 integer from judge response text."""
    text = text.strip()
    # First try: entire response is a digit
    if text in ("1", "2", "3", "4", "5"):
        return float(text)
    # Fallback: find first standalone digit 1-5
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return float(match.group(1))
    return None


def _normalize(raw: float) -> float:
    """Map Likert 1-5 → [0, 1]."""
    return (raw - 1.0) / 4.0


class LLMJudgeEvaluator(Evaluator):
    """Pointwise LLM judge evaluator.

    Calls the judge model twice (bidirectional) to reduce position bias and
    returns the averaged, normalised score in ``{"llm_judge_score": float}``.
    Returns ``{}`` on any failure so the composite excludes this metric rather
    than penalising with 0.
    """

    def __init__(self, config: MigrationConfig | None = None) -> None:
        super().__init__(config)

    @property
    def _judge_model(self) -> str:
        if self.config is not None and self.config.judge_model:
            return self.config.judge_model
        return "openai/gpt-4o"

    def score(self, expected: str, actual: str, **kwargs: Any) -> dict[str, float]:
        """Score behavioral equivalence via LLM judge.

        Accepts an optional ``prompt`` kwarg (the original user prompt) which
        is included in the judge context when provided.
        """
        import litellm

        prompt: str | None = kwargs.get("prompt")

        raw_scores: list[float] = []
        for flip in (False, True):
            messages = _build_messages(expected, actual, prompt, flip=flip)
            try:
                response = litellm.completion(
                    model=self._judge_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=16,
                )
                content = response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning(
                    "LLM judge call failed (%s: %s); returning empty scores",
                    type(exc).__name__,
                    exc,
                )
                return {}

            parsed = _parse_score(content)
            if parsed is None:
                logger.warning(
                    "LLM judge response could not be parsed (got %r); returning empty scores",
                    content,
                )
                return {}
            raw_scores.append(parsed)

        avg_raw = sum(raw_scores) / len(raw_scores)
        return {"llm_judge_score": _normalize(avg_raw)}
