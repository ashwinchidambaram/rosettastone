"""Composite evaluator: combines metrics by output type, computes pairwise win rate."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import litellm

from rosettastone.core.types import EvalResult, OutputType, PromptPair
from rosettastone.evaluate.exact_match import ExactMatchEvaluator
from rosettastone.evaluate.json_validator import JSONEvaluator
from rosettastone.evaluate.types import detect_output_type
from rosettastone.utils.logging import get_logger

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.context import PipelineContext

logger = get_logger("evaluate.composite")


# Default thresholds per output type (used when config.win_thresholds missing a key)
DEFAULT_WIN_THRESHOLDS: dict[str, float] = {
    "json": 0.95,
    "classification": 0.90,
    "short_text": 0.80,
    "long_text": 0.75,
}

# Routing table: which evaluators run for which output type
EVALUATOR_ROUTING: dict[OutputType, list[str]] = {
    OutputType.JSON: ["json_validator", "json_structural"],
    OutputType.CLASSIFICATION: ["exact_match"],
    OutputType.SHORT_TEXT: ["semantic"],
    OutputType.LONG_TEXT: ["semantic", "llm_judge"],
}

# Weights for composite score by metric category
METRIC_WEIGHTS: dict[str, float] = {
    # JSON metrics
    "json_valid": 0.0,  # gating — handled separately
    "json_field_match": 0.4,
    "json_structural_sim": 0.4,
    "json_schema_match": 0.2,
    # Classification
    "exact_match": 0.7,
    "string_similarity": 0.3,
    # Semantic / text
    "bertscore_f1": 0.5,
    "embedding_sim": 0.5,
    # LLM judge
    "llm_judge_score": 0.3,
}


class CompositeEvaluator:
    def __init__(
        self,
        config: MigrationConfig,
        on_progress: Callable[[int, int], None] | None = None,
        ctx: PipelineContext | None = None,
    ) -> None:
        self.config = config
        self.on_progress = on_progress
        self._ctx = ctx

    def evaluate(
        self,
        test_set: list[PromptPair],
        optimized_prompt: str | None = None,
    ) -> list[EvalResult]:
        # Phase 1: collect LLM completions for all pairs
        completions: list[tuple[PromptPair, str] | None] = []
        skipped_count = 0

        for i, pair in enumerate(test_set):
            # Build messages from prompt
            if isinstance(pair.prompt, str):
                messages: list[dict[str, str]] = [{"role": "user", "content": pair.prompt}]
            else:
                messages = pair.prompt  # type: ignore[assignment,unused-ignore]

            if optimized_prompt:
                messages = [{"role": "system", "content": optimized_prompt}] + messages

            try:
                extra_kwargs: dict[str, object] = (
                    dict(getattr(self.config, "lm_extra_kwargs", None) or {})
                )
                response = litellm.completion(
                    model=self.config.target_model,
                    messages=messages,
                    **extra_kwargs,
                )
                cost = getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
                if self._ctx is not None:
                    self._ctx.add_cost("evaluation", cost)
                if not response.choices:
                    logger.warning("Pair %d: empty choices in response, skipping", i)
                    skipped_count += 1
                    completions.append(None)
                    continue
                new_response = response.choices[0].message.content or ""
                completions.append((pair, new_response))
            except Exception as e:
                logger.warning(
                    "Pair %d: litellm.completion failed (%s), skipping",
                    i,
                    type(e).__name__,  # log only the exception type, not the message
                )
                skipped_count += 1
                completions.append(None)

        if skipped_count > 0:
            logger.warning(
                "Skipped %d/%d pairs during evaluation due to LLM errors",
                skipped_count,
                len(test_set),
            )
            if len(test_set) > 0 and skipped_count / len(test_set) > 0.20:
                logger.error(
                    "More than 20%% of pairs were skipped (%d/%d); results may be unreliable",
                    skipped_count,
                    len(test_set),
                )

        # Phase 2: batch-compute BERTScore for all free-text pairs (avoids per-pair model calls)
        bertscore_map: dict[int, float] = {}
        free_text_indices: list[int] = []
        free_text_pairs: list[tuple[str, str]] = []  # (actual, expected)

        for idx, entry in enumerate(completions):
            if entry is None:
                continue
            pair, new_response = entry
            output_type = pair.output_type or detect_output_type(pair.response)
            if output_type in (OutputType.SHORT_TEXT, OutputType.LONG_TEXT):
                free_text_indices.append(idx)
                free_text_pairs.append((new_response, pair.response))

        if free_text_pairs:
            try:
                from rosettastone.evaluate.bertscore import batch_compute_bertscore

                batch_scores = batch_compute_bertscore(free_text_pairs)
                for idx, score in zip(free_text_indices, batch_scores):
                    bertscore_map[idx] = score
            except ImportError:
                pass  # No BERTScore installed; _score_semantic will fall back

        # Phase 3: assemble EvalResult for each successful completion
        results: list[EvalResult] = []

        for idx, entry in enumerate(completions):
            if entry is None:
                continue
            pair, new_response = entry
            output_type = pair.output_type or detect_output_type(pair.response)

            scores = self._score(
                pair.response,
                new_response,
                output_type,
                prompt=pair.prompt,
                bertscore_f1=bertscore_map.get(idx),
            )
            composite = self._composite_score(scores, output_type)
            threshold = self._get_threshold(output_type)

            results.append(
                EvalResult(
                    prompt_pair=pair,
                    new_response=new_response,
                    scores=scores,
                    composite_score=composite,
                    is_win=composite >= threshold,
                    details={
                        "output_type": output_type.value,
                        "evaluators_used": list(scores.keys()),
                        "threshold": threshold,
                    },
                )
            )

            if self.on_progress:
                self.on_progress(idx + 1, len(test_set))

        return results

    def _get_threshold(self, output_type: OutputType) -> float:
        thresholds = getattr(self.config, "win_thresholds", DEFAULT_WIN_THRESHOLDS)
        return cast(
            float,
            thresholds.get(output_type.value, DEFAULT_WIN_THRESHOLDS.get(output_type.value, 0.8)),
        )

    def _score(
        self,
        expected: str,
        actual: str,
        output_type: OutputType,
        prompt: str | list[dict[str, Any]] | None = None,
        bertscore_f1: float | None = None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        prompt_str = prompt if isinstance(prompt, str) else None

        if output_type == OutputType.JSON:
            scores.update(JSONEvaluator(config=self.config).score(expected, actual))

            # JSON structural evaluator (Phase 2 addition — lazy import)
            try:
                from rosettastone.evaluate.json_structural import JSONStructuralEvaluator

                scores.update(JSONStructuralEvaluator(config=self.config).score(expected, actual))
            except ImportError:
                pass

        elif output_type == OutputType.CLASSIFICATION:
            scores.update(ExactMatchEvaluator(config=self.config).score(expected, actual))
        else:
            # Free text — use pre-computed BERTScore if available, else fall back
            scores.update(self._score_semantic(expected, actual, bertscore_f1=bertscore_f1))

        # LLM judge for long text (and optionally all types) — Phase 2
        if output_type == OutputType.LONG_TEXT and not getattr(self.config, "local_only", False):
            try:
                from rosettastone.evaluate.llm_judge import LLMJudgeEvaluator

                judge_scores = LLMJudgeEvaluator(config=self.config, ctx=self._ctx).score(
                    expected, actual, prompt=prompt_str
                )
                scores.update(judge_scores)
            except ImportError:
                pass

        return scores

    def _score_semantic(
        self,
        expected: str,
        actual: str,
        bertscore_f1: float | None = None,
    ) -> dict[str, float]:
        local_only = getattr(self.config, "local_only", False)

        # Use pre-computed batch BERTScore if provided
        if bertscore_f1 is not None:
            return {"bertscore_f1": bertscore_f1}

        try:
            from rosettastone.evaluate.bertscore import BERTScoreEvaluator

            return BERTScoreEvaluator(config=self.config).score(expected, actual)
        except ImportError:
            if local_only:
                logger.warning(
                    "BERTScore not available in local_only mode; falling back to string similarity"
                )
            try:
                from rosettastone.evaluate.embedding import EmbeddingEvaluator

                return EmbeddingEvaluator(config=self.config).score(expected, actual)
            except ImportError:
                return ExactMatchEvaluator(config=self.config).score(expected, actual)

    def _composite_score(self, scores: dict[str, float], output_type: OutputType) -> float:
        if not scores:
            return 0.0

        # Gated scoring: if JSON and json_valid == 0, composite is 0.0
        if output_type == OutputType.JSON and scores.get("json_valid", 1.0) == 0.0:
            return 0.0

        # Weighted composite
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, value in scores.items():
            # Skip gating metrics from weighted average
            if metric == "json_valid":
                continue
            weight = METRIC_WEIGHTS.get(metric, 1.0)
            weighted_sum += value * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        return weighted_sum / total_weight
