"""Composite evaluator: combines metrics by output type, computes pairwise win rate."""

from __future__ import annotations

import concurrent.futures
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
        eval_pair_callback: Callable[[int, int, float, str], None] | None = None,
    ) -> list[EvalResult]:
        # Phase 1: collect LLM completions for all pairs in parallel.
        # Each entry is either (PromptPair, response_str) on success
        # or (failure_reason_str, PromptPair) on failure (F6 taxonomy).
        completions: list[tuple[PromptPair, str] | tuple[str, PromptPair] | None] = (
            [None] * len(test_set)
        )
        skipped_count = 0
        total_eval_cost = 0.0

        # Build per-pair message lists before dispatching to threads
        _tasks: list[tuple[int, PromptPair, list[dict[str, str]]]] = []
        for _i, _pair in enumerate(test_set):
            if isinstance(_pair.prompt, str):
                _msgs: list[dict[str, str]] = [{"role": "user", "content": _pair.prompt}]
            else:
                _msgs = list(_pair.prompt)  # type: ignore[arg-type,unused-ignore]
            if optimized_prompt:
                _msgs = [{"role": "system", "content": optimized_prompt}] + _msgs
            _tasks.append((_i, _pair, _msgs))

        def _call_one(
            args: tuple[int, PromptPair, list[dict[str, str]]],
        ) -> tuple[int, tuple[PromptPair, str] | tuple[str, PromptPair], float, int, int]:
            idx, pair, msgs = args
            try:
                extra_kwargs: dict[str, Any] = dict(
                    getattr(self.config, "lm_extra_kwargs", None) or {}
                )
                response = litellm.completion(
                    model=self.config.target_model,
                    messages=msgs,
                    **extra_kwargs,
                )
                cost = getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
                try:
                    _usage = getattr(response, "usage", None)
                    _pt = int(getattr(_usage, "prompt_tokens", 0) or 0) if _usage else 0
                    _ct = int(getattr(_usage, "completion_tokens", 0) or 0) if _usage else 0
                except Exception:
                    _pt, _ct = 0, 0
                if not response.choices:
                    logger.warning("Pair %d: empty choices in response, skipping", idx)
                    return idx, ("no_response", pair), cost, _pt, _ct
                new_response = response.choices[0].message.content or ""
                if not new_response:
                    logger.warning("Pair %d: null/empty content in response, skipping", idx)
                    return idx, ("no_response", pair), cost, _pt, _ct
                return idx, (pair, new_response), cost, _pt, _ct
            except Exception as e:
                exc_type = type(e).__name__.lower()
                if "timeout" in exc_type:
                    failure_cat = "timeout"
                elif "ratelimit" in exc_type or "rate_limit" in exc_type or "quota" in exc_type:
                    failure_cat = "rate_limit"
                else:
                    failure_cat = "api_error"
                logger.warning(
                    "Pair %d: litellm.completion failed (%s), skipping",
                    idx,
                    type(e).__name__,
                )
                return idx, (failure_cat, pair), 0.0, 0, 0

        total_eval_prompt_tokens = 0
        total_eval_completion_tokens = 0
        try:
            _num_workers = int(self.config.num_threads)
        except (TypeError, ValueError, AttributeError):
            _num_workers = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=_num_workers) as executor:
            for _idx, _result, _cost, _pt, _ct in executor.map(_call_one, _tasks):
                completions[_idx] = _result
                total_eval_cost += _cost
                total_eval_prompt_tokens += _pt
                total_eval_completion_tokens += _ct
                if isinstance(_result[0], str):
                    skipped_count += 1

        if self._ctx is not None and total_eval_cost > 0:
            self._ctx.add_cost("evaluation", total_eval_cost)
        if self._ctx is not None and (
            total_eval_prompt_tokens > 0 or total_eval_completion_tokens > 0
        ):
            try:
                self._ctx.add_tokens(
                    "evaluation", total_eval_prompt_tokens, total_eval_completion_tokens
                )
            except Exception:
                pass

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
            # Skip failure entries: (failure_reason_str, PromptPair)
            if isinstance(entry[0], str):
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
            # Failure entry: (failure_reason: str, pair: PromptPair)
            if isinstance(entry[0], str):
                failure_reason, pair = entry
                output_type = pair.output_type or detect_output_type(pair.response)
                eval_result = EvalResult(
                    prompt_pair=pair,
                    new_response="",
                    scores={},
                    composite_score=0.0,
                    is_win=False,
                    details={
                        "output_type": output_type.value,
                        "evaluators_used": [],
                        "threshold": self._get_threshold(output_type),
                        "skipped": True,
                    },
                    failure_reason=failure_reason,
                )
                results.append(eval_result)
                if self.on_progress:
                    self.on_progress(idx + 1, len(test_set))
                if eval_pair_callback is not None:
                    try:
                        eval_pair_callback(
                            len(results) - 1,
                            len(test_set),
                            eval_result.composite_score,
                            eval_result.details.get("output_type") or "unknown",
                        )
                    except Exception:
                        pass  # Never let callback errors disrupt evaluation
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

            # F6: detect JSON gate failure (json_valid == 0 forces composite to 0)
            json_failure: str | None = None
            if output_type == OutputType.JSON and scores.get("json_valid", 1.0) == 0.0:
                json_failure = "json_gate_failed"

            eval_result = EvalResult(
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
                failure_reason=json_failure,
            )
            results.append(eval_result)

            if self.on_progress:
                self.on_progress(idx + 1, len(test_set))
            if eval_pair_callback is not None:
                try:
                    eval_pair_callback(
                        len(results) - 1,
                        len(test_set),
                        eval_result.composite_score,
                        eval_result.details.get("output_type") or "unknown",
                    )
                except Exception:
                    pass  # Never let callback errors disrupt evaluation

        return results

    def evaluate_multi_run(
        self,
        test_set: list[PromptPair],
        optimized_prompt: str | None = None,
        eval_pair_callback: Callable[[int, int, float, str], None] | None = None,
    ) -> list[EvalResult]:
        """Run evaluate() N times and aggregate results by prompt_pair identity.

        Preserves 3-phase BERTScore batching by calling the full evaluate() per run.
        Uses object identity (id()) to align results across runs, so pairs skipped in
        different runs do not corrupt the aggregation.

        The eval_pair_callback is only fired on the first run to avoid N×pairs events.
        """
        n_runs = self.config.eval_runs
        if n_runs <= 1:
            return self.evaluate(
                test_set, optimized_prompt=optimized_prompt,
                eval_pair_callback=eval_pair_callback,
            )

        run_results: list[list[EvalResult]] = []
        for run_idx in range(n_runs):
            # Only pass the callback on the first run to avoid N×pairs events
            _cb = eval_pair_callback if run_idx == 0 else None
            run_results.append(
                self.evaluate(test_set, optimized_prompt=optimized_prompt, eval_pair_callback=_cb)
            )

        # Build index: map prompt_pair object id → test_set position
        pair_to_idx: dict[int, int] = {id(pair): i for i, pair in enumerate(test_set)}

        # Per-run dicts: {test_set_idx: EvalResult}
        run_dicts: list[dict[int, EvalResult]] = []
        for run_result in run_results:
            rd: dict[int, EvalResult] = {}
            for r in run_result:
                idx = pair_to_idx.get(id(r.prompt_pair))
                if idx is not None:
                    rd[idx] = r
            run_dicts.append(rd)

        # Intersect: only aggregate pairs present in every run
        common_indices = set(run_dicts[0].keys())
        for rd in run_dicts[1:]:
            common_indices &= set(rd.keys())

        aggregated: list[EvalResult] = []
        for idx in sorted(common_indices):
            run_evals = [rd[idx] for rd in run_dicts]
            aggregated.append(self._aggregate_runs(run_evals))
        return aggregated

    def _aggregate_runs(self, run_evals: list[EvalResult]) -> EvalResult:
        """Aggregate multiple EvalResults for the same prompt into one."""
        import statistics

        strategy = self.config.eval_aggregation
        threshold = self.config.variance_flag_threshold

        scores_list = [r.composite_score for r in run_evals]

        if strategy == "mean":
            agg_score = statistics.mean(scores_list)
        elif strategy == "p25":
            sorted_scores = sorted(scores_list)
            idx = int(len(sorted_scores) * 0.25)
            agg_score = sorted_scores[idx]
        else:  # "median" (default)
            agg_score = statistics.median(scores_list)

        score_std = statistics.stdev(scores_list) if len(scores_list) > 1 else 0.0
        is_non_det = score_std > threshold

        # Use the first run's result as the base (preserves prompt_pair, scores breakdown, etc.)
        base = run_evals[0]

        return EvalResult(
            prompt_pair=base.prompt_pair,
            new_response=base.new_response,
            scores=base.scores,
            composite_score=agg_score,
            is_win=agg_score
            >= self._get_threshold(
                base.prompt_pair.output_type or detect_output_type(base.prompt_pair.response)
            ),
            details=base.details,
            run_scores=scores_list,
            score_std=score_std,
            is_non_deterministic=is_non_det,
        )

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
        import time as _time

        scores: dict[str, float] = {}
        prompt_str = prompt if isinstance(prompt, str) else None

        if output_type == OutputType.JSON:
            _t = _time.time()
            scores.update(JSONEvaluator(config=self.config).score(expected, actual))
            try:
                from rosettastone.server.metrics import record_evaluator_duration

                record_evaluator_duration("json_validator", output_type.value, _time.time() - _t)
            except Exception:
                pass

            try:
                from rosettastone.evaluate.json_structural import JSONStructuralEvaluator

                _t = _time.time()
                scores.update(JSONStructuralEvaluator(config=self.config).score(expected, actual))
                try:
                    from rosettastone.server.metrics import record_evaluator_duration

                    record_evaluator_duration(
                        "json_structural", output_type.value, _time.time() - _t
                    )
                except Exception:
                    pass
            except ImportError:
                pass

        elif output_type == OutputType.CLASSIFICATION:
            _t = _time.time()
            scores.update(ExactMatchEvaluator(config=self.config).score(expected, actual))
            try:
                from rosettastone.server.metrics import record_evaluator_duration

                record_evaluator_duration("exact_match", output_type.value, _time.time() - _t)
            except Exception:
                pass
        else:
            # Free text — use pre-computed BERTScore if available, else fall back
            _t = _time.time()
            scores.update(self._score_semantic(expected, actual, bertscore_f1=bertscore_f1))
            try:
                from rosettastone.server.metrics import record_evaluator_duration

                record_evaluator_duration("bertscore", output_type.value, _time.time() - _t)
            except Exception:
                pass

        # LLM judge for long text (and optionally all types) — Phase 2
        if output_type == OutputType.LONG_TEXT and not getattr(self.config, "local_only", False):
            try:
                from rosettastone.evaluate.llm_judge import LLMJudgeEvaluator

                _t = _time.time()
                judge_scores = LLMJudgeEvaluator(config=self.config, ctx=self._ctx).score(
                    expected, actual, prompt=prompt_str
                )
                scores.update(judge_scores)
                try:
                    from rosettastone.server.metrics import record_evaluator_duration

                    record_evaluator_duration(
                        "llm_judge", output_type.value, _time.time() - _t
                    )
                except Exception:
                    pass
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

            result = BERTScoreEvaluator(config=self.config).score(expected, actual)
            logger.debug("Semantic scoring: using BERTScore")
            return result
        except ImportError:
            if local_only:
                logger.warning(
                    "BERTScore not available in local_only mode; falling back to string similarity"
                )
            logger.info("BERTScore unavailable, falling back to EmbeddingEvaluator")
            try:
                from rosettastone.evaluate.embedding import EmbeddingEvaluator

                result = EmbeddingEvaluator(config=self.config).score(expected, actual)
                logger.debug("Semantic scoring: using EmbeddingEvaluator")
                return result
            except ImportError:
                logger.info("EmbeddingEvaluator unavailable, falling back to ExactMatchEvaluator")
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
