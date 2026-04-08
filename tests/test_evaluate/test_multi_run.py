"""Tests for multi-run evaluation and variance tracking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rosettastone.config import MigrationConfig
from rosettastone.core.types import EvalResult, PromptPair


def _make_config(tmp_path: Path, **kwargs) -> MigrationConfig:
    data_file = tmp_path / "d.jsonl"
    data_file.write_text('{"prompt":"q","response":"a","source_model":"openai/gpt-4o"}\n')
    defaults = {
        "source_model": "openai/gpt-4o",
        "target_model": "anthropic/claude-sonnet-4",
        "data_path": data_file,
        "skip_preflight": True,
    }
    defaults.update(kwargs)
    return MigrationConfig(**defaults)


def _make_eval(score: float, pair: PromptPair | None = None) -> EvalResult:
    if pair is None:
        pair = PromptPair(prompt="q", response="a", source_model="openai/gpt-4o")
    return EvalResult(
        prompt_pair=pair,
        new_response="a",
        scores={"exact_match": score},
        composite_score=score,
        is_win=score >= 0.90,
        details={"output_type": "classification"},
    )


class TestSingleRun:
    def test_eval_runs_1_returns_same_as_evaluate(self, tmp_path):
        """eval_runs=1 uses evaluate() directly, no run_scores added."""
        from rosettastone.evaluate.composite import CompositeEvaluator

        config = _make_config(tmp_path, eval_runs=1)
        evaluator = CompositeEvaluator(config)

        pair = PromptPair(prompt="q", response="a", source_model="openai/gpt-4o")

        with patch.object(evaluator, "evaluate", return_value=[_make_eval(0.9)]) as mock_eval:
            results = evaluator.evaluate_multi_run([pair])
            mock_eval.assert_called_once_with(
                [pair], optimized_prompt=None, eval_pair_callback=None
            )

        assert len(results) == 1
        # Single run: run_scores should be empty (not populated)
        assert results[0].run_scores == []
        assert results[0].score_std == 0.0
        assert results[0].is_non_deterministic is False


class TestMultiRun:
    def test_eval_runs_3_calls_evaluate_3_times(self, tmp_path):
        """eval_runs=3 calls evaluate() 3 times."""
        from rosettastone.evaluate.composite import CompositeEvaluator

        config = _make_config(tmp_path, eval_runs=3)
        evaluator = CompositeEvaluator(config)

        pair = PromptPair(prompt="q", response="a", source_model="openai/gpt-4o")
        run_scores = [0.8, 0.9, 0.7]

        call_count = 0

        def fake_evaluate(test_set, optimized_prompt=None, eval_pair_callback=None):
            nonlocal call_count
            # Return result with the original pair object so id() alignment works
            result = _make_eval(run_scores[call_count], pair=pair)
            call_count += 1
            return [result]

        with patch.object(evaluator, "evaluate", side_effect=fake_evaluate):
            results = evaluator.evaluate_multi_run([pair])

        assert call_count == 3
        assert len(results) == 1
        result = results[0]
        assert sorted(result.run_scores) == sorted([0.8, 0.9, 0.7])
        # Median of [0.7, 0.8, 0.9] = 0.8
        assert abs(result.composite_score - 0.8) < 1e-9

    def test_aggregation_median(self, tmp_path):
        """eval_aggregation=median selects the middle value."""
        from rosettastone.evaluate.composite import CompositeEvaluator

        config = _make_config(tmp_path, eval_runs=3, eval_aggregation="median")
        evaluator = CompositeEvaluator(config)

        scores = [0.7, 0.9, 0.8]
        call_count = 0
        pair = PromptPair(prompt="q", response="a", source_model="openai/gpt-4o")

        def fake_eval(test_set, optimized_prompt=None, eval_pair_callback=None):
            nonlocal call_count
            r = _make_eval(scores[call_count], pair=pair)
            call_count += 1
            return [r]

        with patch.object(evaluator, "evaluate", side_effect=fake_eval):
            results = evaluator.evaluate_multi_run([pair])

        # Median of [0.7, 0.8, 0.9] = 0.8
        assert abs(results[0].composite_score - 0.8) < 1e-9

    def test_aggregation_mean(self, tmp_path):
        """eval_aggregation=mean computes arithmetic mean."""
        from rosettastone.evaluate.composite import CompositeEvaluator

        config = _make_config(tmp_path, eval_runs=3, eval_aggregation="mean")
        evaluator = CompositeEvaluator(config)

        scores = [0.6, 0.8, 1.0]
        call_count = 0
        pair = PromptPair(prompt="q", response="a", source_model="openai/gpt-4o")

        def fake_eval(test_set, optimized_prompt=None, eval_pair_callback=None):
            nonlocal call_count
            r = _make_eval(scores[call_count], pair=pair)
            call_count += 1
            return [r]

        with patch.object(evaluator, "evaluate", side_effect=fake_eval):
            results = evaluator.evaluate_multi_run([pair])

        # Mean = 0.8
        assert abs(results[0].composite_score - 0.8) < 1e-9

    def test_aggregation_p25(self, tmp_path):
        """eval_aggregation=p25 selects the 25th percentile score."""
        from rosettastone.evaluate.composite import CompositeEvaluator

        config = _make_config(tmp_path, eval_runs=4, eval_aggregation="p25")
        evaluator = CompositeEvaluator(config)

        scores = [0.9, 0.7, 0.6, 0.8]  # sorted: [0.6, 0.7, 0.8, 0.9]
        call_count = 0
        pair = PromptPair(prompt="q", response="a", source_model="openai/gpt-4o")

        def fake_eval(test_set, optimized_prompt=None, eval_pair_callback=None):
            nonlocal call_count
            r = _make_eval(scores[call_count], pair=pair)
            call_count += 1
            return [r]

        with patch.object(evaluator, "evaluate", side_effect=fake_eval):
            results = evaluator.evaluate_multi_run([pair])

        # p25 of [0.6, 0.7, 0.8, 0.9]: idx = int(4 * 0.25) = 1 -> 0.7
        assert abs(results[0].composite_score - 0.7) < 1e-9


class TestVarianceFlagging:
    def test_is_non_deterministic_when_std_exceeds_threshold(self, tmp_path):
        """score_std > variance_flag_threshold -> is_non_deterministic=True."""
        from rosettastone.evaluate.composite import CompositeEvaluator

        config = _make_config(tmp_path, eval_runs=2, variance_flag_threshold=0.05)
        evaluator = CompositeEvaluator(config)

        # stdev([0.0, 1.0]) = 0.707 >> 0.05
        scores = [0.0, 1.0]
        call_count = 0
        pair = PromptPair(prompt="q", response="a", source_model="openai/gpt-4o")

        def fake_eval(test_set, optimized_prompt=None, eval_pair_callback=None):
            nonlocal call_count
            r = _make_eval(scores[call_count], pair=pair)
            call_count += 1
            return [r]

        with patch.object(evaluator, "evaluate", side_effect=fake_eval):
            results = evaluator.evaluate_multi_run([pair])

        assert results[0].is_non_deterministic is True
        assert results[0].score_std > 0.05

    def test_not_non_deterministic_when_std_below_threshold(self, tmp_path):
        """score_std <= variance_flag_threshold -> is_non_deterministic=False."""
        from rosettastone.evaluate.composite import CompositeEvaluator

        config = _make_config(tmp_path, eval_runs=3, variance_flag_threshold=0.1)
        evaluator = CompositeEvaluator(config)

        # stdev([0.9, 0.91, 0.89]) approx 0.01 < 0.1
        scores = [0.9, 0.91, 0.89]
        call_count = 0
        pair = PromptPair(prompt="q", response="a", source_model="openai/gpt-4o")

        def fake_eval(test_set, optimized_prompt=None, eval_pair_callback=None):
            nonlocal call_count
            r = _make_eval(scores[call_count], pair=pair)
            call_count += 1
            return [r]

        with patch.object(evaluator, "evaluate", side_effect=fake_eval):
            results = evaluator.evaluate_multi_run([pair])

        assert results[0].is_non_deterministic is False


class TestRunAlignment:
    def test_mismatched_skips_align_by_pair_identity(self, tmp_path):
        """When runs skip different pairs, aggregate only pairs present in ALL runs."""
        from rosettastone.evaluate.composite import CompositeEvaluator

        config = _make_config(tmp_path, eval_runs=2, eval_aggregation="median")
        evaluator = CompositeEvaluator(config)

        pair_a = PromptPair(prompt="a", response="a", source_model="openai/gpt-4o")
        pair_b = PromptPair(prompt="b", response="b", source_model="openai/gpt-4o")
        pair_c = PromptPair(prompt="c", response="c", source_model="openai/gpt-4o")

        def _eval(pair, score):
            return EvalResult(
                prompt_pair=pair,
                new_response="x",
                scores={"s": score},
                composite_score=score,
                is_win=score >= 0.9,
                details={},
            )

        # Run 0: returns pair_a and pair_c (skips pair_b)
        # Run 1: returns pair_a and pair_b (skips pair_c)
        # Only pair_a appears in both → 1 aggregated result
        run_0 = [_eval(pair_a, 0.8), _eval(pair_c, 0.9)]
        run_1 = [_eval(pair_a, 0.6), _eval(pair_b, 0.7)]
        call_count = 0

        def fake_eval(test_set, optimized_prompt=None, eval_pair_callback=None):
            nonlocal call_count
            result = [run_0, run_1][call_count]
            call_count += 1
            return result

        with patch.object(evaluator, "evaluate", side_effect=fake_eval):
            results = evaluator.evaluate_multi_run([pair_a, pair_b, pair_c])

        assert len(results) == 1
        assert results[0].prompt_pair is pair_a
        # Median of [0.8, 0.6] = 0.7
        assert abs(results[0].composite_score - 0.7) < 1e-9


class TestNonDeterministicCountInResult:
    def test_non_deterministic_count_in_build_result(self, tmp_path):
        """build_result sets non_deterministic_count from validation results."""
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path, eval_runs=3)
        baseline = [_make_eval(0.9)]
        validation = [
            EvalResult(
                prompt_pair=PromptPair(prompt="q", response="a", source_model="openai/gpt-4o"),
                new_response="a",
                scores={"exact_match": 0.8},
                composite_score=0.8,
                is_win=False,
                details={},
                run_scores=[0.5, 0.8, 1.1],
                score_std=0.3,
                is_non_deterministic=True,
            ),
            _make_eval(0.9),  # deterministic
        ]
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.non_deterministic_count == 1
        assert result.eval_runs == 3

    def test_eval_runs_1_non_deterministic_count_is_zero(self, tmp_path):
        """eval_runs=1 never sets is_non_deterministic, count stays 0."""
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)  # default eval_runs=1
        baseline = [_make_eval(0.9)]
        validation = [_make_eval(0.8)]
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.non_deterministic_count == 0
        assert result.eval_runs == 1
