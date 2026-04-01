"""Tests for build_migration_metric() and IterationTracker in optimize/metric.py.

The metric must return dspy.Prediction(score, feedback) for every input, and
the feedback text must reflect which similarity threshold band was hit.

Patching strategy: metric.py imports compute_bertscore/compute_embedding_sim/
string_similarity with local `from ... import` statements inside the closure.
We patch at the source module level (e.g. rosettastone.evaluate.bertscore.compute_bertscore)
so the patch is visible when the closure executes its local import.
"""

from __future__ import annotations

import sys
import threading
from typing import Any
from unittest.mock import patch

import dspy
import pytest
from pydantic import ValidationError

from rosettastone.config import MigrationConfig
from rosettastone.core.types import PromptPair
from rosettastone.optimize.metric import IterationTracker, build_migration_metric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path) -> MigrationConfig:
    """Minimal valid MigrationConfig backed by a real (empty) file."""
    data_file = tmp_path / "data.jsonl"
    data_file.touch()
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=data_file,
    )


def _make_example(expected: str = "Paris") -> dspy.Example:
    return dspy.Example(prompt="test prompt", expected_response=expected).with_inputs("prompt")


def _make_prediction(response: str = "Paris") -> dspy.Prediction:
    return dspy.Prediction(response=response)


# ---------------------------------------------------------------------------
# Return-type tests
# ---------------------------------------------------------------------------


class TestMetricReturnType:
    """The metric must always return a dspy.Prediction with score and feedback."""

    def test_returns_dspy_prediction(self, tmp_path) -> None:
        """Metric must return dspy.Prediction — GEPA requires this exact type."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example()
        pred = _make_prediction()

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.9):
            result = metric(gold, pred)

        assert isinstance(result, dspy.Prediction), (
            f"Expected dspy.Prediction, got {type(result).__name__}"
        )

    def test_has_score_attribute(self, tmp_path) -> None:
        """dspy.Prediction must carry 'score' — GEPA uses it for optimization signal."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example()
        pred = _make_prediction()

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.9):
            result = metric(gold, pred)

        assert hasattr(result, "score"), "Prediction missing 'score' attribute"

    def test_has_feedback_attribute(self, tmp_path) -> None:
        """dspy.Prediction must carry 'feedback' — GEPA uses it for reflective optimization."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example()
        pred = _make_prediction()

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.9):
            result = metric(gold, pred)

        assert hasattr(result, "feedback"), "Prediction missing 'feedback' attribute"

    def test_score_is_float_in_unit_interval(self, tmp_path) -> None:
        """Score must be in [0.0, 1.0] — GEPA interprets it as a probability-like reward."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example()
        pred = _make_prediction()

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.9):
            result = metric(gold, pred)

        assert isinstance(result.score, float), (
            f"Expected score to be float, got {type(result.score).__name__}"
        )
        assert 0.0 <= result.score <= 1.0, f"Score {result.score} is outside [0.0, 1.0]"

    def test_feedback_is_non_empty_string(self, tmp_path) -> None:
        """Feedback must be a non-empty string — empty feedback gives GEPA no signal to reflect on."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example()
        pred = _make_prediction()

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.9):
            result = metric(gold, pred)

        assert isinstance(result.feedback, str), (
            f"Expected feedback to be str, got {type(result.feedback).__name__}"
        )
        assert len(result.feedback) > 0, "Feedback must not be empty"

    def test_score_capped_at_one(self, tmp_path) -> None:
        """If similarity returns > 1.0 (shouldn't happen but be safe), score is capped at 1.0."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example()
        pred = _make_prediction()

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=1.5):
            result = metric(gold, pred)

        assert result.score <= 1.0, f"Score {result.score} exceeds 1.0 — capping is broken"


# ---------------------------------------------------------------------------
# Threshold band tests
# ---------------------------------------------------------------------------


class TestMetricFeedbackBands:
    """Feedback text must differ across the three similarity threshold bands."""

    def test_low_score_feedback_contains_diverges(self, tmp_path) -> None:
        """Score < 0.7 → feedback must mention 'diverges' so GEPA knows to make big changes."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example("The capital of France is Paris.")
        pred = _make_prediction("I don't know")

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.5):
            result = metric(gold, pred)

        assert "diverges" in result.feedback, (
            f"Expected 'diverges' in feedback for score=0.5, got: {result.feedback!r}"
        )
        assert result.score == pytest.approx(0.5, abs=0.001)

    def test_mid_score_feedback_contains_partially(self, tmp_path) -> None:
        """0.7 <= score < 0.85 → feedback must mention 'partially' for moderate divergence."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example("Paris is the capital of France.")
        pred = _make_prediction("Paris is a city in France.")

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.75):
            result = metric(gold, pred)

        assert "partially" in result.feedback, (
            f"Expected 'partially' in feedback for score=0.75, got: {result.feedback!r}"
        )
        assert result.score == pytest.approx(0.75, abs=0.001)

    def test_high_score_feedback_contains_good_match(self, tmp_path) -> None:
        """score >= 0.85 → feedback must mention 'Good match' to signal success to GEPA."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example("Paris")
        pred = _make_prediction("Paris")

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.9):
            result = metric(gold, pred)

        assert "Good match" in result.feedback, (
            f"Expected 'Good match' in feedback for score=0.9, got: {result.feedback!r}"
        )
        assert result.score == pytest.approx(0.9, abs=0.001)

    def test_boundary_at_0_7_is_partial(self, tmp_path) -> None:
        """score = 0.7 is in the 'partially' band (not 'diverges') — verify boundary is inclusive."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example("Paris")
        pred = _make_prediction("Paris")

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.7):
            result = metric(gold, pred)

        # 0.7 is exactly on the boundary: `if sem_score < 0.7` is False, so "partially" branch
        assert "partially" in result.feedback, (
            f"Score 0.7 should hit 'partially' band, got: {result.feedback!r}"
        )

    def test_boundary_at_0_85_is_good_match(self, tmp_path) -> None:
        """score = 0.85 is in the 'Good match' band — verify upper boundary is inclusive."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example("Paris")
        pred = _make_prediction("Paris")

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.85):
            result = metric(gold, pred)

        # 0.85 is exactly: `elif sem_score < 0.85` is False, so "Good match" branch
        assert "Good match" in result.feedback, (
            f"Score 0.85 should hit 'Good match' band, got: {result.feedback!r}"
        )


# ---------------------------------------------------------------------------
# trace parameter
# ---------------------------------------------------------------------------


class TestMetricTraceParameter:
    """Metric must accept a trace parameter (DSPy passes it during compilation)."""

    def test_accepts_trace_parameter(self, tmp_path) -> None:
        """DSPy's evaluation harness passes trace=... — metric must not fail when it is set."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example()
        pred = _make_prediction()

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.9):
            # Should not raise TypeError about unexpected keyword argument
            result = metric(gold, pred, trace=object())

        assert hasattr(result, "score"), "Metric with trace= must still return valid Prediction"

    def test_accepts_none_trace(self, tmp_path) -> None:
        """trace=None is the default and must work without errors."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example()
        pred = _make_prediction()

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.9):
            result = metric(gold, pred, trace=None)

        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Import fallback chain
# ---------------------------------------------------------------------------


class TestMetricImportFallbacks:
    """Metric uses BERTScore -> embedding -> exact match fallback chain.

    Each path must produce valid Prediction output so the metric works regardless
    of which optional dependencies are installed.
    """

    def test_falls_back_to_embedding_when_bertscore_missing(self, tmp_path) -> None:
        """When bert_score is not installed, metric falls back to embedding similarity."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example("Paris")
        pred = _make_prediction("Paris")

        # Remove the bertscore module from sys.modules so the local import raises ImportError
        with patch.dict(sys.modules, {"bert_score": None}):
            with patch("rosettastone.evaluate.embedding.compute_embedding_sim", return_value=0.88):
                result = metric(gold, pred)

        assert 0.0 <= result.score <= 1.0, f"Embedding fallback score out of range: {result.score}"
        assert len(result.feedback) > 0

    def test_falls_back_to_string_similarity_when_both_missing(self, tmp_path) -> None:
        """When both bert_score and sentence_transformers missing, string_similarity is used."""
        config = _make_config(tmp_path)
        metric = build_migration_metric(config)
        gold = _make_example("Paris")
        pred = _make_prediction("Paris")

        with patch.dict(sys.modules, {"bert_score": None, "sentence_transformers": None}):
            with patch("rosettastone.evaluate.exact_match.string_similarity", return_value=0.6):
                result = metric(gold, pred)

        assert 0.0 <= result.score <= 1.0, (
            f"String similarity fallback score out of range: {result.score}"
        )
        assert len(result.feedback) > 0


# ---------------------------------------------------------------------------
# Known-issue weighting tests
# ---------------------------------------------------------------------------


class TestKnownIssueWeight:
    """known_issue_weight divides the score for pairs that are flagged as known issues."""

    def test_known_issue_weight_applied(self, tmp_path) -> None:
        """Score is halved (divided by 2.0) for a pair with non-None feedback."""
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=data_file,
            known_issue_weight=2.0,
        )
        known_prompt = "What is the capital of France?"
        train_set = [
            PromptPair(
                prompt=known_prompt,
                response="Paris",
                source_model="openai/gpt-4o",
                feedback="known regression",
            )
        ]
        metric = build_migration_metric(config, train_set=train_set)
        gold = dspy.Example(prompt=known_prompt, expected_response="Paris").with_inputs("prompt")
        pred = dspy.Prediction(response="Paris")

        sem_score = 0.8
        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=sem_score):
            result = metric(gold, pred)

        expected = sem_score / 2.0
        assert result.score == pytest.approx(expected, abs=0.001), (
            f"Expected score {expected}, got {result.score}"
        )

    def test_known_issue_weight_clamps_at_zero(self, tmp_path) -> None:
        """A very large weight still clamps score at 0.0 — score never goes negative."""
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=data_file,
            known_issue_weight=1000.0,
        )
        known_prompt = "What is 2 + 2?"
        train_set = [
            PromptPair(
                prompt=known_prompt,
                response="4",
                source_model="openai/gpt-4o",
                feedback="edge case",
            )
        ]
        metric = build_migration_metric(config, train_set=train_set)
        gold = dspy.Example(prompt=known_prompt, expected_response="4").with_inputs("prompt")
        pred = dspy.Prediction(response="4")

        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=0.9):
            result = metric(gold, pred)

        assert result.score >= 0.0, f"Score {result.score} went below 0.0"

    def test_known_issue_weight_no_effect_on_non_issue(self, tmp_path) -> None:
        """Pairs without feedback are unaffected — no divisor is applied."""
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=data_file,
            known_issue_weight=2.0,
        )
        # train_set has a known-issue pair for a *different* prompt
        train_set = [
            PromptPair(
                prompt="some other prompt",
                response="answer",
                source_model="openai/gpt-4o",
                feedback="known bug",
            )
        ]
        metric = build_migration_metric(config, train_set=train_set)
        # The gold prompt does NOT match any key in the feedback map
        gold = dspy.Example(prompt="unrelated prompt", expected_response="result").with_inputs(
            "prompt"
        )
        pred = dspy.Prediction(response="result")

        sem_score = 0.9
        with patch("rosettastone.evaluate.bertscore.compute_bertscore", return_value=sem_score):
            result = metric(gold, pred)

        assert result.score == pytest.approx(sem_score, abs=0.001), (
            f"Non-issue pair score should be {sem_score}, got {result.score}"
        )

    def test_known_issue_weight_config_field_validation(self) -> None:
        """known_issue_weight must be > 0; passing 0.0 must raise a ValidationError."""
        with pytest.raises(ValidationError):
            MigrationConfig(
                source_model="a",
                target_model="b",
                known_issue_weight=0.0,
            )


# ---------------------------------------------------------------------------
# IterationTracker tests
# ---------------------------------------------------------------------------


def _make_stub_metric(score: float = 0.8) -> Any:
    """Return a stub metric function that returns a dspy.Prediction with a fixed score."""

    def stub(example: Any, pred: Any, trace: Any = None) -> dspy.Prediction:
        return dspy.Prediction(score=score, feedback="stub")

    return stub


class TestIterationTrackerFiresCallback:
    """Callback fires exactly once after each full trainset sweep."""

    def test_fires_after_trainset_sweep(self) -> None:
        """Callback fires after exactly trainset_size calls — not before."""
        calls: list[tuple[int, int, float]] = []

        def callback(iteration: int, total: int, mean: float) -> None:
            calls.append((iteration, total, mean))

        tracker = IterationTracker(trainset_size=3, total_iterations=10, callback=callback)
        wrapped = tracker.wrap(_make_stub_metric(score=0.9))

        example = dspy.Example(prompt="q", expected_response="a").with_inputs("prompt")
        pred = dspy.Prediction(response="a")

        # Two calls — no callback yet
        wrapped(example, pred)
        wrapped(example, pred)
        assert calls == [], "Callback must not fire before a full sweep"

        # Third call — one full sweep
        wrapped(example, pred)
        assert len(calls) == 1, "Callback must fire exactly once after trainset_size calls"
        assert calls[0][0] == 1, "First callback must report iteration=1"
        assert calls[0][1] == 10, "Callback must pass total_iterations=10"

    def test_fires_once_per_sweep(self) -> None:
        """Callback fires once per full sweep — not once per call."""
        count = [0]

        def callback(iteration: int, total: int, mean: float) -> None:
            count[0] += 1

        tracker = IterationTracker(trainset_size=2, total_iterations=5, callback=callback)
        wrapped = tracker.wrap(_make_stub_metric())

        example = dspy.Example(prompt="q", expected_response="a").with_inputs("prompt")
        pred = dspy.Prediction(response="a")

        for _ in range(6):  # 3 full sweeps
            wrapped(example, pred)

        assert count[0] == 3, f"Expected 3 callbacks for 6 calls with sweep=2, got {count[0]}"

    def test_zero_trainset_size_is_noop(self) -> None:
        """When trainset_size=0, wrap returns the original function unchanged."""
        called = [False]

        def callback(iteration: int, total: int, mean: float) -> None:
            called[0] = True

        tracker = IterationTracker(trainset_size=0, total_iterations=5, callback=callback)
        original = _make_stub_metric()
        wrapped = tracker.wrap(original)

        # wrap must return the original function when trainset_size == 0
        assert wrapped is original, "trainset_size=0 must return original function unchanged"
        assert not called[0], "Callback must never fire when trainset_size=0"


class TestIterationTrackerMeanScore:
    """running_mean_score passed to the callback is correct."""

    def test_mean_score_is_correct(self) -> None:
        """running_mean_score is the cumulative mean of all scores seen so far."""
        received: list[float] = []

        def callback(iteration: int, total: int, mean: float) -> None:
            received.append(mean)

        trainset_size = 4
        tracker = IterationTracker(
            trainset_size=trainset_size, total_iterations=3, callback=callback
        )

        scores = [0.6, 0.8, 1.0, 0.4]  # mean = 0.7
        example = dspy.Example(prompt="q", expected_response="a").with_inputs("prompt")

        for s in scores:
            pred = dspy.Prediction(response="a")
            tracker.wrap(_make_stub_metric(score=s))(example, pred)

        assert len(received) == 1
        assert received[0] == pytest.approx(sum(scores) / len(scores), abs=0.001)

    def test_mean_score_cumulative_across_sweeps(self) -> None:
        """Mean grows cumulatively (all scores seen), not just the last sweep."""
        sweep1_scores = [0.5, 0.5]  # mean after sweep 1 = 0.5
        sweep2_scores = [1.0, 1.0]  # cumulative mean after sweep 2 = (0.5+0.5+1.0+1.0)/4 = 0.75

        received: list[float] = []

        def callback(iteration: int, total: int, mean: float) -> None:
            received.append(mean)

        tracker = IterationTracker(trainset_size=2, total_iterations=5, callback=callback)
        example = dspy.Example(prompt="q", expected_response="a").with_inputs("prompt")

        for s in sweep1_scores + sweep2_scores:
            tracker.wrap(_make_stub_metric(score=s))(example, dspy.Prediction(response="a"))

        assert len(received) == 2
        assert received[0] == pytest.approx(0.5, abs=0.001)
        assert received[1] == pytest.approx(0.75, abs=0.001)

    def test_score_extraction_fallback(self) -> None:
        """If the metric result has no .score attribute, float(result) is used."""
        received: list[float] = []

        def callback(iteration: int, total: int, mean: float) -> None:
            received.append(mean)

        tracker = IterationTracker(trainset_size=1, total_iterations=1, callback=callback)

        # Metric returns a plain float rather than dspy.Prediction
        def plain_metric(example: Any, pred: Any, trace: Any = None) -> float:
            return 0.77

        wrapped = tracker.wrap(plain_metric)
        example = dspy.Example(prompt="q", expected_response="a").with_inputs("prompt")
        wrapped(example, dspy.Prediction(response="a"))

        assert received == [pytest.approx(0.77, abs=0.001)]


class TestIterationTrackerThreadSafety:
    """IterationTracker must be correct under concurrent metric calls."""

    def test_concurrent_calls_count_correctly(self) -> None:
        """All concurrent metric calls are counted; no calls are lost or double-counted."""
        n_threads = 5
        calls_per_thread = 10  # 5 threads × 10 calls = 50 total = exactly one full sweep
        trainset_size = n_threads * calls_per_thread

        callback_count = [0]
        lock = threading.Lock()

        def callback(iteration: int, total: int, mean: float) -> None:
            with lock:
                callback_count[0] += 1

        tracker = IterationTracker(
            trainset_size=trainset_size,
            total_iterations=10,
            callback=callback,
        )
        wrapped = tracker.wrap(_make_stub_metric(score=0.5))
        example = dspy.Example(prompt="q", expected_response="a").with_inputs("prompt")
        pred = dspy.Prediction(response="a")

        def worker() -> None:
            for _ in range(calls_per_thread):
                wrapped(example, pred)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker._call_count == trainset_size, (
            f"Expected {trainset_size} total calls, got {tracker._call_count}"
        )
        assert callback_count[0] == 1, (
            f"Expected exactly 1 callback for one full sweep, got {callback_count[0]}"
        )

    def test_many_sweeps_concurrent(self) -> None:
        """Callback fires the correct number of times under heavy concurrent load."""
        trainset_size = 10
        n_sweeps = 5
        total_calls = trainset_size * n_sweeps
        n_threads = 10

        callback_count = [0]
        cblock = threading.Lock()

        def callback(iteration: int, total: int, mean: float) -> None:
            with cblock:
                callback_count[0] += 1

        tracker = IterationTracker(
            trainset_size=trainset_size,
            total_iterations=n_sweeps,
            callback=callback,
        )
        wrapped = tracker.wrap(_make_stub_metric(score=0.6))
        example = dspy.Example(prompt="q", expected_response="a").with_inputs("prompt")
        pred = dspy.Prediction(response="a")

        def worker(n: int) -> None:
            for _ in range(n):
                wrapped(example, pred)

        calls_per_thread = total_calls // n_threads
        threads = [threading.Thread(target=worker, args=(calls_per_thread,)) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker._call_count == total_calls, (
            f"Expected {total_calls} total calls, got {tracker._call_count}"
        )
        assert callback_count[0] == n_sweeps, (
            f"Expected {n_sweeps} callbacks for {n_sweeps} sweeps, got {callback_count[0]}"
        )
