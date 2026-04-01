"""Tests for GEPA timeout with intermediate result fallback.

All dspy dependencies are mocked — no API calls, no model downloads.
Timeouts are simulated by patching future.result() to raise concurrent.futures.TimeoutError,
so tests run in milliseconds without needing to wait for actual wall-clock timeouts.
"""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

from rosettastone.config import MigrationConfig
from rosettastone.core.types import OutputType, PromptPair
from rosettastone.optimize.gepa import GEPAOptimizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, timeout: int = 600) -> MigrationConfig:
    data_file = tmp_path / "data.jsonl"
    data_file.touch()
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=data_file,
        gepa_timeout_seconds=timeout,
    )


def _make_pairs(n: int = 2) -> list[PromptPair]:
    return [
        PromptPair(
            prompt=f"prompt {i}",
            response=f"response {i}",
            source_model="openai/gpt-4o",
            output_type=OutputType.SHORT_TEXT,
        )
        for i in range(n)
    ]


def _make_compiled_mock(instructions: str = "Optimized instructions") -> MagicMock:
    mock_compiled = MagicMock()
    mock_compiled.predict.signature.instructions = instructions
    return mock_compiled


def _patch_future_timeout(executor_cls_mock):
    """Configure the mock executor so that future.result() raises TimeoutError."""
    mock_future = MagicMock()
    mock_future.result.side_effect = concurrent.futures.TimeoutError()
    mock_executor_instance = MagicMock()
    mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
    mock_executor_instance.__exit__ = MagicMock(return_value=False)
    mock_executor_instance.submit.return_value = mock_future
    executor_cls_mock.return_value = mock_executor_instance
    return mock_future


def _patch_future_success(executor_cls_mock, compiled_mock):
    """Configure the mock executor so that future.result() returns compiled_mock."""
    mock_future = MagicMock()
    mock_future.result.return_value = compiled_mock
    mock_executor_instance = MagicMock()
    mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
    mock_executor_instance.__exit__ = MagicMock(return_value=False)
    mock_executor_instance.submit.return_value = mock_future
    executor_cls_mock.return_value = mock_executor_instance
    return mock_future


# ---------------------------------------------------------------------------
# Test: timeout fires, no intermediate captured → TimeoutError raised
# ---------------------------------------------------------------------------


class TestTimeoutWithNoIntermediate:
    """When compile() times out and no intermediate result was captured, TimeoutError is raised."""

    def test_raises_timeout_error_when_no_intermediate(self, tmp_path) -> None:
        """future.result() raises TimeoutError with no intermediate → TimeoutError propagates."""
        config = _make_config(tmp_path, timeout=30)
        train = _make_pairs(2)
        val = _make_pairs(1)

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA"),
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
            patch("rosettastone.optimize.gepa.concurrent.futures.ThreadPoolExecutor") as mock_exec,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            _patch_future_timeout(mock_exec)

            with pytest.raises(TimeoutError, match=r"GEPA timed out after 30s"):
                GEPAOptimizer().optimize(train, val, config)

    def test_timeout_error_message_includes_seconds(self, tmp_path) -> None:
        """TimeoutError message must mention the configured timeout in seconds."""
        config = _make_config(tmp_path, timeout=60)
        train = _make_pairs(1)
        val = _make_pairs(1)

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA"),
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
            patch("rosettastone.optimize.gepa.concurrent.futures.ThreadPoolExecutor") as mock_exec,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            _patch_future_timeout(mock_exec)

            with pytest.raises(TimeoutError) as exc_info:
                GEPAOptimizer().optimize(train, val, config)

        assert "60" in str(exc_info.value), (
            f"Expected timeout value in error message, got: {exc_info.value!r}"
        )

    def test_raises_builtin_timeout_error(self, tmp_path) -> None:
        """Must raise builtin TimeoutError (catchable as TimeoutError)."""
        config = _make_config(tmp_path, timeout=30)
        train = _make_pairs(2)
        val = _make_pairs(1)

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA"),
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
            patch("rosettastone.optimize.gepa.concurrent.futures.ThreadPoolExecutor") as mock_exec,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            _patch_future_timeout(mock_exec)

            with pytest.raises(TimeoutError) as exc_info:
                GEPAOptimizer().optimize(train, val, config)

        # Must be catchable as the builtin TimeoutError
        assert isinstance(exc_info.value, TimeoutError)
        assert "no intermediate" in str(exc_info.value).lower() or "timed out" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test: timeout fires, intermediate was captured → intermediate returned
# ---------------------------------------------------------------------------


class TestTimeoutWithIntermediate:
    """When compile() times out but an intermediate result was captured, it is returned."""

    def _run_with_intermediates(
        self,
        tmp_path,
        intermediates: list[str],
        timeout: int = 30,
    ) -> str:
        """Helper: run optimize() with a pre-populated best_intermediate via patched extract.

        Uses a fake metric (replacing build_migration_metric) so that calling
        _iteration_capturing_metric with mock args doesn't invoke real BERTScore.
        The fake GEPA compile calls the metric N times, populate best_intermediate,
        then we simulate a timeout via future.result().
        """
        config = _make_config(tmp_path, timeout=timeout)
        train = _make_pairs(2)
        val = _make_pairs(1)

        extract_call_count = [0]
        captured_metric: list = []

        def fake_extract(prog):
            idx = extract_call_count[0]
            extract_call_count[0] += 1
            if idx < len(intermediates):
                return intermediates[idx]
            return intermediates[-1]

        def fake_base_metric(gold, pred, trace=None):
            # Simple stub — returns 0.5 score without real evaluation
            return 0.5

        def fake_gepa_init(**kwargs):
            captured_metric.append(kwargs.get("metric"))
            mock_gepa = MagicMock()

            def fake_compile(program, trainset):
                # Call the capturing metric N times to populate best_intermediate
                metric = captured_metric[0]
                for _ in range(len(intermediates)):
                    gold = MagicMock()
                    gold.expected_response = "resp"
                    pred = MagicMock()
                    metric(gold, pred)
                return MagicMock()  # value doesn't matter — we'll simulate timeout

            mock_gepa.compile.side_effect = fake_compile
            return mock_gepa

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA", side_effect=fake_gepa_init),
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
            patch("rosettastone.optimize.gepa.concurrent.futures.ThreadPoolExecutor") as mock_exec,
            patch("rosettastone.optimize.gepa.extract_optimized_instructions", side_effect=fake_extract),
            patch("rosettastone.optimize.gepa.build_migration_metric", return_value=fake_base_metric),
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            mock_executor_instance = MagicMock()
            mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = MagicMock(return_value=False)

            def real_submit(fn):
                # Run _run_gepa synchronously to trigger metric calls and populate best_intermediate
                fn()
                mock_future = MagicMock()
                mock_future.result.side_effect = concurrent.futures.TimeoutError()
                return mock_future

            mock_executor_instance.submit.side_effect = real_submit
            mock_exec.return_value = mock_executor_instance

            result = GEPAOptimizer().optimize(train, val, config)
        return result

    def test_returns_intermediate_when_one_captured(self, tmp_path) -> None:
        """When one intermediate was captured before timeout, it is returned."""
        result = self._run_with_intermediates(tmp_path, ["Single intermediate"])
        assert result == "Single intermediate", f"Got: {result!r}"

    def test_returns_most_recent_intermediate(self, tmp_path) -> None:
        """When multiple intermediates are captured, the last one is returned on timeout."""
        intermediates = ["First", "Second", "Third"]
        result = self._run_with_intermediates(tmp_path, intermediates)
        assert result == "Third", f"Expected last intermediate 'Third', got: {result!r}"

    def test_no_exception_raised_when_intermediate_available(self, tmp_path) -> None:
        """When intermediate is available, timeout must not raise an exception."""
        # Should complete without raising
        result = self._run_with_intermediates(tmp_path, ["Some intermediate result"])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test: Normal completion (timeout=600s) returns correct result, no side effects
# ---------------------------------------------------------------------------


class TestNormalCompletion:
    """Normal (non-timeout) completion must behave identically to the pre-timeout implementation."""

    def test_normal_completion_returns_extracted_instructions(self, tmp_path) -> None:
        """When compile() finishes within the timeout, returns extracted instructions."""
        config = _make_config(tmp_path, timeout=600)
        train = _make_pairs(3)
        val = _make_pairs(2)

        mock_compiled = _make_compiled_mock("Be concise and accurate.")

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_cls.return_value.compile.return_value = mock_compiled

            result = GEPAOptimizer().optimize(train, val, config)

        assert result == "Be concise and accurate.", f"Got: {result!r}"

    def test_normal_completion_does_not_raise(self, tmp_path) -> None:
        """Normal completion must never raise TimeoutError."""
        config = _make_config(tmp_path, timeout=600)
        train = _make_pairs(2)
        val = _make_pairs(1)

        mock_compiled = _make_compiled_mock()

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_cls.return_value.compile.return_value = mock_compiled

            # Must not raise
            result = GEPAOptimizer().optimize(train, val, config)

        assert isinstance(result, str)

    def test_default_timeout_is_600(self, tmp_path) -> None:
        """Default gepa_timeout_seconds must be 600 (no behavior change for existing configs)."""
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=data_file,
        )
        assert config.gepa_timeout_seconds == 600

    def test_future_result_called_with_timeout(self, tmp_path) -> None:
        """future.result() must be called with timeout=config.gepa_timeout_seconds."""
        config = _make_config(tmp_path, timeout=120)
        train = _make_pairs(2)
        val = _make_pairs(1)

        mock_compiled = _make_compiled_mock()

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA") as mock_gepa_cls,
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
            patch("rosettastone.optimize.gepa.concurrent.futures.ThreadPoolExecutor") as mock_exec,
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_gepa_cls.return_value.compile.return_value = mock_compiled
            mock_future = _patch_future_success(mock_exec, mock_compiled)

            GEPAOptimizer().optimize(train, val, config)

        # Verify future.result was called with the correct timeout
        mock_future.result.assert_called_once_with(timeout=120)


# ---------------------------------------------------------------------------
# Test: Immediate callback fires → intermediate returned on timeout
# ---------------------------------------------------------------------------


class TestTimeoutWithImmediateCallback:
    """Intermediate captured immediately before timeout fires — should return fast."""

    def test_callback_fires_before_timeout(self, tmp_path) -> None:
        """When extract succeeds on first metric call, intermediate is captured and returned."""
        config = _make_config(tmp_path, timeout=30)
        train = _make_pairs(2)
        val = _make_pairs(1)

        immediate_result = "Immediately captured instructions"
        captured_metric: list = []

        def fake_extract(prog):
            return immediate_result

        def fake_base_metric(gold, pred, trace=None):
            return 0.5

        def fake_gepa_init(**kwargs):
            captured_metric.append(kwargs.get("metric"))
            mock_gepa = MagicMock()

            def fake_compile(program, trainset):
                # Fire metric once immediately to populate best_intermediate
                metric = captured_metric[0]
                gold = MagicMock()
                gold.expected_response = "resp"
                pred = MagicMock()
                metric(gold, pred)
                return MagicMock()

            mock_gepa.compile.side_effect = fake_compile
            return mock_gepa

        with (
            patch("rosettastone.optimize.gepa.dspy.LM"),
            patch("rosettastone.optimize.gepa.dspy.GEPA", side_effect=fake_gepa_init),
            patch("rosettastone.optimize.gepa.dspy.context") as mock_ctx,
            patch("rosettastone.optimize.gepa.concurrent.futures.ThreadPoolExecutor") as mock_exec,
            patch("rosettastone.optimize.gepa.extract_optimized_instructions", side_effect=fake_extract),
            patch("rosettastone.optimize.gepa.build_migration_metric", return_value=fake_base_metric),
        ):
            mock_ctx.return_value.__enter__ = lambda s: s
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            mock_executor_instance = MagicMock()
            mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = MagicMock(return_value=False)

            def real_submit(fn):
                fn()  # Populate best_intermediate via one metric call
                mock_future = MagicMock()
                mock_future.result.side_effect = concurrent.futures.TimeoutError()
                return mock_future

            mock_executor_instance.submit.side_effect = real_submit
            mock_exec.return_value = mock_executor_instance

            result = GEPAOptimizer().optimize(train, val, config)

        assert result == immediate_result, f"Expected immediate intermediate, got: {result!r}"


# ---------------------------------------------------------------------------
# Test: Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """gepa_timeout_seconds field validation."""

    def test_timeout_minimum_is_30(self, tmp_path) -> None:
        """gepa_timeout_seconds must be at least 30 (ge=30 constraint)."""
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MigrationConfig(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                data_path=data_file,
                gepa_timeout_seconds=29,
            )

    def test_timeout_30_is_valid(self, tmp_path) -> None:
        """gepa_timeout_seconds=30 is the minimum valid value."""
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=data_file,
            gepa_timeout_seconds=30,
        )
        assert config.gepa_timeout_seconds == 30

    def test_timeout_29_raises_validation_error(self, tmp_path) -> None:
        """gepa_timeout_seconds=29 is below minimum and must raise ValidationError."""
        data_file = tmp_path / "data.jsonl"
        data_file.touch()
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match=r"greater_than_equal"):
            MigrationConfig(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                data_path=data_file,
                gepa_timeout_seconds=29,
            )
