"""Tests for mid-run cost guardrail enforcement via CostLimitExceeded."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rosettastone.core.types import CostLimitExceeded

# ---------------------------------------------------------------------------
# 1. CostLimitExceeded exception properties
# ---------------------------------------------------------------------------


def test_cost_limit_exceeded_message():
    exc = CostLimitExceeded(actual=1.2345, limit=1.0)
    assert "1.2345" in str(exc)
    assert "1.0000" in str(exc)
    assert exc.actual == pytest.approx(1.2345)
    assert exc.limit == pytest.approx(1.0)


def test_cost_limit_exceeded_is_exception():
    exc = CostLimitExceeded(actual=5.0, limit=2.0)
    assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# 2. Callback behaviour — replicate _make_gepa_cost_callback inline
#    (the function is nested inside Migrator.run(), so we test via integration)
# ---------------------------------------------------------------------------


def _build_gepa_cost_callback(
    gepa_cost_accumulator: list[float],
    max_cost_usd: float | None = None,
):
    """Local replica matching the implementation inside Migrator.run()."""

    def _gepa_cost_callback(
        kwargs: dict,
        completion_response: object,
        start_time: object,
        end_time: object,
    ) -> None:
        cost = kwargs.get("response_cost", 0.0) or 0.0
        gepa_cost_accumulator[0] += cost
        if max_cost_usd is not None and gepa_cost_accumulator[0] > max_cost_usd:
            raise CostLimitExceeded(gepa_cost_accumulator[0], max_cost_usd)

    return _gepa_cost_callback


def test_cost_limit_exceeded_raised_when_cap_exceeded():
    """Callback raises CostLimitExceeded when accumulated cost exceeds cap."""
    accumulator = [0.0]
    callback = _build_gepa_cost_callback(accumulator, max_cost_usd=1.0)

    # First call: accumulates to 0.5 — no exception
    callback({"response_cost": 0.5}, None, None, None)
    assert accumulator[0] == pytest.approx(0.5)

    # Second call: total 1.1 — should raise
    with pytest.raises(CostLimitExceeded) as exc_info:
        callback({"response_cost": 0.6}, None, None, None)
    assert exc_info.value.actual > 1.0
    assert exc_info.value.limit == pytest.approx(1.0)


def test_cost_limit_not_raised_when_no_cap():
    """No exception when max_cost_usd is None, even for large costs."""
    accumulator = [0.0]
    callback = _build_gepa_cost_callback(accumulator, max_cost_usd=None)

    for _ in range(10):
        callback({"response_cost": 100.0}, None, None, None)
    assert accumulator[0] == pytest.approx(1000.0)


def test_cost_limit_not_raised_exactly_at_cap():
    """Callback does NOT raise when accumulated cost equals the cap exactly."""
    accumulator = [0.0]
    callback = _build_gepa_cost_callback(accumulator, max_cost_usd=1.0)

    # Exactly at cap — should NOT raise (strictly greater than triggers)
    callback({"response_cost": 1.0}, None, None, None)
    assert accumulator[0] == pytest.approx(1.0)


def test_cost_limit_missing_response_cost_key():
    """Missing response_cost key defaults to 0.0 and does not raise."""
    accumulator = [0.0]
    callback = _build_gepa_cost_callback(accumulator, max_cost_usd=1.0)

    callback({}, None, None, None)
    assert accumulator[0] == pytest.approx(0.0)


def test_cost_limit_none_response_cost_treated_as_zero():
    """None response_cost is coerced to 0.0 via `or 0.0`."""
    accumulator = [0.0]
    callback = _build_gepa_cost_callback(accumulator, max_cost_usd=1.0)

    callback({"response_cost": None}, None, None, None)
    assert accumulator[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Integration: Migrator raises CostLimitExceeded when GEPA exceeds cap
# ---------------------------------------------------------------------------


def _make_eval_result(is_win: bool = True):
    from rosettastone.core.types import EvalResult, OutputType, PromptPair

    pair = PromptPair(
        prompt="q",
        response="a",
        source_model="openai/gpt-4o",
        output_type=OutputType.SHORT_TEXT,
    )
    return EvalResult(
        prompt_pair=pair,
        new_response="a",
        scores={"composite": 0.9 if is_win else 0.3},
        composite_score=0.9 if is_win else 0.3,
        is_win=is_win,
    )


def test_migrator_raises_cost_limit_exceeded_during_optimize():
    """Migrator.run() propagates CostLimitExceeded when GEPA callback fires."""
    from rosettastone.config import MigrationConfig
    from rosettastone.core.migrator import Migrator
    from rosettastone.core.types import MigrationResult

    config = MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        max_cost_usd=0.50,
        skip_preflight=True,
    )

    eval_result = _make_eval_result(is_win=True)

    def _fake_optimize(train, val, cfg, gepa_cb=None):
        # Simulate a litellm success_callback firing mid-optimize by
        # invoking whatever callback was most recently appended.
        import litellm

        cb = litellm.success_callback[-1]
        # Push cost over the 0.50 cap
        cb({"response_cost": 0.60}, None, None, None)
        return "optimized prompt"

    # Pipeline functions are imported inside Migrator.run() from rosettastone.core.pipeline,
    # so they must be patched at the source module, matching the pattern in test_migrator.py.
    _pipeline = "rosettastone.core.pipeline"
    with (
        patch(f"{_pipeline}.load_and_split_data", return_value=([eval_result.prompt_pair], [], [])),
        patch(f"{_pipeline}.run_pii_scan"),
        patch(f"{_pipeline}.run_pii_scan_text"),
        patch(f"{_pipeline}.run_prompt_audit"),
        patch(f"{_pipeline}.evaluate_baseline", return_value=[eval_result]),
        patch(f"{_pipeline}.optimize_prompt", side_effect=_fake_optimize),
        patch(f"{_pipeline}.evaluate_optimized", return_value=[eval_result]),
        patch(f"{_pipeline}.make_recommendation", return_value=("GO", "ok", {})),
        patch(
            f"{_pipeline}.build_result",
            return_value=MigrationResult(
                config={},
                optimized_prompt="optimized prompt",
                baseline_results=[eval_result],
                validation_results=[eval_result],
                confidence_score=1.0,
                baseline_score=1.0,
                improvement=0.0,
                cost_usd=0.0,
                duration_seconds=0.5,
                warnings=[],
            ),
        ),
        patch(f"{_pipeline}.generate_report"),
    ):
        with pytest.raises(CostLimitExceeded) as exc_info:
            Migrator(config).run()

    assert exc_info.value.actual > 0.50
    assert exc_info.value.limit == pytest.approx(0.50)
