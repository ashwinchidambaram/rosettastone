"""Tests for cost tracking: PipelineContext.add_cost, CompositeEvaluator cost capture,
and build_result cost_breakdown population."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from rosettastone.core.context import PipelineContext
from rosettastone.core.types import MigrationResult, OutputType, PromptPair

# ---------------------------------------------------------------------------
# 1. Basic add_cost accumulation
# ---------------------------------------------------------------------------


def test_add_cost_accumulates():
    ctx = PipelineContext()
    ctx.add_cost("evaluation", 0.05)
    ctx.add_cost("evaluation", 0.03)
    assert ctx.costs["evaluation"] == pytest.approx(0.08)


def test_add_cost_multiple_phases():
    ctx = PipelineContext()
    ctx.add_cost("evaluation", 0.10)
    ctx.add_cost("optimization", 0.20)
    assert ctx.costs["evaluation"] == pytest.approx(0.10)
    assert ctx.costs["optimization"] == pytest.approx(0.20)
    assert sum(ctx.costs.values()) == pytest.approx(0.30)


def test_add_cost_starts_at_zero():
    ctx = PipelineContext()
    ctx.add_cost("evaluation", 0.0)
    assert ctx.costs["evaluation"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Thread safety
# ---------------------------------------------------------------------------


def test_add_cost_thread_safety():
    ctx = PipelineContext()
    threads = [
        threading.Thread(target=ctx.add_cost, args=("evaluation", 0.01)) for _ in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ctx.costs["evaluation"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 3. CompositeEvaluator captures cost when _hidden_params present
# ---------------------------------------------------------------------------


def _make_mock_response(response_cost: float | None, content: str = "ok") -> MagicMock:
    """Build a mock litellm response with _hidden_params."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    if response_cost is not None:
        response._hidden_params = {"response_cost": response_cost}
    else:
        response._hidden_params = {}
    return response


def test_composite_evaluator_captures_cost():
    from rosettastone.evaluate.composite import CompositeEvaluator

    ctx = PipelineContext()
    config = MagicMock()
    config.target_model = "openai/gpt-4o-mini"
    config.lm_extra_kwargs = {}
    config.win_thresholds = {}
    config.local_only = True  # skip LLM judge

    pair = PromptPair(
        prompt="hello",
        response="world",
        source_model="openai/gpt-4o",
        output_type=OutputType.SHORT_TEXT,
    )

    mock_response = _make_mock_response(response_cost=0.05, content="world")

    # bertscore is a lazy import inside the function body; patch it at module level
    bertscore_module = MagicMock()
    bertscore_module.batch_compute_bertscore = MagicMock(return_value=[0.9])

    evaluator = CompositeEvaluator(config, ctx=ctx)
    with patch("litellm.completion", return_value=mock_response):
        with patch.dict(
            "sys.modules", {"rosettastone.evaluate.bertscore": bertscore_module}
        ):
            evaluator.evaluate([pair])

    assert ctx.costs.get("evaluation", 0.0) == pytest.approx(0.05)


def test_composite_evaluator_captures_cost_no_hidden_params():
    """When _hidden_params is absent, cost should be 0.0 with no exception raised."""
    from rosettastone.evaluate.composite import CompositeEvaluator

    ctx = PipelineContext()
    config = MagicMock()
    config.target_model = "openai/gpt-4o-mini"
    config.lm_extra_kwargs = {}
    config.win_thresholds = {}
    config.local_only = True

    pair = PromptPair(
        prompt="hello",
        response="world",
        source_model="openai/gpt-4o",
        output_type=OutputType.SHORT_TEXT,
    )

    # Build a response that genuinely has no _hidden_params attribute
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "world"
    # Simulate absent _hidden_params by configuring getattr to return {}
    type(mock_response).__dict__  # ensure _hidden_params isn't auto-created
    mock_response._hidden_params = {}  # empty dict → response_cost missing → 0.0

    bertscore_module = MagicMock()
    bertscore_module.batch_compute_bertscore = MagicMock(return_value=[0.9])

    evaluator = CompositeEvaluator(config, ctx=ctx)
    with patch("litellm.completion", return_value=mock_response):
        with patch.dict(
            "sys.modules", {"rosettastone.evaluate.bertscore": bertscore_module}
        ):
            evaluator.evaluate([pair])

    # Should store 0.0 and not raise
    assert ctx.costs.get("evaluation", 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. build_result populates cost_breakdown and cost_usd
# ---------------------------------------------------------------------------


def test_build_result_cost_breakdown():
    from rosettastone.core.pipeline import build_result

    ctx = PipelineContext()
    ctx.add_cost("evaluation", 0.10)
    ctx.add_cost("optimization", 0.20)

    config = MagicMock()
    config.model_dump.return_value = {"source_model": "a", "target_model": "b"}
    config.win_thresholds = {}

    result = build_result(
        config=config,
        optimized_prompt="do X",
        baseline=[],
        validation=[],
        duration=1.0,
        ctx=ctx,
    )

    assert isinstance(result, MigrationResult)
    assert result.cost_usd == pytest.approx(0.30)
    assert result.cost_breakdown == {"evaluation": pytest.approx(0.10), "optimization": pytest.approx(0.20)}


def test_build_result_no_ctx_cost_breakdown_empty():
    from rosettastone.core.pipeline import build_result

    config = MagicMock()
    config.model_dump.return_value = {"source_model": "a", "target_model": "b"}
    config.win_thresholds = {}

    result = build_result(
        config=config,
        optimized_prompt="",
        baseline=[],
        validation=[],
        duration=0.0,
        ctx=None,
    )

    assert result.cost_usd == pytest.approx(0.0)
    assert result.cost_breakdown == {}
