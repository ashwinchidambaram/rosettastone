"""Tests for pipeline step functions in rosettastone.core.pipeline.

All external subsystem calls (LLM, evaluate, optimize, safety) are mocked —
no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rosettastone.config import MigrationConfig
from rosettastone.core.context import PipelineContext, SafetySeverity, SafetyWarning
from rosettastone.core.pipeline import (
    PreflightReport,
    build_result,
    evaluate_baseline,
    make_recommendation,
    run_pii_scan,
    run_pii_scan_text,
)
from rosettastone.core.types import EvalResult, MigrationResult, OutputType, PromptPair

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DECISION_MODULE = "rosettastone.core.pipeline"


def _make_config(**kwargs) -> MigrationConfig:
    defaults = dict(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
    )
    defaults.update(kwargs)
    return MigrationConfig(**defaults)


def _make_pair(prompt: str = "What is 2+2?", response: str = "4") -> PromptPair:
    return PromptPair(
        prompt=prompt,
        response=response,
        source_model="openai/gpt-4o",
        output_type=OutputType.SHORT_TEXT,
    )


def _make_eval_result(is_win: bool = True, output_type: str = "short_text") -> EvalResult:
    pair = _make_pair()
    score = 0.9 if is_win else 0.3
    return EvalResult(
        prompt_pair=pair,
        new_response="4",
        scores={"composite": score},
        composite_score=score,
        is_win=is_win,
        details={"output_type": output_type},
    )


# ---------------------------------------------------------------------------
# PreflightReport
# ---------------------------------------------------------------------------


class TestPreflightReport:
    def test_has_blockers_false_when_no_blockers(self):
        report = PreflightReport(warnings=["minor warning"], blockers=[])
        assert report.has_blockers is False

    def test_has_blockers_true_when_blockers_present(self):
        report = PreflightReport(warnings=[], blockers=["context too large"])
        assert report.has_blockers is True

    def test_as_dry_run_result_returns_migration_result(self):
        report = PreflightReport(warnings=["w1"], blockers=[])
        config = _make_config()
        result = report.as_dry_run_result(config)
        assert isinstance(result, MigrationResult)
        assert any("DRY RUN" in w for w in result.warnings)

    def test_as_dry_run_result_includes_preflight_warnings(self):
        report = PreflightReport(warnings=["low sample count"], blockers=[])
        config = _make_config()
        result = report.as_dry_run_result(config)
        assert "low sample count" in result.warnings

    def test_as_dry_run_result_has_zero_scores(self):
        report = PreflightReport(warnings=[], blockers=[])
        config = _make_config()
        result = report.as_dry_run_result(config)
        assert result.confidence_score == 0.0
        assert result.baseline_score == 0.0
        assert result.improvement == 0.0


# ---------------------------------------------------------------------------
# run_pii_scan — respects config.pii_scan flag
# ---------------------------------------------------------------------------


class TestRunPiiScan:
    def test_run_pii_scan_respects_config_flag_when_disabled(self):
        """When config.pii_scan=False, run_pii_scan must return without scanning
        and must not add any safety warnings to the context."""
        config = _make_config(pii_scan=False)
        ctx = PipelineContext()
        pairs = [_make_pair(prompt="SSN: 123-45-6789")]

        # The early-return guard (`if not config.pii_scan: return`) fires before
        # any import of the scanner, so we simply verify no warnings are added.
        run_pii_scan(pairs, ctx, config)

        assert ctx.safety_warnings == []

    def test_run_pii_scan_calls_scanner_when_enabled(self):
        """When config.pii_scan=True, the underlying scanner should be invoked."""
        config = _make_config(pii_scan=True)
        ctx = PipelineContext()
        pairs = [_make_pair()]

        # Mock the scan_pairs function to return no warnings (clean data)
        with patch("rosettastone.safety.pii_scanner.scan_pairs", return_value=[]) as mock_scan:
            run_pii_scan(pairs, ctx, config)

        mock_scan.assert_called_once_with(pairs)

    def test_run_pii_scan_adds_warning_to_context(self):
        """A PII finding from scan_pairs must be converted to a SafetyWarning in ctx."""
        config = _make_config(pii_scan=True)
        ctx = PipelineContext()
        pairs = [_make_pair()]

        # Create a mock PIIWarning
        mock_pii_warning = MagicMock()
        mock_pii_warning.severity = "HIGH"
        mock_pii_warning.pii_type = "ssn"
        mock_pii_warning.count = 1
        mock_pii_warning.pair_index = 0

        with patch("rosettastone.safety.pii_scanner.scan_pairs", return_value=[mock_pii_warning]):
            run_pii_scan(pairs, ctx, config)

        assert len(ctx.safety_warnings) == 1
        assert ctx.safety_warnings[0].warning_type == "pii"
        assert ctx.safety_warnings[0].severity == SafetySeverity.HIGH

    def test_run_pii_scan_disabled_no_scanner_import_needed(self):
        """When pii_scan=False the function must return immediately with no side effects,
        without needing any safety module to be importable."""
        config = _make_config(pii_scan=False)
        ctx = PipelineContext()
        pairs = [_make_pair()]

        # No patches needed — the early-return guard should prevent any scanner call
        run_pii_scan(pairs, ctx, config)
        assert ctx.safety_warnings == []


# ---------------------------------------------------------------------------
# run_pii_scan_text — bug documentation
# ---------------------------------------------------------------------------


class TestRunPiiScanText:
    def test_run_pii_scan_text_respects_config_flag_when_disabled(self):
        """When config.pii_scan=False, run_pii_scan_text must return early without scanning."""
        config = _make_config(pii_scan=False)
        ctx = PipelineContext()
        clean_text = "Hello, world!"

        with patch("rosettastone.safety.pii_scanner.scan_text", return_value=[]) as mock_scan:
            run_pii_scan_text(clean_text, ctx, config)

        # The early-return guard fires before any scanning occurs.
        mock_scan.assert_not_called()

    def test_run_pii_scan_text_adds_high_severity_warning(self):
        """HIGH-severity PII found in optimized prompt text must be added to ctx."""
        config = _make_config(pii_scan=True)
        ctx = PipelineContext()
        text_with_ssn = "Respond using SSN 123-45-6789 as the identifier."

        with patch(
            "rosettastone.safety.pii_scanner.scan_text",
            return_value=[("ssn", "HIGH")],
        ):
            run_pii_scan_text(text_with_ssn, ctx, config)

        assert len(ctx.safety_warnings) == 1
        assert ctx.safety_warnings[0].severity == SafetySeverity.HIGH
        assert ctx.safety_warnings[0].warning_type == "pii_in_prompt"

    def test_run_pii_scan_text_skips_non_high_severity(self):
        """Only HIGH-severity findings should be added as safety warnings."""
        config = _make_config(pii_scan=True)
        ctx = PipelineContext()
        text = "Server at 192.168.1.1"

        with patch(
            "rosettastone.safety.pii_scanner.scan_text",
            return_value=[("ipv4", "LOW")],
        ):
            run_pii_scan_text(text, ctx, config)

        # LOW severity should be silently ignored (not added to ctx.safety_warnings)
        assert ctx.safety_warnings == []

    def test_run_pii_scan_text_with_no_config_still_scans(self):
        """run_pii_scan_text with config=None should still call the regex scanner."""
        ctx = PipelineContext()

        with patch(
            "rosettastone.safety.pii_scanner.scan_text",
            return_value=[],
        ) as mock_scan:
            run_pii_scan_text("clean text", ctx, config=None)

        mock_scan.assert_called_once()


# ---------------------------------------------------------------------------
# make_recommendation — delegates to decision engine
# ---------------------------------------------------------------------------


class TestMakeRecommendation:
    def test_make_recommendation_go(self):
        """High confidence validation results should produce a GO recommendation."""
        from rosettastone.decision.recommendation import MIN_RELIABLE_SAMPLES, Recommendation

        # Build enough high-scoring results to exceed thresholds
        validation = []
        for _ in range(MIN_RELIABLE_SAMPLES):
            validation.append(_make_eval_result(is_win=True, output_type="short_text"))

        config = _make_config()
        ctx = PipelineContext()  # no safety warnings

        rec, reasoning, per_type = make_recommendation(validation, ctx, config)

        assert rec == str(Recommendation.GO)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert isinstance(per_type, dict)

    def test_make_recommendation_no_go(self):
        """A HIGH-severity safety warning must produce a NO_GO recommendation."""
        from rosettastone.decision.recommendation import MIN_RELIABLE_SAMPLES, Recommendation

        validation = [
            _make_eval_result(is_win=True, output_type="json") for _ in range(MIN_RELIABLE_SAMPLES)
        ]

        config = _make_config()
        ctx = PipelineContext()
        ctx.safety_warnings.append(
            SafetyWarning(
                warning_type="pii_in_prompt",
                severity=SafetySeverity.HIGH,
                message="HIGH-severity PII (ssn) found in optimized prompt",
                details={"pii_type": "ssn"},
            )
        )

        rec, reasoning, per_type = make_recommendation(validation, ctx, config)

        assert rec == str(Recommendation.NO_GO)
        assert "ssn" in reasoning.lower() or "pii" in reasoning.lower()

    def test_make_recommendation_conditional(self):
        """Insufficient samples should produce a CONDITIONAL recommendation."""
        from rosettastone.decision.recommendation import MIN_RELIABLE_SAMPLES, Recommendation

        # Fewer than MIN_RELIABLE_SAMPLES results
        validation = [
            _make_eval_result(is_win=True, output_type="json")
            for _ in range(MIN_RELIABLE_SAMPLES - 1)
        ]

        config = _make_config()
        ctx = PipelineContext()

        rec, reasoning, per_type = make_recommendation(validation, ctx, config)

        assert rec == str(Recommendation.CONDITIONAL)
        assert isinstance(reasoning, str)

    def test_make_recommendation_returns_tuple_of_three(self):
        """make_recommendation must return a 3-tuple: (str, str, dict)."""
        config = _make_config()
        ctx = PipelineContext()

        result = make_recommendation([], ctx, config)

        assert isinstance(result, tuple)
        assert len(result) == 3
        rec, reasoning, per_type = result
        assert isinstance(rec, str)
        assert isinstance(reasoning, str)
        assert isinstance(per_type, dict)

    def test_make_recommendation_per_type_scores_populated(self):
        """per_type_scores dict should be populated when there are results."""
        from rosettastone.decision.recommendation import MIN_RELIABLE_SAMPLES

        validation = [
            _make_eval_result(is_win=True, output_type="short_text")
            for _ in range(MIN_RELIABLE_SAMPLES)
        ]
        config = _make_config()
        ctx = PipelineContext()

        _, _, per_type = make_recommendation(validation, ctx, config)

        assert "short_text" in per_type


# ---------------------------------------------------------------------------
# evaluate_baseline — delegates to CompositeEvaluator
# ---------------------------------------------------------------------------


class TestEvaluateBaseline:
    def test_evaluate_baseline_returns_scores(self):
        """evaluate_baseline must return a list of EvalResult objects."""
        config = _make_config()
        test_pairs = [_make_pair(), _make_pair()]
        eval_results = [_make_eval_result(), _make_eval_result()]

        with patch(
            "rosettastone.evaluate.composite.CompositeEvaluator.evaluate",
            return_value=eval_results,
        ):
            results = evaluate_baseline(test_pairs, config)

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, EvalResult)

    def test_evaluate_baseline_passes_test_pairs(self):
        """evaluate_baseline must pass the test pairs to CompositeEvaluator.evaluate."""
        config = _make_config()
        test_pairs = [_make_pair()]

        with patch(
            "rosettastone.evaluate.composite.CompositeEvaluator.evaluate",
            return_value=[_make_eval_result()],
        ) as mock_evaluate:
            evaluate_baseline(test_pairs, config)

        mock_evaluate.assert_called_once_with(test_pairs, optimized_prompt=None)

    def test_evaluate_baseline_empty_input_returns_empty(self):
        """An empty test set should return an empty list."""
        config = _make_config()

        with patch(
            "rosettastone.evaluate.composite.CompositeEvaluator.evaluate",
            return_value=[],
        ):
            results = evaluate_baseline([], config)

        assert results == []


# ---------------------------------------------------------------------------
# build_result — empty validation set gives zero confidence
# ---------------------------------------------------------------------------


class TestBuildResult:
    def test_empty_validation_set_gives_zero_confidence(self):
        """When validation list is empty, confidence_score must be 0.0."""
        config = _make_config()
        ctx = PipelineContext()

        result = build_result(
            config=config,
            optimized_prompt="some prompt",
            baseline=[],
            validation=[],
            duration=1.0,
            ctx=ctx,
        )

        assert result.confidence_score == 0.0

    def test_all_wins_gives_confidence_one(self):
        """All-winning validation results should produce confidence_score=1.0."""
        config = _make_config()
        ctx = PipelineContext()
        n = 5
        validation = [_make_eval_result(is_win=True) for _ in range(n)]
        baseline = [_make_eval_result(is_win=True) for _ in range(n)]

        result = build_result(
            config=config,
            optimized_prompt="prompt",
            baseline=baseline,
            validation=validation,
            duration=1.0,
            ctx=ctx,
        )

        assert result.confidence_score == 1.0

    def test_no_wins_gives_confidence_zero(self):
        """No-win results should produce confidence_score=0.0."""
        config = _make_config()
        ctx = PipelineContext()
        n = 5
        validation = [_make_eval_result(is_win=False) for _ in range(n)]
        baseline = [_make_eval_result(is_win=False) for _ in range(n)]

        result = build_result(
            config=config,
            optimized_prompt="prompt",
            baseline=baseline,
            validation=validation,
            duration=1.0,
            ctx=ctx,
        )

        assert result.confidence_score == 0.0

    def test_improvement_computed_correctly(self):
        """improvement must equal confidence_score minus baseline_score."""
        config = _make_config()
        ctx = PipelineContext()

        # 2 baseline wins out of 4 → baseline_score = 0.5
        baseline = [_make_eval_result(is_win=True)] * 2 + [_make_eval_result(is_win=False)] * 2
        # 3 validation wins out of 4 → confidence_score = 0.75
        validation = [_make_eval_result(is_win=True)] * 3 + [_make_eval_result(is_win=False)] * 1

        result = build_result(
            config=config,
            optimized_prompt="prompt",
            baseline=baseline,
            validation=validation,
            duration=1.0,
            ctx=ctx,
        )

        assert abs(result.improvement - (result.confidence_score - result.baseline_score)) < 1e-9

    def test_duration_passed_through(self):
        """The duration argument must be stored in result.duration_seconds."""
        config = _make_config()
        ctx = PipelineContext()

        result = build_result(
            config=config,
            optimized_prompt="",
            baseline=[],
            validation=[],
            duration=42.5,
            ctx=ctx,
        )

        assert result.duration_seconds == pytest.approx(42.5)

    def test_empty_validation_adds_warning_message(self):
        """An empty validation list should append a warning about skipped pairs."""
        config = _make_config()
        ctx = PipelineContext()

        result = build_result(
            config=config,
            optimized_prompt="",
            baseline=[],
            validation=[],
            duration=1.0,
            ctx=ctx,
        )

        assert any("validation" in w.lower() or "skipped" in w.lower() for w in result.warnings)

    def test_ctx_warnings_included_in_result(self):
        """Warnings accumulated in PipelineContext must appear in MigrationResult.warnings."""
        config = _make_config()
        ctx = PipelineContext()
        ctx.warnings.append("low sample count")
        validation = [_make_eval_result(is_win=True)]

        result = build_result(
            config=config,
            optimized_prompt="prompt",
            baseline=validation,
            validation=validation,
            duration=1.0,
            ctx=ctx,
        )

        assert "low sample count" in result.warnings

    def test_safety_warnings_serialized(self):
        """Safety warnings on ctx must appear in result.safety_warnings as dicts."""
        config = _make_config()
        ctx = PipelineContext()
        ctx.safety_warnings.append(
            SafetyWarning(
                warning_type="pii",
                severity=SafetySeverity.HIGH,
                message="SSN found",
            )
        )
        validation = [_make_eval_result(is_win=True)]

        result = build_result(
            config=config,
            optimized_prompt="prompt",
            baseline=validation,
            validation=validation,
            duration=1.0,
            ctx=ctx,
        )

        assert len(result.safety_warnings) == 1
        sw = result.safety_warnings[0]
        assert isinstance(sw, dict)
        assert sw["warning_type"] == "pii"
        assert sw["severity"] == "HIGH"

    def test_recommendation_from_ctx(self):
        """Recommendation stored in ctx must appear in the MigrationResult."""
        config = _make_config()
        ctx = PipelineContext()
        ctx.recommendation = ("GO", "All types passed.", {"json": {}})
        validation = [_make_eval_result(is_win=True)]

        result = build_result(
            config=config,
            optimized_prompt="prompt",
            baseline=validation,
            validation=validation,
            duration=1.0,
            ctx=ctx,
        )

        assert result.recommendation == "GO"
        assert result.recommendation_reasoning == "All types passed."

    def test_optimized_prompt_stored(self):
        """The optimized_prompt argument must be stored verbatim in the result."""
        config = _make_config()
        ctx = PipelineContext()
        prompt = "You are a precise assistant."

        result = build_result(
            config=config,
            optimized_prompt=prompt,
            baseline=[],
            validation=[],
            duration=0.1,
            ctx=ctx,
        )

        assert result.optimized_prompt == prompt
