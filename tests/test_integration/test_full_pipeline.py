"""Full pipeline integration test — Migrator.run() end-to-end with mocked LLM.

Exercises the complete migration pipeline: preflight → ingest → PII scan →
baseline eval → optimize → PII scan text → prompt audit → validation eval →
recommendation → report. All external calls (LLM, file I/O) are mocked.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rosettastone.config import MigrationConfig
from rosettastone.core.migrator import Migrator
from rosettastone.core.types import EvalResult, MigrationResult, PromptPair

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIPELINE = "rosettastone.core.pipeline"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_file(tmp_path):
    """Create a minimal JSONL data file with 5 prompt/response pairs."""
    p = tmp_path / "test_data.jsonl"
    lines = [
        json.dumps(
            {
                "prompt": f"What is {i}+{i}?",
                "response": f"{i + i}",
                "source_model": "openai/gpt-4o",
            }
        )
        for i in range(5)
    ]
    p.write_text("\n".join(lines))
    return p


@pytest.fixture
def config(data_file, tmp_path):
    """Minimal MigrationConfig for integration testing."""
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=data_file,
        output_dir=tmp_path / "outputs",
        skip_preflight=True,  # Skip real LLM availability checks
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pairs(n: int = 5) -> list[PromptPair]:
    return [
        PromptPair(
            prompt=f"Q{i}",
            response=f"A{i}",
            source_model="openai/gpt-4o",
        )
        for i in range(n)
    ]


def _make_eval_results(
    pairs: list[PromptPair],
    score: float = 0.9,
    is_win: bool = True,
    response_prefix: str = "resp",
) -> list[EvalResult]:
    return [
        EvalResult(
            prompt_pair=p,
            new_response=f"{response_prefix}-{p.response}",
            scores={"bertscore_f1": score},
            composite_score=score,
            is_win=is_win,
            details={},
        )
        for p in pairs
    ]


def _make_preflight_report(
    warnings: list[str] | None = None,
    has_blockers: bool = False,
    estimated_cost_usd: float = 0.0,
    dry_run_result: MigrationResult | None = None,
) -> MagicMock:
    report = MagicMock()
    report.warnings = warnings or []
    report.has_blockers = has_blockers
    report.estimated_cost_usd = estimated_cost_usd
    if dry_run_result is not None:
        report.as_dry_run_result.return_value = dry_run_result
    return report


# ---------------------------------------------------------------------------
# Core end-to-end tests
# ---------------------------------------------------------------------------


class TestFullPipelineEndToEnd:
    def test_full_pipeline_produces_valid_result(self, config, tmp_path):
        """Complete migration pipeline with all stages mocked returns a valid MigrationResult."""
        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        baseline_results = _make_eval_results(test_set, score=0.8, response_prefix="base")
        validation_results = _make_eval_results(test_set, score=0.9, response_prefix="opt")
        optimized_prompt = "Optimized: be more precise"

        with (
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=baseline_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value=optimized_prompt),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=validation_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("GO", "All types pass.", {"json": {"win_rate": 0.9}}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            migrator = Migrator(config)
            result = migrator.run()

        assert isinstance(result, MigrationResult)
        assert result.optimized_prompt == optimized_prompt
        assert result.recommendation == "GO"
        assert len(result.baseline_results) == len(test_set)
        assert len(result.validation_results) == len(test_set)
        assert result.confidence_score >= 0.0
        assert result.baseline_score >= 0.0
        assert result.duration_seconds >= 0.0

    def test_result_fields_are_fully_populated(self, config):
        """MigrationResult must have all expected top-level fields set to correct types."""
        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        baseline_results = _make_eval_results(test_set, score=0.85)
        validation_results = _make_eval_results(test_set, score=0.92)

        with (
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=baseline_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="Optimized prompt"),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=validation_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("GO", "Looks great.", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            result = Migrator(config).run()

        assert isinstance(result.config, dict)
        assert isinstance(result.optimized_prompt, str)
        assert isinstance(result.baseline_results, list)
        assert isinstance(result.validation_results, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.safety_warnings, list)
        assert isinstance(result.prompt_regressions, list)
        assert result.improvement == pytest.approx(result.confidence_score - result.baseline_score)


# ---------------------------------------------------------------------------
# Recommendation variations
# ---------------------------------------------------------------------------


class TestRecommendationVariations:
    def test_pipeline_with_no_go_recommendation(self, config):
        """When recommendation is NO_GO the result reflects it."""
        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        baseline_results = _make_eval_results(test_set, score=0.4, is_win=False)
        validation_results = _make_eval_results(test_set, score=0.3, is_win=False)

        with (
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=baseline_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="worse prompt"),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=validation_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("NO_GO", "Win rate too low.", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            result = Migrator(config).run()

        assert result.recommendation == "NO_GO"

    def test_pipeline_with_conditional_recommendation(self, config):
        """CONDITIONAL recommendation is propagated correctly."""
        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        results = _make_eval_results(test_set, score=0.78, is_win=True)

        with (
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="marginally better"),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("CONDITIONAL", "Some types marginal.", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            result = Migrator(config).run()

        assert result.recommendation == "CONDITIONAL"


# ---------------------------------------------------------------------------
# Safety / PII warnings
# ---------------------------------------------------------------------------


class TestSafetyWarnings:
    def test_pipeline_with_pii_warnings_from_scan(self, config):
        """PII warnings injected by run_pii_scan appear in result.safety_warnings."""
        from rosettastone.core.context import SafetySeverity, SafetyWarning

        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        eval_results = _make_eval_results(test_set)

        def _add_pii_warning(pairs_arg, ctx, config_arg):
            ctx.safety_warnings.append(
                SafetyWarning(
                    warning_type="pii",
                    severity=SafetySeverity.MEDIUM,
                    message="PII detected: EMAIL (1 occurrences in 1 pair(s))",
                )
            )

        with (
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=eval_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="safe prompt"),
            patch(f"{_PIPELINE}.run_pii_scan", side_effect=_add_pii_warning),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=eval_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("GO", "All good.", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            result = Migrator(config).run()

        assert len(result.safety_warnings) >= 1
        types = {w["warning_type"] for w in result.safety_warnings}
        assert "pii" in types

    def test_pipeline_safety_warnings_are_serializable(self, config):
        """Safety warnings in result must be plain dicts (JSON-serializable)."""
        from rosettastone.core.context import SafetySeverity, SafetyWarning

        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        eval_results = _make_eval_results(test_set)

        def _inject_warning(pairs_arg, ctx, config_arg):
            ctx.safety_warnings.append(
                SafetyWarning(
                    warning_type="pii",
                    severity=SafetySeverity.HIGH,
                    message="HIGH PII in training data",
                )
            )

        with (
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=eval_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="prompt"),
            patch(f"{_PIPELINE}.run_pii_scan", side_effect=_inject_warning),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=eval_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("NO_GO", "PII blocker.", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            result = Migrator(config).run()

        # All safety_warnings entries must be plain dicts with known keys
        for w in result.safety_warnings:
            assert isinstance(w, dict)
            assert "warning_type" in w
            assert "severity" in w
            assert "message" in w
            # Must be JSON-serializable
            json.dumps(w)


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------


class TestStageOrdering:
    def test_pipeline_calls_stages_in_order(self, config):
        """Pipeline stages execute in the correct sequence."""
        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        eval_results = _make_eval_results(test_set)

        call_order: list[str] = []

        def _record(name: str):
            def _side_effect(*args: Any, **kwargs: Any) -> None:
                call_order.append(name)

            return _side_effect

        def _record_return(name: str, return_value: Any):
            def _side_effect(*args: Any, **kwargs: Any) -> Any:
                call_order.append(name)
                return return_value

            return _side_effect

        with (
            patch(
                f"{_PIPELINE}.load_and_split_data",
                side_effect=_record_return("load_and_split_data", (train, val, test_set)),
            ),
            patch(f"{_PIPELINE}.run_pii_scan", side_effect=_record("run_pii_scan")),
            patch(
                f"{_PIPELINE}.evaluate_baseline",
                side_effect=_record_return("evaluate_baseline", eval_results),
            ),
            patch(
                f"{_PIPELINE}.optimize_prompt",
                side_effect=_record_return("optimize_prompt", "optimized"),
            ),
            patch(
                f"{_PIPELINE}.run_pii_scan_text",
                side_effect=_record("run_pii_scan_text"),
            ),
            patch(
                f"{_PIPELINE}.run_prompt_audit",
                side_effect=_record("run_prompt_audit"),
            ),
            patch(
                f"{_PIPELINE}.evaluate_optimized",
                side_effect=_record_return("evaluate_optimized", eval_results),
            ),
            patch(
                f"{_PIPELINE}.make_recommendation",
                side_effect=_record_return("make_recommendation", ("GO", "all good", {})),
            ),
            patch(
                f"{_PIPELINE}.generate_report",
                side_effect=_record("generate_report"),
            ),
        ):
            Migrator(config).run()

        # Verify key ordering constraints
        assert call_order.index("load_and_split_data") < call_order.index("run_pii_scan")
        assert call_order.index("run_pii_scan") < call_order.index("evaluate_baseline")
        assert call_order.index("evaluate_baseline") < call_order.index("optimize_prompt")
        assert call_order.index("optimize_prompt") < call_order.index("run_pii_scan_text")
        assert call_order.index("run_pii_scan_text") < call_order.index("run_prompt_audit")
        assert call_order.index("run_prompt_audit") < call_order.index("evaluate_optimized")
        assert call_order.index("evaluate_optimized") < call_order.index("make_recommendation")
        assert call_order.index("make_recommendation") < call_order.index("generate_report")


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_pipeline_with_progress_callback(self, config):
        """Progress callback is invoked with stage names during pipeline execution."""
        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        eval_results = _make_eval_results(test_set)

        received_stages: list[str] = []

        def _progress(stage: str, stage_pct: float, overall_pct: float) -> None:
            received_stages.append(stage)

        with (
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=eval_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="optimized"),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=eval_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("GO", "all good", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            migrator = Migrator(config, progress_callback=_progress)
            migrator.run()

        assert len(received_stages) > 0
        # Key stage names emitted by migrator._emit()
        expected_stages = {"data_load", "baseline_eval", "optimize", "validation_eval", "report"}
        assert expected_stages.issubset(set(received_stages))

    def test_progress_callback_receives_valid_percentages(self, config):
        """Progress callback arguments must satisfy: 0.0 <= stage_pct, overall_pct <= 1.0."""
        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        eval_results = _make_eval_results(test_set)

        bad_calls: list[tuple[str, float, float]] = []

        def _progress(stage: str, stage_pct: float, overall_pct: float) -> None:
            if not (0.0 <= stage_pct <= 1.0 and 0.0 <= overall_pct <= 1.0):
                bad_calls.append((stage, stage_pct, overall_pct))

        with (
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=eval_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="optimized"),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=eval_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("GO", "all good", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            Migrator(config, progress_callback=_progress).run()

        assert bad_calls == [], f"Invalid callback args: {bad_calls}"


# ---------------------------------------------------------------------------
# skip_preflight behaviour
# ---------------------------------------------------------------------------


class TestSkipPreflight:
    def test_pipeline_skip_preflight_does_not_call_run_preflight(self, config):
        """When skip_preflight=True, run_preflight is never called."""
        assert config.skip_preflight is True

        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        eval_results = _make_eval_results(test_set)

        with (
            patch(f"{_PIPELINE}.run_preflight") as mock_preflight,
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=eval_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="optimized"),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=eval_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("GO", "all good", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            Migrator(config).run()

        mock_preflight.assert_not_called()

    def test_pipeline_with_preflight_enabled_calls_run_preflight(self, data_file, tmp_path):
        """When skip_preflight=False, run_preflight IS called with the config."""
        cfg = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=data_file,
            output_dir=tmp_path / "outputs",
            skip_preflight=False,
        )

        preflight_report = _make_preflight_report()

        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        eval_results = _make_eval_results(test_set)

        with (
            patch(
                f"{_PIPELINE}.run_preflight", return_value=preflight_report
            ) as mock_preflight,
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=eval_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="optimized"),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=eval_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("GO", "all good", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            Migrator(cfg).run()

        mock_preflight.assert_called_once_with(cfg)


# ---------------------------------------------------------------------------
# Dry run behaviour
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_with_skip_preflight_returns_early(self, data_file, tmp_path):
        """dry_run=True + skip_preflight=True returns immediately without running stages."""
        cfg = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=data_file,
            output_dir=tmp_path / "outputs",
            dry_run=True,
            skip_preflight=True,
        )

        with (
            patch(f"{_PIPELINE}.load_and_split_data") as mock_load,
            patch(f"{_PIPELINE}.evaluate_baseline") as mock_baseline,
            patch(f"{_PIPELINE}.optimize_prompt") as mock_optimize,
            patch(f"{_PIPELINE}.evaluate_optimized") as mock_validate,
            patch(f"{_PIPELINE}.generate_report") as mock_report,
        ):
            result = Migrator(cfg).run()

        # When dry_run=True and skip_preflight=True, migrator returns early with NO_GO
        assert isinstance(result, MigrationResult)
        assert result.recommendation == "NO_GO"
        assert result.recommendation_reasoning is not None
        assert "Dry run" in result.recommendation_reasoning
        mock_load.assert_not_called()
        mock_baseline.assert_not_called()
        mock_optimize.assert_not_called()
        mock_validate.assert_not_called()
        mock_report.assert_not_called()

    def test_dry_run_with_preflight_stops_before_optimize(self, data_file, tmp_path):
        """dry_run=True + skip_preflight=False runs preflight then returns without optimize."""
        cfg = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=data_file,
            output_dir=tmp_path / "outputs",
            dry_run=True,
            skip_preflight=False,
        )

        dry_run_result = MigrationResult(
            config=cfg.model_dump(mode="json"),
            optimized_prompt="",
            baseline_results=[],
            validation_results=[],
            confidence_score=0.0,
            baseline_score=0.0,
            improvement=0.0,
            cost_usd=0.0,
            duration_seconds=0.0,
            warnings=["DRY RUN — no migration performed"],
        )
        preflight_report = _make_preflight_report(
            warnings=["token budget marginal"],
            estimated_cost_usd=0.5,
            dry_run_result=dry_run_result,
        )

        with (
            patch(f"{_PIPELINE}.run_preflight", return_value=preflight_report),
            patch(f"{_PIPELINE}.load_and_split_data") as mock_load,
            patch(f"{_PIPELINE}.evaluate_baseline") as mock_baseline,
            patch(f"{_PIPELINE}.optimize_prompt") as mock_optimize,
            patch(f"{_PIPELINE}.evaluate_optimized") as mock_validate,
            patch(f"{_PIPELINE}.generate_report") as mock_report,
        ):
            result = Migrator(cfg).run()

        mock_load.assert_not_called()
        mock_baseline.assert_not_called()
        mock_optimize.assert_not_called()
        mock_validate.assert_not_called()
        mock_report.assert_not_called()

        assert isinstance(result, MigrationResult)
        assert any("DRY RUN" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Regression analysis
# ---------------------------------------------------------------------------


class TestRegressionAnalysis:
    def test_prompt_regressions_populated_from_baseline_and_validation(self, config):
        """result.prompt_regressions contains one entry per (baseline, validation) pair."""
        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]
        # Baseline scores higher → regression after optimization
        baseline_results = _make_eval_results(test_set, score=0.95, is_win=True)
        validation_results = _make_eval_results(test_set, score=0.70, is_win=False)

        with (
            patch(f"{_PIPELINE}.load_and_split_data", return_value=(train, val, test_set)),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=baseline_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="regressed"),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=validation_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("NO_GO", "regression detected", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            result = Migrator(config).run()

        assert len(result.prompt_regressions) == len(test_set)
        # Each regression must have negative delta (score went down)
        for reg in result.prompt_regressions:
            assert reg.delta == pytest.approx(0.70 - 0.95)

    def test_confidence_and_baseline_scores_match_win_rates(self, config):
        """confidence_score and baseline_score reflect fraction of is_win=True results."""
        pairs = _make_pairs(5)
        train, val, test_set = pairs[:2], pairs[2:3], pairs[3:]

        # 1 out of 2 test wins for baseline → baseline_score = 0.5
        baseline_results = [
            _make_eval_results([test_set[0]], score=0.9, is_win=True)[0],
            _make_eval_results([test_set[1]], score=0.3, is_win=False)[0],
        ]
        # 2 out of 2 wins for validation → confidence_score = 1.0
        validation_results = _make_eval_results(test_set[:2], score=0.9, is_win=True)

        with (
            patch(
                f"{_PIPELINE}.load_and_split_data",
                return_value=(train, val, test_set[:2]),
            ),
            patch(f"{_PIPELINE}.evaluate_baseline", return_value=baseline_results),
            patch(f"{_PIPELINE}.optimize_prompt", return_value="better"),
            patch(f"{_PIPELINE}.run_pii_scan"),
            patch(f"{_PIPELINE}.run_pii_scan_text"),
            patch(f"{_PIPELINE}.run_prompt_audit"),
            patch(f"{_PIPELINE}.evaluate_optimized", return_value=validation_results),
            patch(
                f"{_PIPELINE}.make_recommendation",
                return_value=("GO", "improved", {}),
            ),
            patch(f"{_PIPELINE}.generate_report"),
        ):
            result = Migrator(config).run()

        assert result.baseline_score == pytest.approx(0.5)
        assert result.confidence_score == pytest.approx(1.0)
        assert result.improvement == pytest.approx(0.5)
