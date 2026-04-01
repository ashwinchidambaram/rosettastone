"""Tests for per-prompt regression analysis in build_result()."""
from __future__ import annotations

from pathlib import Path

from rosettastone.config import MigrationConfig
from rosettastone.core.types import EvalResult, PromptPair


def _make_config(tmp_path: Path) -> MigrationConfig:
    data_file = tmp_path / "d.jsonl"
    data_file.write_text('{"prompt":"q","response":"a","source_model":"openai/gpt-4o"}\n')
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=data_file,
        skip_preflight=True,
    )


def _make_result(score: float, is_win: bool, output_type: str = "classification") -> EvalResult:
    return EvalResult(
        prompt_pair=PromptPair(prompt="q", response="a", source_model="openai/gpt-4o"),
        new_response="a",
        scores={"exact_match": score},
        composite_score=score,
        is_win=is_win,
        details={"output_type": output_type},
    )


class TestRegressionStatus:
    def test_improved_when_delta_ge_0_05(self, tmp_path):
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        baseline = [_make_result(0.80, True)]
        validation = [_make_result(0.86, True)]  # delta=+0.06 -> improved
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.prompt_regressions[0].status == "improved"

    def test_stable_when_delta_ge_neg_0_05(self, tmp_path):
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        baseline = [_make_result(0.85, True)]
        validation = [_make_result(0.83, True)]  # delta=-0.02 -> stable
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.prompt_regressions[0].status == "stable"

    def test_regressed_when_delta_lt_neg_0_05_but_above_threshold(self, tmp_path):
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        # classification threshold=0.90; optimized=0.91 > 0.90, but delta=-0.08
        baseline = [_make_result(0.99, True)]
        validation = [_make_result(0.91, True)]  # delta=-0.08, 0.91>0.90 -> regressed
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.prompt_regressions[0].status == "regressed"

    def test_at_risk_when_delta_lt_neg_0_05_and_below_threshold(self, tmp_path):
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        # classification threshold=0.90; optimized=0.80 < 0.90, delta=-0.10 -> at_risk
        baseline = [_make_result(0.90, True)]
        validation = [_make_result(0.80, False)]  # delta=-0.10, 0.80<0.90 -> at_risk
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.prompt_regressions[0].status == "at_risk"

    def test_stable_at_boundary_neg_0_05(self, tmp_path):
        """delta just above -0.05 (e.g. -0.04) is stable (>= -0.05)."""
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        baseline = [_make_result(0.94, True)]
        validation = [_make_result(0.90, True)]  # delta=-0.04 -> stable (above threshold too)
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.prompt_regressions[0].status == "stable"

    def test_stable_near_neg_0_05_boundary(self, tmp_path):
        """Values with delta in (-0.05, 0) are stable — tests boundary region."""
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        # delta = 0.86 - 0.90 = -0.04 → clearly above -0.05, stable
        baseline = [_make_result(0.90, True)]
        validation = [_make_result(0.86, True)]
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.prompt_regressions[0].status == "stable"
        assert result.prompt_regressions[0].delta > -0.05


class TestRegressionSort:
    def test_at_risk_sorted_before_regressed(self, tmp_path):
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        # classification threshold=0.90
        # pair 0: regressed (delta=-0.08, score=0.91 >= 0.90)
        # pair 1: at_risk   (delta=-0.15, score=0.80 < 0.90)
        baseline = [_make_result(0.99, True), _make_result(0.95, True)]
        validation = [_make_result(0.91, True), _make_result(0.80, False)]
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.prompt_regressions[0].status == "at_risk"
        assert result.prompt_regressions[1].status == "regressed"

    def test_within_at_risk_sorted_by_delta_ascending(self, tmp_path):
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        # Both at_risk; pair 0 delta=-0.20, pair 1 delta=-0.15
        # Pair 0 should come first (worse delta)
        baseline = [_make_result(0.95, True), _make_result(0.95, True)]
        validation = [_make_result(0.75, False), _make_result(0.80, False)]
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.prompt_regressions[0].delta < result.prompt_regressions[1].delta


class TestLengthMismatch:
    def test_length_mismatch_produces_only_matched_count(self, tmp_path):
        """zip stops at shorter list -- no exception, regression count = min(len(b), len(v))."""
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        baseline = [_make_result(0.90, True) for _ in range(5)]
        validation = [_make_result(0.85, True) for _ in range(3)]
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert len(result.prompt_regressions) == 3


class TestCounts:
    def test_regression_and_at_risk_counts(self, tmp_path):
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        # classification threshold=0.90
        baseline = [
            _make_result(0.99, True),  # -> regressed (0.91, delta=-0.08, above threshold)
            _make_result(0.95, True),  # -> at_risk (0.80, delta=-0.15, below threshold)
            _make_result(0.85, True),  # -> improved (0.92, delta=+0.07)
        ]
        validation = [
            _make_result(0.91, True),
            _make_result(0.80, False),
            _make_result(0.92, True),
        ]
        result = build_result(config, "opt", baseline, validation, 1.0)
        assert result.regression_count == 1
        assert result.at_risk_count == 1

    def test_metric_deltas_populated(self, tmp_path):
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        base = EvalResult(
            prompt_pair=PromptPair(prompt="q", response="a", source_model="openai/gpt-4o"),
            new_response="a",
            scores={"bertscore": 0.80, "exact_match": 0.90},
            composite_score=0.85,
            is_win=True,
            details={"output_type": "classification"},
        )
        val = EvalResult(
            prompt_pair=PromptPair(prompt="q", response="a", source_model="openai/gpt-4o"),
            new_response="b",
            scores={"bertscore": 0.90, "exact_match": 0.70},
            composite_score=0.80,
            is_win=True,
            details={"output_type": "classification"},
        )
        result = build_result(config, "opt", [base], [val], 1.0)
        r = result.prompt_regressions[0]
        assert abs(r.metric_deltas["bertscore"] - 0.10) < 1e-9
        assert abs(r.metric_deltas["exact_match"] - (-0.20)) < 1e-9
