"""Tests for shadow log evaluator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rosettastone.config import MigrationConfig
from rosettastone.shadow.log_format import ShadowLogEntry, write_log_entry


def _make_config(tmp_path: Path) -> MigrationConfig:
    data_file = tmp_path / "d.jsonl"
    data_file.write_text('{"prompt":"q","response":"a","source_model":"openai/gpt-4o"}\n')
    return MigrationConfig(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        data_path=data_file,
        skip_preflight=True,
    )


class TestScoreShadowLogs:
    def test_empty_dir_returns_zero_summary(self, tmp_path):
        from rosettastone.shadow.evaluator import score_shadow_logs

        config = _make_config(tmp_path)
        result = score_shadow_logs(tmp_path / "empty_logs", config)
        assert result["total_pairs"] == 0
        assert result["win_rate"] == 0.0
        assert len(result["warnings"]) >= 1

    def test_five_entries_produces_valid_summary(self, tmp_path):
        from rosettastone.core.types import EvalResult, PromptPair
        from rosettastone.shadow.evaluator import score_shadow_logs

        log_dir = tmp_path / "logs"
        for i in range(5):
            write_log_entry(
                ShadowLogEntry(
                    prompt=f"q{i}",
                    source_model="openai/gpt-4o",
                    target_model="anthropic/claude-sonnet-4",
                    source_response=f"answer {i}",
                    target_response=f"answer {i}",  # identical -> high score
                ),
                log_dir,
            )

        config = _make_config(tmp_path)

        mock_result = [
            EvalResult(
                prompt_pair=PromptPair(
                    prompt=f"q{i}",
                    response=f"answer {i}",
                    source_model="openai/gpt-4o",
                ),
                new_response=f"answer {i}",
                scores={"exact_match": 1.0},
                composite_score=1.0,
                is_win=True,
                details={"output_type": "classification"},
            )
            for i in range(5)
        ]

        with patch(
            "rosettastone.evaluate.composite.CompositeEvaluator.evaluate_multi_run",
            return_value=mock_result,
        ):
            result = score_shadow_logs(log_dir, config)

        assert result["total_pairs"] == 5
        assert result["wins"] == 5
        assert result["win_rate"] == 1.0
        assert "classification" in result["per_type_scores"]

    def test_score_shadow_logs_returns_dict(self, tmp_path):
        """score_shadow_logs returns a dict with all expected top-level keys."""
        from rosettastone.core.types import EvalResult, PromptPair
        from rosettastone.shadow.evaluator import score_shadow_logs

        log_dir = tmp_path / "logs"
        write_log_entry(
            ShadowLogEntry(
                prompt="What is 2+2?",
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                source_response="4",
                target_response="4",
            ),
            log_dir,
        )

        config = _make_config(tmp_path)

        mock_result = [
            EvalResult(
                prompt_pair=PromptPair(
                    prompt="What is 2+2?",
                    response="4",
                    source_model="openai/gpt-4o",
                ),
                new_response="4",
                scores={"exact_match": 1.0},
                composite_score=1.0,
                is_win=True,
                details={"output_type": "classification"},
            )
        ]

        with patch(
            "rosettastone.evaluate.composite.CompositeEvaluator.evaluate_multi_run",
            return_value=mock_result,
        ):
            result = score_shadow_logs(log_dir, config)

        assert isinstance(result, dict)
        for expected_key in (
            "win_rate", "total_pairs", "wins", "non_deterministic_count",
            "cost_usd", "per_type_scores", "warnings",
        ):
            assert expected_key in result, f"Missing key: {expected_key}"
        assert result["total_pairs"] == 1

    def test_score_shadow_logs_empty_directory(self, tmp_path):
        """Pointing at an empty directory returns gracefully with zero counts."""
        from rosettastone.shadow.evaluator import score_shadow_logs

        log_dir = tmp_path / "empty_logs"
        log_dir.mkdir()

        config = _make_config(tmp_path)
        result = score_shadow_logs(log_dir, config)

        assert result["total_pairs"] == 0
        assert result["wins"] == 0
        assert result["win_rate"] == 0.0
        assert result["per_type_scores"] == {}
        assert len(result["warnings"]) >= 1

    def test_score_shadow_logs_malformed_entry(self, tmp_path):
        """A JSONL file with one valid and one malformed line: the valid line is processed."""
        from rosettastone.core.types import EvalResult, PromptPair
        from rosettastone.shadow.evaluator import score_shadow_logs

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        valid_entry = ShadowLogEntry(
            prompt="Hello?",
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            source_response="Hi there",
            target_response="Hello!",
        )

        log_file = log_dir / "shadow_2024-01-01.jsonl"
        with log_file.open("w") as f:
            f.write(valid_entry.model_dump_json() + "\n")
            f.write("this is not valid json {{{\n")  # malformed line, should be skipped

        config = _make_config(tmp_path)

        mock_result = [
            EvalResult(
                prompt_pair=PromptPair(
                    prompt="Hello?",
                    response="Hi there",
                    source_model="openai/gpt-4o",
                ),
                new_response="Hello!",
                scores={"exact_match": 0.5},
                composite_score=0.5,
                is_win=False,
                details={"output_type": "classification"},
            )
        ]

        with patch(
            "rosettastone.evaluate.composite.CompositeEvaluator.evaluate_multi_run",
            return_value=mock_result,
        ):
            result = score_shadow_logs(log_dir, config)

        # Only the valid entry should be processed; malformed line silently skipped
        assert result["total_pairs"] == 1

    def test_score_shadow_logs_win_rate_calculation(self, tmp_path):
        """Win rate is computed correctly from known win/loss results."""
        from rosettastone.core.types import EvalResult, PromptPair
        from rosettastone.shadow.evaluator import score_shadow_logs

        log_dir = tmp_path / "logs"

        # Write 4 entries
        for i in range(4):
            write_log_entry(
                ShadowLogEntry(
                    prompt=f"prompt {i}",
                    source_model="openai/gpt-4o",
                    target_model="anthropic/claude-sonnet-4",
                    source_response=f"source answer {i}",
                    target_response=f"target answer {i}",
                ),
                log_dir,
            )

        config = _make_config(tmp_path)

        # 3 wins, 1 loss -> win_rate = 0.75
        def _make_result(idx: int, is_win: bool) -> EvalResult:
            score = 1.0 if is_win else 0.0
            return EvalResult(
                prompt_pair=PromptPair(
                    prompt=f"prompt {idx}",
                    response=f"source answer {idx}",
                    source_model="openai/gpt-4o",
                ),
                new_response=f"target answer {idx}",
                scores={"exact_match": score},
                composite_score=score,
                is_win=is_win,
                details={"output_type": "classification"},
            )

        mock_result = [
            _make_result(0, is_win=True),
            _make_result(1, is_win=True),
            _make_result(2, is_win=True),
            _make_result(3, is_win=False),
        ]

        with patch(
            "rosettastone.evaluate.composite.CompositeEvaluator.evaluate_multi_run",
            return_value=mock_result,
        ):
            result = score_shadow_logs(log_dir, config)

        assert result["total_pairs"] == 4
        assert result["wins"] == 3
        assert abs(result["win_rate"] - 0.75) < 1e-9
