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
                    target_response=f"answer {i}",  # identical → high score
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
