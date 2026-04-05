"""Tests for CLI commands: migrate, preflight, evaluate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rosettastone.cli.main import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_ARGS = [
    "--data",
    "examples/sample_data.jsonl",
    "--from",
    "openai/gpt-4o",
    "--to",
    "anthropic/claude-sonnet-4",
]


def _make_migration_result(**overrides) -> MagicMock:
    defaults = dict(
        confidence_score=0.85,
        baseline_score=0.70,
        improvement=0.15,
        cost_usd=5.0,
        duration_seconds=120.0,
        warnings=[],
        # Phase 2 fields
        recommendation=None,
        recommendation_reasoning=None,
        per_type_scores={},
        safety_warnings=[],
        validation_results=[],
        baseline_results=[],
        optimized_prompt="You are a helpful assistant.",
        config={},
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


# ---------------------------------------------------------------------------
# migrate command
# ---------------------------------------------------------------------------


def test_migrate_command():
    """migrate constructs MigrationConfig and calls Migrator.run()."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_migrator = MagicMock()
        mock_migrator.run.return_value = _make_migration_result()
        mock_cls.return_value = mock_migrator

        result = runner.invoke(app, ["migrate"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        assert "Migration complete" in result.output
        mock_cls.assert_called_once()
        mock_migrator.run.assert_called_once()


def test_migrate_output_contains_scores():
    """migrate prints confidence, baseline, and improvement."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result(
            confidence_score=0.90,
            baseline_score=0.75,
            improvement=0.15,
            cost_usd=3.50,
            duration_seconds=45.0,
        )

        result = runner.invoke(app, ["migrate"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        assert "90%" in result.output
        assert "75%" in result.output


def test_migrate_with_warnings():
    """migrate prints warnings when the result includes them."""
    warning_msg = "Token budget exceeded for 3 examples"
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result(warnings=[warning_msg])

        result = runner.invoke(app, ["migrate"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        assert warning_msg in result.output


def test_migrate_no_warnings_section_when_empty():
    """migrate does not print a Warnings section when warnings list is empty."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result(warnings=[])

        result = runner.invoke(app, ["migrate"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        assert "Warnings" not in result.output


def test_migrate_config_constructed_correctly():
    """migrate passes source_model, target_model, and data_path to Migrator."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result()

        runner.invoke(app, ["migrate"] + BASE_ARGS)

        call_args = mock_cls.call_args
        # Migrator is instantiated with a positional MigrationConfig argument
        config = call_args[0][0]
        assert config.source_model == "openai/gpt-4o"
        assert config.target_model == "anthropic/claude-sonnet-4"
        assert config.data_path == Path("examples/sample_data.jsonl")


def test_migrate_dry_run_flag():
    """--dry-run sets dry_run=True on the config passed to Migrator."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result()

        runner.invoke(app, ["migrate"] + BASE_ARGS + ["--dry-run"])

        config = mock_cls.call_args[0][0]
        assert config.dry_run is True


# ---------------------------------------------------------------------------
# preflight command
# ---------------------------------------------------------------------------


def test_preflight_command():
    """preflight runs with dry_run=True and prints pre-flight report."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_migrator = MagicMock()
        mock_migrator.run.return_value = MagicMock(warnings=["Check passed"])
        mock_cls.return_value = mock_migrator

        result = runner.invoke(app, ["preflight"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        assert "Pre-flight" in result.output
        mock_cls.assert_called_once()


def test_preflight_sets_dry_run_true():
    """preflight always sets dry_run=True on the config, regardless of flags."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = MagicMock(warnings=[])

        runner.invoke(app, ["preflight"] + BASE_ARGS)

        config = mock_cls.call_args[0][0]
        assert config.dry_run is True


def test_preflight_displays_warnings():
    """preflight prints each warning returned by the migrator."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = MagicMock(
            warnings=["Context too large", "Unsupported output type"]
        )

        result = runner.invoke(app, ["preflight"] + BASE_ARGS)

        assert "Context too large" in result.output
        assert "Unsupported output type" in result.output


# ---------------------------------------------------------------------------
# evaluate command
# ---------------------------------------------------------------------------


def test_evaluate_command():
    """evaluate calls load_and_split_data + evaluate_baseline and prints win rate."""
    mock_result = MagicMock(is_win=True)
    with (
        patch("rosettastone.core.pipeline.load_and_split_data") as mock_load,
        patch("rosettastone.core.pipeline.evaluate_baseline") as mock_eval,
    ):
        mock_load.return_value = ([], [], [mock_result, mock_result])
        mock_eval.return_value = [mock_result, mock_result]

        result = runner.invoke(app, ["evaluate"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        assert "Win rate" in result.output
        mock_load.assert_called_once()
        mock_eval.assert_called_once()


def test_evaluate_win_rate_calculation():
    """evaluate computes win rate correctly (wins / total)."""
    win = MagicMock(is_win=True)
    loss = MagicMock(is_win=False)
    with (
        patch("rosettastone.core.pipeline.load_and_split_data") as mock_load,
        patch("rosettastone.core.pipeline.evaluate_baseline") as mock_eval,
    ):
        mock_load.return_value = ([], [], [win, win, loss, loss])
        mock_eval.return_value = [win, win, loss, loss]

        result = runner.invoke(app, ["evaluate"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        # 2 wins out of 4 = 50%
        assert "2/4" in result.output
        assert "50%" in result.output


def test_evaluate_sets_skip_preflight():
    """evaluate sets skip_preflight=True on the config."""
    with (
        patch("rosettastone.core.pipeline.load_and_split_data") as mock_load,
        patch("rosettastone.core.pipeline.evaluate_baseline") as mock_eval,
    ):
        captured_config: list = []

        def capture_load(config):
            captured_config.append(config)
            return ([], [], [MagicMock(is_win=True)])

        mock_load.side_effect = capture_load
        mock_eval.return_value = [MagicMock(is_win=True)]

        runner.invoke(app, ["evaluate"] + BASE_ARGS)

        assert captured_config, "load_and_split_data was never called"
        assert captured_config[0].skip_preflight is True


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


def test_missing_data_arg_migrate():
    """migrate exits non-zero when --data is missing."""
    result = runner.invoke(
        app,
        [
            "migrate",
            "--from",
            "openai/gpt-4o",
            "--to",
            "anthropic/claude-sonnet-4",
        ],
    )
    assert result.exit_code != 0


def test_missing_from_arg_migrate():
    """migrate exits non-zero when --from is missing."""
    result = runner.invoke(
        app,
        [
            "migrate",
            "--data",
            "examples/sample_data.jsonl",
            "--to",
            "anthropic/claude-sonnet-4",
        ],
    )
    assert result.exit_code != 0


def test_missing_to_arg_migrate():
    """migrate exits non-zero when --to is missing."""
    result = runner.invoke(
        app,
        [
            "migrate",
            "--data",
            "examples/sample_data.jsonl",
            "--from",
            "openai/gpt-4o",
        ],
    )
    assert result.exit_code != 0


def test_from_alias_works():
    """--from works despite 'from' being a Python keyword."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result()

        result = runner.invoke(app, ["migrate"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        config = mock_cls.call_args[0][0]
        assert config.source_model == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


def test_serve_command_registered():
    """serve command is registered on the Typer app."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start the RosettaStone FastAPI server" in result.output


def test_serve_default_options():
    """serve uses default host 127.0.0.1 and port 8000."""
    with patch("uvicorn.run") as mock_uvicorn:
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        assert "http://127.0.0.1:8000" in result.output
        mock_uvicorn.assert_called_once()
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 8000
        assert call_kwargs["reload"] is False
        assert call_kwargs["factory"] is True


def test_serve_custom_host_port():
    """serve accepts --host and --port options."""
    with patch("uvicorn.run") as mock_uvicorn:
        result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "9000"])
        assert result.exit_code == 0
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 9000


def test_serve_reload_flag():
    """serve --reload sets reload=True."""
    with patch("uvicorn.run") as mock_uvicorn:
        result = runner.invoke(app, ["serve", "--reload"])
        assert result.exit_code == 0
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["reload"] is True


def test_serve_app_factory_reference():
    """serve passes correct app factory reference to uvicorn."""
    with patch("uvicorn.run") as mock_uvicorn:
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        # First positional arg is the app reference
        assert mock_uvicorn.call_args[0][0] == "rosettastone.server.app:create_app"


# ---------------------------------------------------------------------------
# --lm-extra-kwargs flag tests
# ---------------------------------------------------------------------------


def test_migrate_lm_extra_kwargs_valid_json():
    """--lm-extra-kwargs with valid JSON is parsed into config.lm_extra_kwargs."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result()

        lm_kwargs = '{"api_base": "http://localhost:8000/v1"}'
        result = runner.invoke(app, ["migrate"] + BASE_ARGS + ["--lm-extra-kwargs", lm_kwargs])

        assert result.exit_code == 0, result.output
        config = mock_cls.call_args[0][0]
        assert config.lm_extra_kwargs == {"api_base": "http://localhost:8000/v1"}


def test_migrate_lm_extra_kwargs_multiple_keys():
    """--lm-extra-kwargs with multiple keys is parsed correctly."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result()

        lm_kwargs = '{"api_base": "http://localhost:8000/v1", "api_key": "sk-test", "timeout": 60}'
        result = runner.invoke(app, ["migrate"] + BASE_ARGS + ["--lm-extra-kwargs", lm_kwargs])

        assert result.exit_code == 0, result.output
        config = mock_cls.call_args[0][0]
        assert config.lm_extra_kwargs == {
            "api_base": "http://localhost:8000/v1",
            "api_key": "sk-test",
            "timeout": 60,
        }


def test_migrate_lm_extra_kwargs_invalid_json():
    """--lm-extra-kwargs with invalid JSON exits with error."""
    result = runner.invoke(app, ["migrate"] + BASE_ARGS + ["--lm-extra-kwargs", "not json"])

    assert result.exit_code != 0
    assert "must be valid JSON" in result.output


def test_migrate_lm_extra_kwargs_omitted():
    """--lm-extra-kwargs omitted results in empty dict in config."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result()

        result = runner.invoke(app, ["migrate"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        config = mock_cls.call_args[0][0]
        assert config.lm_extra_kwargs == {}


# ---------------------------------------------------------------------------
# --gepa-timeout-seconds flag tests
# ---------------------------------------------------------------------------


def test_migrate_gepa_timeout_seconds_set():
    """--gepa-timeout-seconds sets gepa_timeout_seconds on config."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result()

        result = runner.invoke(app, ["migrate"] + BASE_ARGS + ["--gepa-timeout-seconds", "60"])

        assert result.exit_code == 0, result.output
        config = mock_cls.call_args[0][0]
        assert config.gepa_timeout_seconds == 60


def test_migrate_gepa_timeout_seconds_different_value():
    """--gepa-timeout-seconds accepts different integer values."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result()

        result = runner.invoke(app, ["migrate"] + BASE_ARGS + ["--gepa-timeout-seconds", "120"])

        assert result.exit_code == 0, result.output
        config = mock_cls.call_args[0][0]
        assert config.gepa_timeout_seconds == 120


def test_migrate_gepa_timeout_seconds_omitted():
    """--gepa-timeout-seconds omitted results in None in config."""
    with patch("rosettastone.core.migrator.Migrator") as mock_cls:
        mock_cls.return_value.run.return_value = _make_migration_result()

        result = runner.invoke(app, ["migrate"] + BASE_ARGS)

        assert result.exit_code == 0, result.output
        config = mock_cls.call_args[0][0]
        assert config.gepa_timeout_seconds is None
