"""Tests for migration checkpointing and resume."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlmodel import Session

from rosettastone.server.models import MigrationRecord


class TestCheckpointWriter:
    def test_checkpoint_writer_saves_stage_and_data(self, engine) -> None:
        """_make_checkpoint_writer callback writes checkpoint_stage and data to DB."""
        from rosettastone.server.api.tasks import _make_checkpoint_writer

        # Create a migration record
        with Session(engine) as sess:
            record = MigrationRecord(source_model="a/b", target_model="c/d", status="running")
            sess.add(record)
            sess.commit()
            sess.refresh(record)
            mid = record.id

        writer = _make_checkpoint_writer(mid, engine)
        writer("baseline_eval", json.dumps({"stage_output": {"score": 0.85}}))

        with Session(engine) as sess:
            updated = sess.get(MigrationRecord, mid)
            assert updated.checkpoint_stage == "baseline_eval"
            assert json.loads(updated.checkpoint_data_json)["stage_output"]["score"] == 0.85

    def test_checkpoint_writer_overwrites_previous_checkpoint(self, engine) -> None:
        """Each call to the writer updates to the latest stage."""
        from rosettastone.server.api.tasks import _make_checkpoint_writer

        with Session(engine) as sess:
            record = MigrationRecord(source_model="a/b", target_model="c/d", status="running")
            sess.add(record)
            sess.commit()
            sess.refresh(record)
            mid = record.id

        writer = _make_checkpoint_writer(mid, engine)
        writer("baseline_eval", json.dumps({"stage_output": {"score": 0.7}}))
        writer("optimize", json.dumps({"stage_output": {"optimized_prompt": "new prompt"}}))

        with Session(engine) as sess:
            updated = sess.get(MigrationRecord, mid)
            assert updated.checkpoint_stage == "optimize"
            payload = json.loads(updated.checkpoint_data_json)
            assert payload["stage_output"]["optimized_prompt"] == "new prompt"

    def test_checkpoint_writer_missing_record_does_not_raise(self, engine) -> None:
        """Writer silently does nothing when migration_id doesn't exist."""
        from rosettastone.server.api.tasks import _make_checkpoint_writer

        writer = _make_checkpoint_writer(99999, engine)
        # Should not raise
        writer("baseline_eval", json.dumps({"stage_output": None}))


class TestResumeEndpoint:
    def test_resume_failed_migration_with_checkpoint(self, client: TestClient, engine) -> None:
        """POST /api/v1/migrations/{id}/resume re-enqueues a failed migration."""
        with Session(engine) as sess:
            record = MigrationRecord(
                source_model="a/b",
                target_model="c/d",
                status="failed",
                checkpoint_stage="baseline_eval",
                checkpoint_data_json=json.dumps({"stage_output": {"score": 0.8}}),
                config_json=json.dumps({"source_model": "a/b", "target_model": "c/d"}),
            )
            sess.add(record)
            sess.commit()
            sess.refresh(record)
            mid = record.id

        # Set up a mock task worker on app.state (required by the endpoint)
        mock_task_worker = MagicMock()
        client.app.state.task_worker = mock_task_worker

        resp = client.post(f"/api/v1/migrations/{mid}/resume")

        assert resp.status_code == 200
        mock_task_worker.enqueue.assert_called_once()

        # Verify _resume_from was passed to enqueue
        call_args = mock_task_worker.enqueue.call_args
        payload = call_args[0][2]  # third positional arg is config dict
        assert payload["_resume_from"] == "baseline_eval"

        with Session(engine) as sess:
            updated = sess.get(MigrationRecord, mid)
            assert updated.status == "pending"

    def test_resume_running_migration_returns_409(self, client: TestClient, engine) -> None:
        """POST /api/v1/migrations/{id}/resume returns 409 for non-failed migration."""
        client.app.state.task_worker = MagicMock()
        with Session(engine) as sess:
            record = MigrationRecord(
                source_model="a/b",
                target_model="c/d",
                status="running",
                checkpoint_stage="optimize",
                config_json=json.dumps({"source_model": "a/b", "target_model": "c/d"}),
            )
            sess.add(record)
            sess.commit()
            sess.refresh(record)
            mid = record.id

        resp = client.post(f"/api/v1/migrations/{mid}/resume")
        assert resp.status_code == 409

    def test_resume_without_checkpoint_returns_409(self, client: TestClient, engine) -> None:
        """POST /api/v1/migrations/{id}/resume returns 409 when no checkpoint."""
        client.app.state.task_worker = MagicMock()
        with Session(engine) as sess:
            record = MigrationRecord(
                source_model="a/b",
                target_model="c/d",
                status="failed",
                config_json=json.dumps({"source_model": "a/b", "target_model": "c/d"}),
            )
            sess.add(record)
            sess.commit()
            sess.refresh(record)
            mid = record.id

        resp = client.post(f"/api/v1/migrations/{mid}/resume")
        assert resp.status_code == 409

    def test_resume_complete_migration_returns_409(self, client: TestClient, engine) -> None:
        """POST /api/v1/migrations/{id}/resume returns 409 for completed migration."""
        client.app.state.task_worker = MagicMock()
        with Session(engine) as sess:
            record = MigrationRecord(
                source_model="a/b",
                target_model="c/d",
                status="complete",
                checkpoint_stage="validation_eval",
                config_json=json.dumps({"source_model": "a/b", "target_model": "c/d"}),
            )
            sess.add(record)
            sess.commit()
            sess.refresh(record)
            mid = record.id

        resp = client.post(f"/api/v1/migrations/{mid}/resume")
        assert resp.status_code == 409

    def test_resume_nonexistent_migration_returns_404(self, client: TestClient) -> None:
        """POST /api/v1/migrations/99999/resume returns 404."""
        resp = client.post("/api/v1/migrations/99999/resume")
        assert resp.status_code == 404


class TestMigratorCheckpointing:
    def test_checkpoint_callback_accepted_and_stored(self) -> None:
        """Migrator accepts checkpoint_callback and stores it on self."""
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator

        calls: list[tuple[str, str]] = []

        def cb(stage: str, data: str) -> None:
            calls.append((stage, data))

        config = MigrationConfig(
            source_model="x/a",
            target_model="x/b",
            data_path="examples/sample_data.jsonl",
            dry_run=True,
        )
        m = Migrator(config, checkpoint_callback=cb)
        assert m.checkpoint_callback is cb

    def test_migrator_accepts_resume_params(self) -> None:
        """Migrator accepts resume_checkpoint_stage and resume_checkpoint_data."""
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator

        config = MigrationConfig(
            source_model="x/a",
            target_model="x/b",
            data_path="examples/sample_data.jsonl",
            dry_run=True,
        )
        m = Migrator(
            config,
            resume_checkpoint_stage="baseline_eval",
            resume_checkpoint_data='{"stage_output": {"score": 0.8}}',
        )
        assert m.resume_checkpoint_stage == "baseline_eval"
        assert m.resume_checkpoint_data == '{"stage_output": {"score": 0.8}}'

    def test_checkpoint_callback_called_after_pipeline_stages(self) -> None:
        """Migrator calls checkpoint_callback after each substantive pipeline stage."""
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator
        from rosettastone.core.types import EvalResult, MigrationResult, PromptPair

        checkpoints: list[str] = []

        def cb(stage: str, data: str) -> None:
            checkpoints.append(stage)

        config = MigrationConfig(
            source_model="x/a",
            target_model="x/b",
            data_path="examples/sample_data.jsonl",
            skip_preflight=True,
        )

        pair = PromptPair(
            prompt="hi",
            response="hello",
            source_model="x/a",
        )
        eval_result = EvalResult(
            prompt_pair=pair,
            new_response="hello",
            scores={"composite": 0.9},
            composite_score=0.9,
            is_win=True,
        )
        mock_result = MigrationResult(
            config={},
            optimized_prompt="opt prompt",
            baseline_results=[eval_result],
            validation_results=[eval_result],
            confidence_score=1.0,
            baseline_score=1.0,
            improvement=0.0,
            cost_usd=0.0,
            duration_seconds=0.1,
            warnings=[],
        )

        with (
            patch(
                "rosettastone.core.pipeline.load_and_split_data",
                return_value=([pair], [pair], [pair]),
            ),
            patch("rosettastone.core.pipeline.run_pii_scan"),
            patch("rosettastone.core.pipeline.evaluate_baseline", return_value=[eval_result]),
            patch("rosettastone.core.pipeline.optimize_prompt", return_value="opt prompt"),
            patch("rosettastone.core.pipeline.run_pii_scan_text"),
            patch("rosettastone.core.pipeline.run_prompt_audit"),
            patch("rosettastone.core.pipeline.evaluate_optimized", return_value=[eval_result]),
            patch("rosettastone.core.pipeline.make_recommendation", return_value=("GO", "ok", {})),
            patch("rosettastone.core.pipeline.build_result", return_value=mock_result),
            patch("rosettastone.core.pipeline.generate_report"),
        ):
            Migrator(config, checkpoint_callback=cb).run()

        # Verify the key stages were checkpointed
        assert "ingest" in checkpoints
        assert "baseline_eval" in checkpoints
        assert "optimize" in checkpoints
        assert "validation_eval" in checkpoints

    def test_checkpoint_callback_exception_does_not_abort(self) -> None:
        """A checkpoint_callback that raises must not abort the migration."""
        from rosettastone.config import MigrationConfig
        from rosettastone.core.migrator import Migrator
        from rosettastone.core.types import EvalResult, MigrationResult, PromptPair

        def bad_cb(stage: str, data: str) -> None:
            raise RuntimeError("checkpoint broke")

        config = MigrationConfig(
            source_model="x/a",
            target_model="x/b",
            data_path="examples/sample_data.jsonl",
            skip_preflight=True,
        )
        pair = PromptPair(prompt="hi", response="hello", source_model="x/a")
        eval_result = EvalResult(
            prompt_pair=pair,
            new_response="hello",
            scores={"composite": 0.9},
            composite_score=0.9,
            is_win=True,
        )
        mock_result = MigrationResult(
            config={},
            optimized_prompt="opt",
            baseline_results=[eval_result],
            validation_results=[eval_result],
            confidence_score=1.0,
            baseline_score=1.0,
            improvement=0.0,
            cost_usd=0.0,
            duration_seconds=0.1,
            warnings=[],
        )

        with (
            patch(
                "rosettastone.core.pipeline.load_and_split_data",
                return_value=([pair], [pair], [pair]),
            ),
            patch("rosettastone.core.pipeline.run_pii_scan"),
            patch("rosettastone.core.pipeline.evaluate_baseline", return_value=[eval_result]),
            patch("rosettastone.core.pipeline.optimize_prompt", return_value="opt"),
            patch("rosettastone.core.pipeline.run_pii_scan_text"),
            patch("rosettastone.core.pipeline.run_prompt_audit"),
            patch("rosettastone.core.pipeline.evaluate_optimized", return_value=[eval_result]),
            patch("rosettastone.core.pipeline.make_recommendation", return_value=("GO", "ok", {})),
            patch("rosettastone.core.pipeline.build_result", return_value=mock_result),
            patch("rosettastone.core.pipeline.generate_report"),
        ):
            result = Migrator(config, checkpoint_callback=bad_cb).run()

        # Migration completed despite callback raising
        assert isinstance(result, MigrationResult)
