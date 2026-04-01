"""Tests for the pipeline API router."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from rosettastone.server.models import PipelineRecord, PipelineStageRecord

# ---------------------------------------------------------------------------
# Shared YAML fixtures
# ---------------------------------------------------------------------------

_VALID_YAML = textwrap.dedent("""\
    pipeline:
      name: test_pipeline
      source_model: openai/gpt-4o
      target_model: anthropic/claude-haiku-4-5
      modules:
        - name: step1
          prompt_template: "You are helpful."
          input_fields: [prompt]
          output_fields: [result]
          depends_on: []
        - name: step2
          prompt_template: "Refine."
          input_fields: [result]
          output_fields: [final]
          depends_on: [step1]
""")

_INVALID_YAML = "pipeline: { name: [not, a, string"  # malformed


# ---------------------------------------------------------------------------
# CRUD tests
# ---------------------------------------------------------------------------


class TestCreatePipeline:
    def test_create_pipeline_valid_yaml(self, client, engine) -> None:
        """POST with valid YAML returns 201 and PipelineSummary."""
        mock_task_worker = MagicMock()
        client.app.state.task_worker = mock_task_worker

        resp = client.post(
            "/api/v1/pipelines/migrate",
            json={
                "config_yaml": _VALID_YAML,
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-haiku-4-5",
            },
        )

        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test_pipeline"
        assert data["source_model"] == "openai/gpt-4o"
        assert data["target_model"] == "anthropic/claude-haiku-4-5"
        assert data["status"] == "pending"
        assert "id" in data
        # Background task should have been submitted
        mock_task_worker.enqueue.assert_called_once()

    def test_create_pipeline_invalid_yaml(self, client, engine) -> None:
        """POST with malformed YAML returns 422."""
        mock_task_worker = MagicMock()
        client.app.state.task_worker = mock_task_worker

        resp = client.post(
            "/api/v1/pipelines/migrate",
            json={
                "config_yaml": _INVALID_YAML,
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-haiku-4-5",
            },
        )

        assert resp.status_code == 422

    def test_create_pipeline_missing_required_fields_in_yaml(self, client, engine) -> None:
        """POST with YAML missing required pipeline fields returns 422."""
        mock_task_worker = MagicMock()
        client.app.state.task_worker = mock_task_worker

        bad_yaml = textwrap.dedent("""\
            pipeline:
              name: no_models
              modules: []
        """)

        resp = client.post(
            "/api/v1/pipelines/migrate",
            json={
                "config_yaml": bad_yaml,
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-haiku-4-5",
            },
        )

        assert resp.status_code == 422


class TestListPipelines:
    def test_list_pipelines_empty(self, client) -> None:
        """GET /api/v1/pipelines returns empty list when no pipelines exist."""
        resp = client.get("/api/v1/pipelines")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_pipelines_returns_items(self, client, engine) -> None:
        """After creating a pipeline, GET returns it in the list."""
        mock_task_worker = MagicMock()
        client.app.state.task_worker = mock_task_worker

        client.post(
            "/api/v1/pipelines/migrate",
            json={
                "config_yaml": _VALID_YAML,
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-haiku-4-5",
            },
        )

        resp = client.get("/api/v1/pipelines")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["name"] == "test_pipeline"


class TestGetPipelineStatus:
    def test_get_pipeline_status_not_found(self, client) -> None:
        """GET /api/v1/pipelines/999/status returns 404."""
        resp = client.get("/api/v1/pipelines/999/status")
        assert resp.status_code == 404

    def test_get_pipeline_status_returns_stages(self, client, engine) -> None:
        """After creating a pipeline and adding stages, status endpoint returns detail with stages."""
        # Insert pipeline directly into DB
        with Session(engine) as session:
            pipeline = PipelineRecord(
                name="staged_pipeline",
                config_yaml=_VALID_YAML,
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-haiku-4-5",
                status="completed",
            )
            session.add(pipeline)
            session.commit()
            session.refresh(pipeline)

            stage = PipelineStageRecord(
                pipeline_id=pipeline.id,
                module_name="step1",
                status="completed",
                optimized_prompt="optimized instructions",
                duration_seconds=1.5,
            )
            session.add(stage)
            session.commit()
            pipeline_id = pipeline.id

        resp = client.get(f"/api/v1/pipelines/{pipeline_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["module_name"] == "step1"
        assert data["stages"][0]["optimized_prompt"] == "optimized instructions"


class TestGetPipelineModules:
    def test_get_pipeline_modules_not_found(self, client) -> None:
        """GET /api/v1/pipelines/999/modules returns 404."""
        resp = client.get("/api/v1/pipelines/999/modules")
        assert resp.status_code == 404

    def test_get_pipeline_modules_returns_list(self, client, engine) -> None:
        """GET /api/v1/pipelines/{id}/modules returns list of stage summaries."""
        with Session(engine) as session:
            pipeline = PipelineRecord(
                name="modules_pipeline",
                config_yaml=_VALID_YAML,
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-haiku-4-5",
                status="completed",
            )
            session.add(pipeline)
            session.commit()
            session.refresh(pipeline)

            for module_name in ["step1", "step2"]:
                stage = PipelineStageRecord(
                    pipeline_id=pipeline.id,
                    module_name=module_name,
                    status="completed",
                    optimized_prompt=f"{module_name} instructions",
                    duration_seconds=0.5,
                )
                session.add(stage)
            session.commit()
            pipeline_id = pipeline.id

        resp = client.get(f"/api/v1/pipelines/{pipeline_id}/modules")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        module_names = {item["module_name"] for item in data}
        assert module_names == {"step1", "step2"}


# ---------------------------------------------------------------------------
# HTMX fragment
# ---------------------------------------------------------------------------


class TestStagesFragment:
    def test_stages_fragment_returns_html(self, client, engine) -> None:
        """GET /ui/pipelines/{id}/stages-fragment returns 200 HTML."""
        with Session(engine) as session:
            pipeline = PipelineRecord(
                name="html_pipeline",
                config_yaml=_VALID_YAML,
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-haiku-4-5",
                status="running",
            )
            session.add(pipeline)
            session.commit()
            session.refresh(pipeline)
            pipeline_id = pipeline.id

        resp = client.get(f"/ui/pipelines/{pipeline_id}/stages-fragment")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_stages_fragment_not_found(self, client) -> None:
        """GET /ui/pipelines/999/stages-fragment for missing pipeline returns 404."""
        resp = client.get("/ui/pipelines/999/stages-fragment")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Pipeline runner (unit tests with mocks)
# ---------------------------------------------------------------------------


class TestRunPipelineBackground:
    def test_run_pipeline_background_sets_running(self, engine) -> None:
        """After calling runner with mocked optimizer, status goes running → completed."""
        from rosettastone.server.pipeline_runner import run_pipeline_background

        with Session(engine) as session:
            pipeline = PipelineRecord(
                name="bg_pipeline",
                config_yaml=_VALID_YAML,
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-haiku-4-5",
                status="pending",
            )
            session.add(pipeline)
            session.commit()
            session.refresh(pipeline)
            pipeline_id = pipeline.id

        with patch(
            "rosettastone.optimize.teacher_student.TeacherStudentOptimizer"
        ) as mock_optimizer_cls:
            mock_optimizer = MagicMock()
            mock_optimizer.pipeline_optimize.return_value = {
                "step1": "optimized step1",
                "step2": "optimized step2",
            }
            mock_optimizer_cls.return_value = mock_optimizer

            run_pipeline_background(pipeline_id, engine=engine)

        with Session(engine) as session:
            pipeline = session.get(PipelineRecord, pipeline_id)
            assert pipeline is not None
            assert pipeline.status == "complete"

    def test_run_pipeline_background_not_found_returns_early(self, engine) -> None:
        """pipeline_id=999 returns without error (no exception raised)."""
        from rosettastone.server.pipeline_runner import run_pipeline_background

        # Should not raise — just log and return
        run_pipeline_background(999, engine=engine)

    def test_run_pipeline_background_on_error_sets_failed(self, engine) -> None:
        """If optimizer raises, pipeline status is set to 'failed'."""
        from rosettastone.server.pipeline_runner import run_pipeline_background

        with Session(engine) as session:
            pipeline = PipelineRecord(
                name="fail_pipeline",
                config_yaml=_VALID_YAML,
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-haiku-4-5",
                status="pending",
            )
            session.add(pipeline)
            session.commit()
            session.refresh(pipeline)
            pipeline_id = pipeline.id

        with patch(
            "rosettastone.optimize.teacher_student.TeacherStudentOptimizer"
        ) as mock_optimizer_cls:
            mock_optimizer = MagicMock()
            mock_optimizer.pipeline_optimize.side_effect = RuntimeError("optimizer exploded")
            mock_optimizer_cls.return_value = mock_optimizer

            with pytest.raises(RuntimeError):
                run_pipeline_background(pipeline_id, engine=engine)

        with Session(engine) as session:
            pipeline = session.get(PipelineRecord, pipeline_id)
            assert pipeline is not None
            assert pipeline.status == "failed"
