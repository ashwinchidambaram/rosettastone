"""Granular unit tests for pipeline_runner.py."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from rosettastone.server.models import PipelineRecord, PipelineStageRecord
from rosettastone.server.pipeline_runner import run_pipeline_background

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_YAML = textwrap.dedent("""\
    pipeline:
      name: runner_test_pipeline
      source_model: openai/gpt-4o
      target_model: anthropic/claude-haiku-4-5
      modules:
        - name: module1
          prompt_template: "You are helpful."
          input_fields: [prompt]
          output_fields: [result]
          depends_on: []
        - name: module2
          prompt_template: "Refine."
          input_fields: [result]
          output_fields: [final]
          depends_on: [module1]
""")


@pytest.fixture()
def mem_engine():
    """In-memory SQLite engine with StaticPool for cross-thread access."""
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


def _insert_pipeline(session: Session, config_yaml: str = _VALID_YAML) -> PipelineRecord:
    """Helper: insert a pending PipelineRecord and return it."""
    pipeline = PipelineRecord(
        name="runner_test",
        config_yaml=config_yaml,
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-haiku-4-5",
        status="pending",
    )
    session.add(pipeline)
    session.commit()
    session.refresh(pipeline)
    return pipeline


def _mock_optimizer(return_value: dict[str, str]) -> MagicMock:
    """Build a mocked TeacherStudentOptimizer instance."""
    optimizer = MagicMock()
    optimizer.pipeline_optimize.return_value = return_value
    return optimizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunnerNoDataPath:
    def test_runner_loads_empty_train_set_when_no_data_path(self, mem_engine) -> None:
        """config.data_path=None results in train_set=[] (no file loading attempted)."""
        with Session(mem_engine) as session:
            pipeline = _insert_pipeline(session, _VALID_YAML)
            pipeline_id = pipeline.id

        captured_calls: list[list] = []

        def fake_pipeline_optimize(config, train_set, migration_config):  # type: ignore[no-untyped-def]
            captured_calls.append(list(train_set))
            return {m.name: "instructions" for m in config.modules}

        with patch("rosettastone.optimize.teacher_student.TeacherStudentOptimizer") as mock_cls:
            mock_optimizer = MagicMock()
            mock_optimizer.pipeline_optimize.side_effect = fake_pipeline_optimize
            mock_cls.return_value = mock_optimizer

            run_pipeline_background(pipeline_id, engine=mem_engine)

        assert len(captured_calls) == 1
        assert captured_calls[0] == []  # empty train set


class TestRunnerStageRecords:
    def test_runner_creates_stage_records(self, mem_engine) -> None:
        """One PipelineStageRecord per module is created after successful run."""
        with Session(mem_engine) as session:
            pipeline = _insert_pipeline(session)
            pipeline_id = pipeline.id

        with patch("rosettastone.optimize.teacher_student.TeacherStudentOptimizer") as mock_cls:
            mock_cls.return_value = _mock_optimizer({"module1": "opt1", "module2": "opt2"})
            run_pipeline_background(pipeline_id, engine=mem_engine)

        with Session(mem_engine) as session:
            stages = list(
                session.exec(
                    select(PipelineStageRecord).where(
                        PipelineStageRecord.pipeline_id == pipeline_id
                    )
                ).all()
            )

        assert len(stages) == 2
        stage_names = {s.module_name for s in stages}
        assert stage_names == {"module1", "module2"}

    def test_runner_optimized_prompt_persisted(self, mem_engine) -> None:
        """optimized_prompt from optimizer output is stored on each stage record."""
        with Session(mem_engine) as session:
            pipeline = _insert_pipeline(session)
            pipeline_id = pipeline.id

        with patch("rosettastone.optimize.teacher_student.TeacherStudentOptimizer") as mock_cls:
            mock_cls.return_value = _mock_optimizer(
                {"module1": "instructions for module1", "module2": "instructions for module2"}
            )
            run_pipeline_background(pipeline_id, engine=mem_engine)

        with Session(mem_engine) as session:
            stages = list(
                session.exec(
                    select(PipelineStageRecord).where(
                        PipelineStageRecord.pipeline_id == pipeline_id
                    )
                ).all()
            )

        stage_map = {s.module_name: s for s in stages}
        assert stage_map["module1"].optimized_prompt == "instructions for module1"
        assert stage_map["module2"].optimized_prompt == "instructions for module2"

    def test_runner_stage_duration_set(self, mem_engine) -> None:
        """duration_seconds is non-None and non-negative after a successful run."""
        with Session(mem_engine) as session:
            pipeline = _insert_pipeline(session)
            pipeline_id = pipeline.id

        with patch("rosettastone.optimize.teacher_student.TeacherStudentOptimizer") as mock_cls:
            mock_cls.return_value = _mock_optimizer({"module1": "x", "module2": "y"})
            run_pipeline_background(pipeline_id, engine=mem_engine)

        with Session(mem_engine) as session:
            stages = list(
                session.exec(
                    select(PipelineStageRecord).where(
                        PipelineStageRecord.pipeline_id == pipeline_id
                    )
                ).all()
            )

        for stage in stages:
            assert stage.duration_seconds is not None
            assert stage.duration_seconds >= 0.0


class TestRunnerStatusTransitions:
    def test_runner_marks_completed(self, mem_engine) -> None:
        """Pipeline status = 'completed' on successful run."""
        with Session(mem_engine) as session:
            pipeline = _insert_pipeline(session)
            pipeline_id = pipeline.id

        with patch("rosettastone.optimize.teacher_student.TeacherStudentOptimizer") as mock_cls:
            mock_cls.return_value = _mock_optimizer({"module1": "a", "module2": "b"})
            run_pipeline_background(pipeline_id, engine=mem_engine)

        with Session(mem_engine) as session:
            pipeline = session.get(PipelineRecord, pipeline_id)
            assert pipeline is not None
            assert pipeline.status == "completed"

    def test_runner_marks_failed_on_exception(self, mem_engine) -> None:
        """Status = 'failed' when optimizer raises an exception."""
        with Session(mem_engine) as session:
            pipeline = _insert_pipeline(session)
            pipeline_id = pipeline.id

        with patch("rosettastone.optimize.teacher_student.TeacherStudentOptimizer") as mock_cls:
            mock_optimizer = MagicMock()
            mock_optimizer.pipeline_optimize.side_effect = ValueError("something went wrong")
            mock_cls.return_value = mock_optimizer

            with pytest.raises(ValueError):
                run_pipeline_background(pipeline_id, engine=mem_engine)

        with Session(mem_engine) as session:
            pipeline = session.get(PipelineRecord, pipeline_id)
            assert pipeline is not None
            assert pipeline.status == "failed"

    def test_runner_not_found_returns_early(self, mem_engine) -> None:
        """pipeline_id that doesn't exist returns without error or exception."""
        # Should not raise — just log and return
        run_pipeline_background(99999, engine=mem_engine)
