"""Background task runner for pipeline migrations triggered from the web UI."""

from __future__ import annotations

import logging
import time

from sqlmodel import Session

logger = logging.getLogger(__name__)


def run_pipeline_background(pipeline_id: int, engine=None) -> None:
    """Background task: parse YAML, run GEPA on pipeline, persist per-module results.

    Args:
        pipeline_id: ID of the PipelineRecord to optimize.
        engine: SQLAlchemy engine (defaults to get_engine(); pass test engine for testing).
    """
    if engine is None:
        from rosettastone.server.database import get_engine

        engine = get_engine()

    # Load and mark running
    with Session(engine) as session:
        from rosettastone.server.models import PipelineRecord

        pipeline = session.get(PipelineRecord, pipeline_id)
        if pipeline is None:
            logger.error("Pipeline %d not found", pipeline_id)
            return

        pipeline.status = "running"
        session.add(pipeline)
        session.commit()
        session.refresh(pipeline)

        # Capture fields before session closes
        config_yaml = pipeline.config_yaml

    try:
        import yaml

        from rosettastone.optimize.pipeline_config import PipelineConfig

        raw = yaml.safe_load(config_yaml)
        config = PipelineConfig(**raw.get("pipeline", raw))

        # Load training data
        from pathlib import Path

        train_set: list = []
        if config.data_path:
            from rosettastone.ingest.jsonl import JSONLAdapter

            adapter = JSONLAdapter(Path(config.data_path))
            train_set = adapter.load()
            logger.info(
                "Pipeline %d: loaded %d training pairs from data_path",
                pipeline_id,
                len(train_set),
            )
        else:
            logger.info("Pipeline %d: no data_path set, running with empty train set", pipeline_id)

        # Build MigrationConfig
        from rosettastone.config import MigrationConfig

        migration_config = MigrationConfig(
            source_model=config.source_model,
            target_model=config.target_model,
            data_path=Path(config.data_path) if config.data_path else None,
        )

        # Run Teacher/Student optimization
        from rosettastone.optimize.teacher_student import TeacherStudentOptimizer

        optimizer = TeacherStudentOptimizer()
        start_time = time.time()
        optimized_modules = optimizer.pipeline_optimize(config, train_set, migration_config)
        elapsed = time.time() - start_time

        logger.info(
            "Pipeline %d: optimization complete in %.1fs, %d modules optimized",
            pipeline_id,
            elapsed,
            len(optimized_modules),
        )

        # Persist per-module results
        num_modules = len(config.modules) or 1
        per_module_duration = elapsed / num_modules

        with Session(engine) as session:
            from rosettastone.server.models import PipelineRecord, PipelineStageRecord

            pipeline = session.get(PipelineRecord, pipeline_id)
            if pipeline is None:
                return

            for module in config.modules:
                stage = PipelineStageRecord(
                    pipeline_id=pipeline_id,
                    module_name=module.name,
                    status="completed",
                    optimized_prompt=optimized_modules.get(module.name, ""),
                    duration_seconds=per_module_duration,
                )
                session.add(stage)

            pipeline.status = "completed"
            session.add(pipeline)
            session.commit()

    except Exception as exc:
        logger.error("Pipeline %d optimization failed: %s", pipeline_id, type(exc).__name__)
        try:
            with Session(engine) as session:
                from rosettastone.server.models import PipelineRecord

                pipeline = session.get(PipelineRecord, pipeline_id)
                if pipeline is None:
                    return
                pipeline.status = "failed"
                session.add(pipeline)
                session.commit()
        except Exception as commit_err:
            logger.error(
                "Failed to update pipeline %d status to failed: %s", pipeline_id, commit_err
            )
        raise
