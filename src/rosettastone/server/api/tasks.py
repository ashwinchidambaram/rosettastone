"""Background task runner for migrations triggered from the web UI."""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlmodel import Session

from rosettastone.server.api.audit import log_audit
from rosettastone.server.api.versioning import create_version
from rosettastone.server.database import get_engine
from rosettastone.server.logging_config import set_request_id
from rosettastone.server.models import (
    GEPAIterationRecord,
    MigrationRecord,
    TestCaseRecord,
    WarningRecord,
)

logger = logging.getLogger(__name__)


def _sample_latency(result: Any, config: Any) -> dict[str, Any] | None:
    """Sample latency by calling source and target models with a few prompts."""
    try:
        import litellm

        # Take up to 5 prompts from validation results
        prompts: list[Any] = []
        for er in result.validation_results[:5]:
            prompt = er.prompt_pair.prompt
            if isinstance(prompt, str):
                prompts.append(prompt)
            elif isinstance(prompt, list):
                prompts.append(prompt)

        if not prompts:
            return None

        source_times: list[float] = []
        target_times: list[float] = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            try:
                start = time.perf_counter()
                litellm.completion(model=config.source_model, messages=messages, max_tokens=50)
                source_times.append(time.perf_counter() - start)
            except Exception:
                pass

            try:
                start = time.perf_counter()
                litellm.completion(model=config.target_model, messages=messages, max_tokens=50)
                target_times.append(time.perf_counter() - start)
            except Exception:
                pass

        data: dict[str, float] = {}
        if source_times:
            source_sorted = sorted(source_times)
            data["source_p50"] = statistics.median(source_sorted)
            idx = int(len(source_sorted) * 0.95)
            data["source_p95"] = source_sorted[idx] if len(source_sorted) > 1 else source_sorted[0]
        if target_times:
            target_sorted = sorted(target_times)
            data["target_p50"] = statistics.median(target_sorted)
            idx = int(len(target_sorted) * 0.95)
            data["target_p95"] = target_sorted[idx] if len(target_sorted) > 1 else target_sorted[0]

        return data if data else None
    except ImportError:
        logger.debug("litellm not available for latency sampling")
        return None
    except Exception as exc:
        logger.warning("Latency sampling failed: %s", exc)
        return None


def _estimate_per_call_cost(config: Any) -> dict[str, Any] | None:
    """Estimate per-call cost using litellm model pricing."""
    try:
        import litellm

        data: dict[str, float] = {}
        models = [("source_cost", config.source_model), ("target_cost", config.target_model)]
        for label, model in models:
            try:
                info = litellm.get_model_info(model)
                input_cost = info.get("input_cost_per_token", 0) or 0
                output_cost = info.get("output_cost_per_token", 0) or 0
                # Estimate: 500 input tokens + 200 output tokens per call
                data[label] = input_cost * 500 + output_cost * 200
            except Exception:
                pass

        return data if data else None
    except ImportError:
        return None
    except Exception:
        return None


def _make_checkpoint_writer(migration_id: int, engine: Any) -> Callable[[str, str], None]:
    """Return a callback that persists a checkpoint after each pipeline stage."""

    def write_checkpoint(stage: str, data_json: str) -> None:
        try:
            with Session(engine) as sess:
                record = sess.get(MigrationRecord, migration_id)
                if record is not None:
                    record.checkpoint_stage = stage
                    record.checkpoint_data_json = data_json
                    sess.add(record)
                    sess.commit()
        except Exception as exc:
            logger.warning(
                "Failed to write checkpoint for migration %d at stage %s: %s",
                migration_id,
                stage,
                exc,
            )

    return write_checkpoint


def _make_gepa_callback(migration_id: int, engine: Any = None) -> Callable[[int, int, float], None]:
    """Return a callback that emits SSE gepa_iteration events from the GEPA optimizer thread."""

    def on_iteration(iteration: int, total_iterations: int, mean_score: float) -> None:
        from rosettastone.server.progress import emit_progress

        emit_progress(
            migration_id,
            {
                "type": "gepa_iteration",
                "migration_id": migration_id,
                "iteration": iteration,
                "total_iterations": total_iterations,
                "running_mean_score": mean_score,
            },
        )

        # Fire-and-forget DB write — swallow any error so SSE is unaffected
        try:
            _eng = engine if engine is not None else get_engine()
            with Session(_eng) as _s:
                _s.add(
                    GEPAIterationRecord(
                        migration_id=migration_id,
                        iteration=iteration,
                        total_iterations=total_iterations,
                        mean_score=mean_score,
                    )
                )
                _s.commit()
        except Exception as exc:
            logger.warning(
                "Failed to persist GEPA iteration %d for migration %d: %s",
                iteration,
                migration_id,
                exc,
            )

    return on_iteration


def _make_eval_pair_callback(migration_id: int, engine: Any) -> Any:
    """Return a callback that emits SSE eval_pair events after each evaluated pair."""

    def on_eval_pair(pair_idx: int, total: int, score: float, output_type: str) -> None:
        from rosettastone.server.progress import emit_progress

        emit_progress(
            migration_id,
            {
                "type": "eval_pair",
                "migration_id": migration_id,
                "pair_index": pair_idx,
                "total_pairs": total,
                "composite_score": round(score, 4),
                "output_type": output_type,
            },
        )

    return on_eval_pair


def _make_progress_writer(migration_id: int, engine: Any, migrator: Any = None) -> Any:
    """Return a callback that writes stage progress to the DB and emits SSE events.

    Args:
        migration_id: ID of the MigrationRecord to update.
        engine: SQLAlchemy engine.
        migrator: Optional Migrator instance. When provided, live cost and warning
            counts from its PipelineContext are included in the SSE event payload.
    """
    import time as _time

    _start = _time.monotonic()

    def _write_progress(stage: str, stage_pct: float, overall_pct: float) -> None:
        # Write to DB
        try:
            with Session(engine) as sess:
                record = sess.get(MigrationRecord, migration_id)
                if record is not None:
                    record.current_stage = stage
                    record.stage_progress = stage_pct
                    record.overall_progress = overall_pct
                    sess.add(record)
                    sess.commit()
        except Exception as exc:
            logger.warning("Failed to write stage progress for migration %d: %s", migration_id, exc)

        # Emit to SSE clients
        from rosettastone.server.progress import emit_progress

        event_data: dict[str, Any] = {
            "type": "progress",
            "migration_id": migration_id,
            "current_stage": stage,
            "stage_progress": stage_pct,
            "overall_progress": overall_pct,
        }

        # Include live cost and warning data if PipelineContext is available
        if migrator is not None:
            ctx = migrator.context
            if ctx is not None:
                try:
                    event_data["total_cost_usd"] = round(sum(ctx.costs.values()), 4)
                    event_data["warning_count"] = len(ctx.warnings)
                except Exception:
                    pass  # Never let cost/warning enrichment block progress events

        # Add ETA to SSE event using linear extrapolation
        if overall_pct > 0.05:
            elapsed = _time.monotonic() - _start
            estimated_total = elapsed / overall_pct
            eta_seconds = max(0, estimated_total - elapsed)
            event_data["eta_seconds"] = round(eta_seconds, 1)

        emit_progress(migration_id, event_data)

    return _write_progress


def run_migration_background(
    migration_id: int,
    config_dict: dict[str, Any],
    engine: Any = None,
) -> None:
    """Run a migration in a background thread.

    Args:
        migration_id: ID of the MigrationRecord to update.
        config_dict: Serialized MigrationConfig fields.
        engine: SQLAlchemy engine (defaults to get_engine(); pass test engine for testing).
    """
    config_dict = dict(config_dict)

    if engine is None:
        engine = get_engine()

    # Propagate the migration ID as the correlation ID for structured logging
    # in this background thread so all log lines are traceable.
    set_request_id(f"migration-{migration_id}")

    with Session(engine) as session:
        record = session.get(MigrationRecord, migration_id)
        if record is None:
            logger.error("Migration %d not found", migration_id)
            return

        record.status = "running"
        session.add(record)
        session.commit()

    output_dir = Path.home() / ".rosettastone" / "migrations" / str(migration_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rosettastone.config import MigrationConfig  # noqa: I001
        from rosettastone.core.migrator import Migrator, MigrationBlockedError

        # Extract resume / checkpoint params (not part of MigrationConfig)
        resume_from = config_dict.pop("_resume_from", None)
        checkpoint_data = config_dict.pop("_checkpoint_data", None)

        # Build config — override output_dir to per-migration directory
        config_dict["output_dir"] = str(output_dir)
        config = MigrationConfig(**config_dict)

        gepa_cb = _make_gepa_callback(migration_id, engine)
        checkpoint_cb = _make_checkpoint_writer(migration_id, engine)
        eval_pair_cb = _make_eval_pair_callback(migration_id, engine)

        # Create migrator first so progress writer can reference it for live cost/warning data.
        # Progress callback is wired in after construction via _progress_callback attribute.
        migrator = Migrator(
            config,
            migration_id=migration_id,
            engine=engine,
            gepa_iteration_callback=gepa_cb,
            checkpoint_callback=checkpoint_cb,
            resume_checkpoint_stage=resume_from,
            resume_checkpoint_data=checkpoint_data,
            eval_pair_callback=eval_pair_cb,
        )
        migrator._progress_callback = _make_progress_writer(migration_id, engine, migrator=migrator)

        result = migrator.run()

        # Latency sampling — measure first 5 prompts against source and target
        latency_data = _sample_latency(result, config)

        # Cost projection
        cost_data = _estimate_per_call_cost(config)

        # Check for dry_run
        is_dry_run = config_dict.get("dry_run", False)

        with Session(engine) as session:
            record = session.get(MigrationRecord, migration_id)
            if record is None:
                return

            if is_dry_run:
                record.status = "dry_run_complete"
            else:
                record.status = "complete"

            record.optimized_prompt = result.optimized_prompt
            record.confidence_score = result.confidence_score
            record.baseline_score = result.baseline_score
            record.improvement = result.improvement
            record.cost_usd = result.cost_usd
            record.duration_seconds = result.duration_seconds
            record.recommendation = result.recommendation
            record.recommendation_reasoning = result.recommendation_reasoning
            record.total_tokens = result.total_tokens
            record.token_breakdown_json = json.dumps(result.token_breakdown)
            record.optimization_score_history_json = json.dumps(result.optimization_iterations)
            # Store latency and cost data
            if latency_data:
                record.source_latency_p50 = latency_data.get("source_p50")
                record.source_latency_p95 = latency_data.get("source_p95")
                record.target_latency_p50 = latency_data.get("target_p50")
                record.target_latency_p95 = latency_data.get("target_p95")
            if cost_data:
                record.projected_source_cost_per_call = cost_data.get("source_cost")
                record.projected_target_cost_per_call = cost_data.get("target_cost")

            # Embed observability fields into config_json for persistence
            # (no new DB columns needed — values are numeric/categorical only, no PII)
            config_dict_with_obs = dict(config_dict)
            config_dict_with_obs["_stage_timing"] = getattr(result, "stage_timing", {})
            config_dict_with_obs["_non_deterministic_count"] = getattr(
                result, "non_deterministic_count", 0
            )
            config_dict_with_obs["_eval_runs"] = getattr(result, "eval_runs", 1)
            record.config_json = json.dumps(config_dict_with_obs)
            record.per_type_scores_json = json.dumps(result.per_type_scores)
            record.warnings_json = json.dumps(result.warnings)
            record.safety_warnings_json = json.dumps(
                [sw if isinstance(sw, (str, dict)) else str(sw) for sw in result.safety_warnings]
            )

            store_content = config_dict.get("store_prompt_content", False)

            # Create TestCaseRecords from results
            for phase, results_list in [
                ("baseline", result.baseline_results),
                ("validation", result.validation_results),
            ]:
                for eval_result in results_list:
                    metadata = eval_result.prompt_pair.metadata or {}
                    output_type = metadata.get(
                        "output_type",
                        eval_result.prompt_pair.output_type or "short_text",
                    )

                    tc = TestCaseRecord(
                        migration_id=migration_id,
                        phase=phase,
                        output_type=str(output_type),
                        composite_score=eval_result.composite_score,
                        is_win=eval_result.is_win,
                        scores_json=json.dumps(eval_result.scores),
                        details_json=json.dumps(eval_result.details),
                        response_length=len(eval_result.prompt_pair.response),
                        new_response_length=len(eval_result.new_response),
                        evaluators_used=", ".join(eval_result.scores.keys()),
                        failure_reason=eval_result.failure_reason,
                    )

                    if store_content:
                        prompt = eval_result.prompt_pair.prompt
                        tc.prompt_text = prompt if isinstance(prompt, str) else json.dumps(prompt)
                        tc.response_text = eval_result.prompt_pair.response
                        tc.new_response_text = eval_result.new_response

                    session.add(tc)

            # Create WarningRecords
            for warning_msg in result.warnings:
                w = WarningRecord(
                    migration_id=migration_id,
                    warning_type="pipeline",
                    severity="MEDIUM",
                    message=str(warning_msg),
                )
                session.add(w)

            for sw in result.safety_warnings:
                if isinstance(sw, dict):
                    w = WarningRecord(
                        migration_id=migration_id,
                        warning_type=sw.get("warning_type", "safety"),
                        severity=sw.get("severity", "HIGH"),
                        message=sw.get("message", str(sw)),
                    )
                else:
                    w = WarningRecord(
                        migration_id=migration_id,
                        warning_type="safety",
                        severity="HIGH",
                        message=str(sw),
                    )
                session.add(w)

            session.add(record)

            # Record actual cost spend for budget tracking (multi-user mode only)
            from rosettastone.server.api.costs import record_spend
            from rosettastone.server.rbac import _is_multi_user

            if _is_multi_user() and record.owner_id is not None and result.cost_usd:
                record_spend(record.owner_id, result.cost_usd, session)

            # Auto-version and audit log (atomic with the migration record update)
            if not is_dry_run:
                try:
                    create_version(migration_id, session)
                    log_audit(session, "migration", migration_id, "complete")
                except Exception as ver_err:
                    logger.warning("Failed to create version/audit: %s", ver_err)

            session.commit()

        if not is_dry_run:
            logger.info(
                "migration_complete",
                extra={
                    "migration_id": migration_id,
                    "total_tokens": result.total_tokens,
                    "cost_usd": result.cost_usd,
                    "baseline_score": result.baseline_score,
                    "confidence_score": result.confidence_score,
                    "recommendation": result.recommendation,
                    "duration_ms": round(result.duration_seconds * 1000),
                    "stage_durations": {k: round(v * 1000) for k, v in result.stage_timing.items()},
                },
            )

        # Emit final status to SSE clients
        final_status = "dry_run_complete" if is_dry_run else "complete"
        from rosettastone.server.progress import emit_progress

        emit_progress(
            migration_id,
            {"type": "status", "status": final_status, "migration_id": migration_id},
        )

    except Exception as exc:
        # Import here to avoid circular import issues in test environments
        try:
            from rosettastone.core.migrator import MigrationBlockedError
            from rosettastone.core.types import CostLimitExceeded

            is_blocked = isinstance(exc, MigrationBlockedError)
            is_cost_exceeded = isinstance(exc, CostLimitExceeded)
        except ImportError:
            is_blocked = False
            is_cost_exceeded = False

        if is_cost_exceeded:
            error_status = "failed"
        elif is_blocked:
            error_status = "blocked"
        else:
            error_status = "failed"

        try:
            with Session(engine) as session:
                record = session.get(MigrationRecord, migration_id)
                if record is None:
                    return
                if is_blocked:
                    record.status = "blocked"
                    record.recommendation_reasoning = f"Blocked by preflight: {type(exc).__name__}"
                    logger.debug("Migration %s blocked: %s", migration_id, str(exc))
                    log_audit(session, "migration", migration_id, "blocked")
                elif is_cost_exceeded:
                    record.status = "failed"
                    record.recommendation_reasoning = str(exc)
                    logger.warning("Migration %s aborted: %s", migration_id, exc)
                    log_audit(session, "migration", migration_id, "failed")
                else:
                    record.status = "failed"
                    record.recommendation_reasoning = f"Migration failed: {exc}"
                    log_audit(session, "migration", migration_id, "failed")
                session.add(record)
                session.commit()
        except Exception as commit_err:
            logger.error(
                "Failed to update migration %d status: %s",
                migration_id,
                commit_err,
                exc_info=True,
            )

        # Emit final status to SSE clients
        from rosettastone.server.progress import emit_progress

        emit_progress(
            migration_id,
            {"type": "status", "status": error_status, "migration_id": migration_id},
        )

    finally:
        # Clean up temp uploaded data file (keep output dir for reports)
        data_path = config_dict.get("data_path")
        if data_path:
            p = Path(data_path)
            if p.exists() and str(p).startswith(str(output_dir)):
                try:
                    p.unlink()
                except OSError:
                    pass
