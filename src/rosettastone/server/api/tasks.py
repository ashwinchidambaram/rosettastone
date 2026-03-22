"""Background task runner for migrations triggered from the web UI."""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

from sqlmodel import Session

from rosettastone.server.database import get_engine
from rosettastone.server.models import MigrationRecord, TestCaseRecord, WarningRecord

logger = logging.getLogger(__name__)


def _sample_latency(result, config) -> dict | None:
    """Sample latency by calling source and target models with a few prompts."""
    try:
        import litellm

        # Take up to 5 prompts from validation results
        prompts = []
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


def _estimate_per_call_cost(config) -> dict | None:
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


def run_migration_background(
    migration_id: int,
    config_dict: dict,
    engine=None,
) -> None:
    """Run a migration in a background thread.

    Args:
        migration_id: ID of the MigrationRecord to update.
        config_dict: Serialized MigrationConfig fields.
        engine: SQLAlchemy engine (defaults to get_engine(); pass test engine for testing).
    """
    if engine is None:
        engine = get_engine()

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

        # Build config — override output_dir to per-migration directory
        config_dict["output_dir"] = str(output_dir)
        config = MigrationConfig(**config_dict)

        result = Migrator(config).run()

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
            # Store latency and cost data
            if latency_data:
                record.source_latency_p50 = latency_data.get("source_p50")
                record.source_latency_p95 = latency_data.get("source_p95")
                record.target_latency_p50 = latency_data.get("target_p50")
                record.target_latency_p95 = latency_data.get("target_p95")
            if cost_data:
                record.projected_source_cost_per_call = cost_data.get("source_cost")
                record.projected_target_cost_per_call = cost_data.get("target_cost")

            record.config_json = json.dumps(config_dict)
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
                    )

                    if store_content:
                        prompt = eval_result.prompt_pair.prompt
                        tc.prompt_text = (
                            prompt if isinstance(prompt, str) else json.dumps(prompt)
                        )
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
            session.commit()

    except Exception as exc:
        # Import here to avoid circular import issues in test environments
        try:
            from rosettastone.core.migrator import MigrationBlockedError
            is_blocked = isinstance(exc, MigrationBlockedError)
        except ImportError:
            is_blocked = False

        try:
            with Session(engine) as session:
                record = session.get(MigrationRecord, migration_id)
                if record is None:
                    return
                if is_blocked:
                    record.status = "blocked"
                    record.recommendation_reasoning = str(exc)
                else:
                    record.status = "failed"
                    record.recommendation_reasoning = f"Migration failed: {exc}"
                session.add(record)
                session.commit()
        except Exception as commit_err:
            logger.error("Failed to update migration %d status: %s", migration_id, commit_err)

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
