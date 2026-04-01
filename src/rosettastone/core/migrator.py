from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from rosettastone.utils.logging import get_logger

if TYPE_CHECKING:
    from rosettastone.config import MigrationConfig
    from rosettastone.core.types import MigrationResult

logger = get_logger(__name__)


class MigrationBlockedError(Exception):
    def __init__(self, preflight_report: object) -> None:
        self.preflight_report = preflight_report
        super().__init__(f"Migration blocked by pre-flight checks: {preflight_report}")


class Migrator:
    def __init__(
        self,
        config: MigrationConfig,
        progress_callback: Callable[[str, float, float], None] | None = None,
        migration_id: int | None = None,
        engine: object | None = None,
        gepa_iteration_callback: Callable[[int, int, float], None] | None = None,
        checkpoint_callback: Callable[[str, str], None] | None = None,
        resume_checkpoint_stage: str | None = None,
        resume_checkpoint_data: str | None = None,
    ) -> None:
        self.config = config
        self._progress_callback = progress_callback
        self.migration_id = migration_id
        self.engine = engine
        self._gepa_iteration_callback = gepa_iteration_callback
        self.checkpoint_callback = checkpoint_callback
        self.resume_checkpoint_stage = resume_checkpoint_stage
        self.resume_checkpoint_data = resume_checkpoint_data

    def _emit(self, stage: str, stage_pct: float, overall_pct: float) -> None:
        """Invoke progress_callback if set; swallow any exception it raises."""
        if self._progress_callback is not None:
            try:
                self._progress_callback(stage, stage_pct, overall_pct)
            except Exception:
                pass

    def _checkpoint(self, stage: str, data: object) -> None:
        """Persist a checkpoint after a pipeline stage completes.

        Serializes ``data`` to JSON and calls checkpoint_callback if set.
        Swallows serialization and callback errors to avoid aborting the migration.
        """
        if self.checkpoint_callback is None:
            return
        import json

        try:
            data_json = json.dumps({"stage_output": data})
        except (TypeError, ValueError):
            # Fall back to stage-name-only checkpoint if data isn't serializable
            data_json = json.dumps({"stage_output": None})
        try:
            self.checkpoint_callback(stage, data_json)
        except Exception as e:
            logger.warning("Checkpoint failed for stage %s: %s", stage, type(e).__name__)

    def _persist_preflight_estimate(self, estimated_cost_usd: float) -> None:
        """Store estimated cost to the migration record in DB."""
        if self.migration_id is None or self.engine is None:
            return
        try:
            from sqlmodel import Session

            with Session(self.engine) as session:
                from rosettastone.server.models import MigrationRecord

                record = session.get(MigrationRecord, self.migration_id)
                if record is not None:
                    record.estimated_cost_usd = estimated_cost_usd
                    session.add(record)
                    session.commit()
        except Exception:
            pass  # Silently swallow DB errors — preflight estimate is non-critical

    def _update_migration_failed(self, error_message: str) -> None:
        """Mark migration as failed with error message."""
        if self.migration_id is None or self.engine is None:
            return
        try:
            from sqlmodel import Session

            with Session(self.engine) as session:
                from rosettastone.server.models import MigrationRecord

                record = session.get(MigrationRecord, self.migration_id)
                if record is not None:
                    record.status = "failed"
                    record.recommendation_reasoning = error_message
                    session.add(record)
                    session.commit()
        except Exception:
            pass  # Silently swallow DB errors

    def run(self) -> MigrationResult:
        import json

        from rosettastone.core.context import PipelineContext
        from rosettastone.core.pipeline import (
            build_result,
            evaluate_baseline,
            evaluate_optimized,
            generate_report,
            load_and_split_data,
            make_recommendation,
            optimize_prompt,
            run_pii_scan,
            run_pii_scan_text,
            run_preflight,
            run_prompt_audit,
        )
        from rosettastone.core.types import MigrationResult

        # Respect dry_run regardless of skip_preflight
        if self.config.dry_run and self.config.skip_preflight:
            return MigrationResult(
                config=self.config.model_dump(),
                optimized_prompt="",
                baseline_results=[],
                validation_results=[],
                confidence_score=0.0,
                baseline_score=0.0,
                improvement=0.0,
                cost_usd=0.0,
                duration_seconds=0.0,
                warnings=[],
                recommendation="NO_GO",
                recommendation_reasoning="Dry run — no migration performed.",
            )

        # Determine which stage to resume from (if any)
        resume_stage = self.resume_checkpoint_stage
        resume_data: dict = {}
        if resume_stage and self.resume_checkpoint_data:
            try:
                resume_data = json.loads(self.resume_checkpoint_data)
            except (json.JSONDecodeError, ValueError):
                resume_data = {}

        # Stage ordering used to decide which stages to skip on resume
        stage_order = [
            "preflight",
            "ingest",
            "baseline_eval",
            "optimize",
            "validation_eval",
        ]

        def _already_done(stage: str) -> bool:
            """Return True if this stage was completed in the saved checkpoint."""
            if not resume_stage:
                return False
            try:
                return stage_order.index(stage) <= stage_order.index(resume_stage)
            except ValueError:
                return False

        start = time.time()
        ctx = PipelineContext()

        # Step 0: Pre-flight
        if not self.config.skip_preflight:
            if not _already_done("preflight"):
                preflight_report = run_preflight(self.config)
                ctx.warnings.extend(preflight_report.warnings)
                if preflight_report.has_blockers:
                    raise MigrationBlockedError(preflight_report)
                if self.config.dry_run:
                    return preflight_report.as_dry_run_result(self.config)  # type: ignore[no-any-return]

                # Store estimated cost and check cost cap
                if self.migration_id is not None and self.engine is not None:
                    self._persist_preflight_estimate(preflight_report.estimated_cost_usd)

                # Check cost cap against estimated cost
                if self.config.max_cost_usd is not None:
                    estimated_cost = preflight_report.estimated_cost_usd
                    if estimated_cost > self.config.max_cost_usd:
                        msg = (
                            f"Estimated cost ${estimated_cost:.4f} exceeds "
                            f"max_cost_usd cap of ${self.config.max_cost_usd:.4f}"
                        )
                        if self.migration_id is not None and self.engine is not None:
                            self._update_migration_failed(msg)
                        raise ValueError(msg)

                self._checkpoint("preflight", {"warnings": preflight_report.warnings})
            self._emit("preflight", 1.0, 0.12)

        # Step 1: Ingest — always re-ingest on resume (fast and needed for data)
        t0 = time.time()
        train, val, test = load_and_split_data(self.config, ctx)
        ctx.timing["ingest"] = time.time() - t0
        if not _already_done("ingest"):
            self._checkpoint("ingest", {"pair_count": len(train) + len(val) + len(test)})
        self._emit("data_load", 1.0, 0.25)

        # Step 1.5: PII scan on ingested data
        t0 = time.time()
        run_pii_scan(train + val + test, ctx, self.config)
        ctx.timing["pii_scan"] = time.time() - t0
        self._emit("pii_scan", 1.0, 0.37)

        # Step 2: Baseline — restore from checkpoint or run
        if _already_done("baseline_eval"):
            # We don't try to reconstruct full EvalResult objects; re-run is acceptable
            # but we skip re-checkpointing since we're just passing through
            t0 = time.time()
            baseline = evaluate_baseline(test, self.config, ctx=ctx)
            ctx.timing["baseline_eval"] = time.time() - t0
        else:
            t0 = time.time()
            baseline = evaluate_baseline(test, self.config, ctx=ctx)
            ctx.timing["baseline_eval"] = time.time() - t0
            baseline_score = sum(1 for r in baseline if r.is_win) / max(len(baseline), 1)
            self._checkpoint("baseline_eval", {"baseline_score": baseline_score})
        self._emit("baseline_eval", 1.0, 0.50)

        import litellm as _litellm

        from rosettastone.optimize.gepa import GEPATimeoutWithResult

        def _make_gepa_cost_callback(
            gepa_cost_accumulator: list[float],
        ) -> Callable[[dict, object, object, object], None]:
            def _gepa_cost_callback(
                kwargs: dict,
                completion_response: object,
                start_time: object,
                end_time: object,
            ) -> None:
                cost = (
                    getattr(completion_response, "_hidden_params", {}).get("response_cost", 0.0)
                    or 0.0
                )
                gepa_cost_accumulator[0] += cost

            return _gepa_cost_callback

        # Step 3: Optimize — restore optimized prompt from checkpoint if available
        if _already_done("optimize"):
            # Try to recover the optimized prompt from checkpoint data
            saved_prompt = resume_data.get("stage_output", {})
            if isinstance(saved_prompt, dict):
                optimized_prompt = saved_prompt.get("optimized_prompt", "")
            else:
                optimized_prompt = ""
            # Fall back to re-running optimize if checkpoint didn't preserve the prompt
            if not optimized_prompt:
                _gepa_cost: list[float] = [0.0]
                _gepa_cb = _make_gepa_cost_callback(_gepa_cost)
                _litellm.success_callback.append(_gepa_cb)
                t0 = time.time()
                try:
                    optimized_prompt = optimize_prompt(
                        train, val, self.config, self._gepa_iteration_callback
                    )
                except GEPATimeoutWithResult as exc:
                    ctx.warnings.append(exc.message)
                    optimized_prompt = exc.instructions
                except TimeoutError:
                    timeout = getattr(self.config, "gepa_timeout_seconds", 600)
                    ctx.warnings.append(
                        f"GEPA timed out after {timeout}s with no usable intermediate result. "
                        f"Migration failed."
                    )
                    raise
                finally:
                    _litellm.success_callback = [
                        cb for cb in _litellm.success_callback if cb is not _gepa_cb
                    ]
                    ctx.add_cost("optimization", _gepa_cost[0])
                ctx.timing["optimize"] = time.time() - t0
            else:
                ctx.timing["optimize"] = 0.0
        else:
            _gepa_cost2: list[float] = [0.0]
            _gepa_cb2 = _make_gepa_cost_callback(_gepa_cost2)
            _litellm.success_callback.append(_gepa_cb2)
            t0 = time.time()
            try:
                optimized_prompt = optimize_prompt(
                    train, val, self.config, self._gepa_iteration_callback
                )
            except GEPATimeoutWithResult as exc:
                ctx.warnings.append(exc.message)
                optimized_prompt = exc.instructions
            except TimeoutError:
                timeout = getattr(self.config, "gepa_timeout_seconds", 600)
                ctx.warnings.append(
                    f"GEPA timed out after {timeout}s with no usable intermediate result. "
                    f"Migration failed."
                )
                raise
            finally:
                _litellm.success_callback = [
                    cb for cb in _litellm.success_callback if cb is not _gepa_cb2
                ]
                ctx.add_cost("optimization", _gepa_cost2[0])
            ctx.timing["optimize"] = time.time() - t0
            self._checkpoint("optimize", {"optimized_prompt": optimized_prompt})
        self._emit("optimize", 1.0, 0.75)

        # Step 3.5: Safety checks on optimized prompt
        t0 = time.time()
        run_pii_scan_text(optimized_prompt, ctx, self.config)
        self._emit("pii_scan_text", 1.0, 0.80)
        run_prompt_audit(optimized_prompt, train, ctx, self.config)
        ctx.timing["prompt_safety"] = time.time() - t0
        self._emit("prompt_audit", 1.0, 0.85)

        # Step 4: Validate
        if not _already_done("validation_eval"):
            t0 = time.time()
            validation = evaluate_optimized(test, optimized_prompt, self.config, ctx=ctx)
            ctx.timing["validation_eval"] = time.time() - t0
            val_score = sum(1 for r in validation if r.is_win) / max(len(validation), 1)
            self._checkpoint("validation_eval", {"validation_score": val_score})
        else:
            t0 = time.time()
            validation = evaluate_optimized(test, optimized_prompt, self.config, ctx=ctx)
            ctx.timing["validation_eval"] = time.time() - t0
        self._emit("validation_eval", 1.0, 0.95)

        # Check for GEPA regression: warn if optimized score < baseline score
        confidence_score = sum(1 for r in validation if r.is_win) / max(len(validation), 1)
        baseline_score_check = sum(1 for r in baseline if r.is_win) / max(len(baseline), 1)
        if confidence_score < baseline_score_check:
            ctx.warnings.append(
                f"GEPA optimization regressed performance: optimized score "
                f"{confidence_score:.3f} < baseline {baseline_score_check:.3f}. "
                f"Consider using the un-optimized prompt."
            )

        # Step 4.5: Recommendation
        t0 = time.time()
        rec, reasoning, per_type = make_recommendation(validation, ctx, self.config)
        ctx.recommendation = (rec, reasoning, per_type)
        ctx.timing["recommendation"] = time.time() - t0
        self._emit("recommendation", 1.0, 0.98)

        # Step 5: Report
        duration = time.time() - start
        result = build_result(self.config, optimized_prompt, baseline, validation, duration, ctx)
        generate_report(result, self.config.output_dir)
        self._emit("report", 1.0, 1.0)

        return result
