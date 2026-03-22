"""Batch migration runner — execute multiple migrations from a YAML manifest."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BatchEntry(BaseModel):
    name: str
    source_model: str
    target_model: str
    data_path: str
    gepa_auto: str = "light"
    dry_run: bool = False
    output_dir: str | None = None
    redis_url: str | None = None
    reflection_model: str | None = None
    judge_model: str | None = None


class BatchManifest(BaseModel):
    version: int = 1
    defaults: dict[str, object] = Field(default_factory=dict)
    migrations: list[BatchEntry]


@dataclass
class BatchResult:
    name: str
    source_model: str
    target_model: str
    status: str  # "complete", "blocked", "failed"
    recommendation: str | None = None
    confidence: float | None = None
    error: str | None = None


def load_manifest(path: Path) -> BatchManifest:
    """Load and validate a batch YAML manifest, applying defaults to each entry."""
    import yaml

    raw = yaml.safe_load(path.read_text())
    manifest = BatchManifest.model_validate(raw)

    # Apply defaults to each migration entry where the field still holds its default value
    entry_defaults = BatchEntry.model_fields
    for entry in manifest.migrations:
        for key, value in manifest.defaults.items():
            if key not in entry_defaults:
                continue
            field_default = entry_defaults[key].default
            current = getattr(entry, key, field_default)
            if current == field_default:
                object.__setattr__(entry, key, value)

    return manifest


def run_batch(manifest: BatchManifest, output_base: Path) -> list[BatchResult]:
    """Execute each migration in the manifest sequentially and return results."""
    from rosettastone.config import MigrationConfig
    from rosettastone.core.migrator import MigrationBlockedError, Migrator

    results: list[BatchResult] = []

    for entry in manifest.migrations:
        logger.info(
            "Starting migration: %s (%s -> %s)",
            entry.name, entry.source_model, entry.target_model,
        )

        # Sanitize the name for use as a directory component
        sanitized_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in entry.name)
        if entry.output_dir:
            output_dir = str(entry.output_dir)
        else:
            output_dir = str(output_base / sanitized_name)

        config_dict: dict[str, object] = {
            "source_model": entry.source_model,
            "target_model": entry.target_model,
            "data_path": entry.data_path,
            "gepa_auto": entry.gepa_auto,
            "dry_run": entry.dry_run,
            "output_dir": output_dir,
        }
        if entry.redis_url:
            config_dict["redis_url"] = entry.redis_url
        if entry.reflection_model:
            config_dict["reflection_model"] = entry.reflection_model
        if entry.judge_model:
            config_dict["judge_model"] = entry.judge_model

        try:
            config = MigrationConfig(**config_dict)  # type: ignore[arg-type]
            result = Migrator(config).run()
            results.append(
                BatchResult(
                    name=entry.name,
                    source_model=entry.source_model,
                    target_model=entry.target_model,
                    status="complete",
                    recommendation=result.recommendation,
                    confidence=result.confidence_score,
                )
            )
            logger.info(
                "Completed: %s (confidence=%.2f)", entry.name, result.confidence_score,
            )
        except MigrationBlockedError as exc:
            logger.warning("Migration blocked: %s — %s", entry.name, exc)
            results.append(
                BatchResult(
                    name=entry.name,
                    source_model=entry.source_model,
                    target_model=entry.target_model,
                    status="blocked",
                    error=str(exc),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Migration failed: %s — %s", entry.name, exc)
            results.append(
                BatchResult(
                    name=entry.name,
                    source_model=entry.source_model,
                    target_model=entry.target_model,
                    status="failed",
                    error=str(exc),
                )
            )

    return results


def format_batch_summary(results: list[BatchResult]) -> str:
    """Return a markdown-formatted summary table of batch migration results."""
    lines: list[str] = []

    header = "| Name | Source → Target | Status | Recommendation | Confidence |"
    separator = "|------|-----------------|--------|----------------|------------|"
    lines.append(header)
    lines.append(separator)

    for r in results:
        route = f"{r.source_model} → {r.target_model}"
        recommendation = r.recommendation or "—"
        confidence = f"{r.confidence:.0%}" if r.confidence is not None else "—"
        status_display = r.status.upper()
        if r.error:
            err_msg = f"{r.error[:40]}..." if len(r.error) > 40 else r.error
            status_display = f"{r.status.upper()} ({err_msg})"
        lines.append(f"| {r.name} | {route} | {status_display} | {recommendation} | {confidence} |")

    # Aggregate footer counts
    go_count = sum(1 for r in results if r.recommendation == "GO")
    conditional_count = sum(1 for r in results if r.recommendation == "CONDITIONAL")
    no_go_count = sum(1 for r in results if r.recommendation == "NO_GO")
    blocked_count = sum(1 for r in results if r.status == "blocked")
    failed_count = sum(1 for r in results if r.status == "failed")

    footer_parts: list[str] = []
    if go_count:
        footer_parts.append(f"{go_count} GO")
    if conditional_count:
        footer_parts.append(f"{conditional_count} CONDITIONAL")
    if no_go_count:
        footer_parts.append(f"{no_go_count} NO_GO")
    if blocked_count:
        footer_parts.append(f"{blocked_count} blocked")
    if failed_count:
        footer_parts.append(f"{failed_count} failed")

    lines.append("")
    lines.append(", ".join(footer_parts) if footer_parts else "No results.")

    return "\n".join(lines)
