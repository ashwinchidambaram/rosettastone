"""Optional Prometheus metrics for RosettaStone."""

from __future__ import annotations

# Attempt to import prometheus_client — optional dependency
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# Only define metrics if prometheus_client is available
if _PROMETHEUS_AVAILABLE:
    MIGRATIONS_TOTAL = Counter(
        "migrations_total",
        "Total migrations by status",
        ["status", "source_model", "target_model"],
    )
    API_REQUESTS_TOTAL = Counter(
        "api_requests_total",
        "Total API requests",
        ["method", "path", "status_code"],
    )
    MIGRATION_DURATION_SECONDS = Histogram(
        "migration_duration_seconds",
        "Migration duration in seconds",
    )
    MIGRATION_COST_USD = Histogram(
        "migration_cost_usd",
        "Migration cost in USD",
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    )
    API_REQUEST_DURATION_SECONDS = Histogram(
        "api_request_duration_seconds",
        "API request duration in seconds",
        ["method", "path"],
    )
    MIGRATIONS_RUNNING = Gauge(
        "migrations_running",
        "Number of migrations currently running",
    )
    TASK_QUEUE_DEPTH = Gauge(
        "task_queue_depth",
        "Number of tasks in the queue with status=queued",
    )
    PIPELINE_STAGE_DURATION_SECONDS = Histogram(
        "pipeline_stage_duration_seconds",
        "Pipeline stage execution duration in seconds",
        ["stage", "status"],
        buckets=[0.5, 1, 5, 10, 30, 60, 120, 300, 600],
    )
    EVALUATOR_DURATION_SECONDS = Histogram(
        "evaluator_duration_seconds",
        "Individual evaluator execution duration in seconds",
        ["evaluator_name", "output_type"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    )
    TASK_QUEUE_WAIT_SECONDS = Histogram(
        "task_queue_wait_seconds",
        "Time a task spent waiting in the queue before being claimed",
        buckets=[0.5, 1, 5, 10, 30, 60, 120, 300, 600],
    )
    RATE_LIMIT_HITS_TOTAL = Counter(
        "rate_limit_hits_total",
        "Total rate limit rejections",
        ["endpoint", "user_id_hash"],
    )


def metrics_response() -> tuple[bytes, str]:
    """Return Prometheus metrics in text format.

    Returns:
        Tuple of (metrics_bytes, content_type_string)

    Raises:
        RuntimeError: If prometheus_client is not available.
    """
    if not _PROMETHEUS_AVAILABLE:
        raise RuntimeError("prometheus_client not available")
    return (generate_latest(REGISTRY), CONTENT_TYPE_LATEST)


def is_available() -> bool:
    """Check if prometheus_client is available.

    Returns:
        True if prometheus_client is installed, False otherwise.
    """
    return _PROMETHEUS_AVAILABLE


def record_stage_duration(stage: str, duration: float, status: str = "success") -> None:
    """Record a pipeline stage duration. No-op if prometheus_client unavailable."""
    if _PROMETHEUS_AVAILABLE:
        PIPELINE_STAGE_DURATION_SECONDS.labels(stage=stage, status=status).observe(duration)


def record_evaluator_duration(evaluator_name: str, output_type: str, duration: float) -> None:
    """Record an evaluator execution duration. No-op if prometheus_client unavailable."""
    if _PROMETHEUS_AVAILABLE:
        EVALUATOR_DURATION_SECONDS.labels(
            evaluator_name=evaluator_name, output_type=output_type
        ).observe(duration)


def record_task_queue_wait(wait_seconds: float) -> None:
    """Record how long a task waited in the queue. No-op if prometheus_client unavailable."""
    if _PROMETHEUS_AVAILABLE:
        TASK_QUEUE_WAIT_SECONDS.observe(wait_seconds)


def record_rate_limit_hit(endpoint: str, user_key: str) -> None:
    """Record a rate limit rejection. Hashes user_key before storing as label.
    No-op if prometheus_client unavailable.
    """
    if _PROMETHEUS_AVAILABLE:
        import hashlib

        user_id_hash = hashlib.sha256(user_key.encode()).hexdigest()[:12]
        RATE_LIMIT_HITS_TOTAL.labels(endpoint=endpoint, user_id_hash=user_id_hash).inc()
