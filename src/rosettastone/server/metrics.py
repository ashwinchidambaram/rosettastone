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
