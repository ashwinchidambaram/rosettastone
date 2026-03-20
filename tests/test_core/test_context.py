"""Tests for PipelineContext and supporting types."""

from rosettastone.core.context import (
    PipelineContext,
    SafetySeverity,
    SafetyWarning,
    TypeStats,
)
from rosettastone.core.types import OutputType


class TestSafetyWarning:
    def test_creation(self):
        w = SafetyWarning(
            warning_type="pii",
            severity=SafetySeverity.HIGH,
            message="SSN detected",
        )
        assert w.warning_type == "pii"
        assert w.severity == SafetySeverity.HIGH
        assert w.message == "SSN detected"
        assert w.details == {}

    def test_with_details(self):
        w = SafetyWarning(
            warning_type="pii",
            severity=SafetySeverity.MEDIUM,
            message="Email found",
            details={"count": 3},
        )
        assert w.details == {"count": 3}


class TestTypeStats:
    def test_defaults(self):
        ts = TypeStats()
        assert ts.win_rate == 0.0
        assert ts.sample_count == 0
        assert ts.confidence_interval == (0.0, 0.0)

    def test_custom_values(self):
        ts = TypeStats(win_rate=0.85, sample_count=50, p50=0.82, confidence_interval=(0.72, 0.93))
        assert ts.win_rate == 0.85
        assert ts.sample_count == 50
        assert ts.p50 == 0.82


class TestPipelineContext:
    def test_empty_defaults(self):
        ctx = PipelineContext()
        assert ctx.warnings == []
        assert ctx.safety_warnings == []
        assert ctx.costs == {}
        assert ctx.timing == {}
        assert ctx.per_type_stats == {}

    def test_accumulate_warnings(self):
        ctx = PipelineContext()
        ctx.warnings.append("Low sample count")
        ctx.safety_warnings.append(
            SafetyWarning(warning_type="pii", severity=SafetySeverity.HIGH, message="SSN found")
        )
        assert len(ctx.warnings) == 1
        assert len(ctx.safety_warnings) == 1

    def test_accumulate_costs(self):
        ctx = PipelineContext()
        ctx.costs["baseline_eval"] = 0.25
        ctx.costs["optimization"] = 1.50
        assert ctx.costs["baseline_eval"] == 0.25
        assert sum(ctx.costs.values()) == 1.75

    def test_accumulate_timing(self):
        ctx = PipelineContext()
        ctx.timing["ingest"] = 0.5
        ctx.timing["optimize"] = 12.3
        assert ctx.timing["ingest"] == 0.5

    def test_per_type_stats(self):
        ctx = PipelineContext()
        ctx.per_type_stats[OutputType.JSON] = TypeStats(win_rate=0.95, sample_count=20)
        ctx.per_type_stats[OutputType.SHORT_TEXT] = TypeStats(win_rate=0.82, sample_count=30)
        assert ctx.per_type_stats[OutputType.JSON].win_rate == 0.95
        assert len(ctx.per_type_stats) == 2

    def test_independent_instances(self):
        """Each PipelineContext should have independent mutable defaults."""
        ctx1 = PipelineContext()
        ctx2 = PipelineContext()
        ctx1.warnings.append("only in ctx1")
        assert ctx2.warnings == []
