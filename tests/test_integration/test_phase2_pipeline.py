"""Integration tests for Phase 2 pipeline wiring.

These tests verify that the pipeline correctly routes to:
- Redis vs JSONL data sources
- GEPA vs MIPROv2 optimizers
- PII scanning and prompt auditing
- Recommendation engine
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from rosettastone.config import MigrationConfig
from rosettastone.core.context import PipelineContext, SafetySeverity
from rosettastone.core.types import EvalResult, PromptPair

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **overrides) -> MigrationConfig:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        "\n".join(
            json.dumps({"prompt": f"q{i}", "response": f"a{i}", "source_model": "openai/gpt-4o"})
            for i in range(5)
        )
    )
    defaults = {
        "source_model": "openai/gpt-4o",
        "target_model": "anthropic/claude-sonnet-4",
        "data_path": data_file,
        "skip_preflight": True,
    }
    defaults.update(overrides)
    return MigrationConfig(**defaults)


def _make_eval_result(
    output_type: str = "classification", score: float = 0.95, is_win: bool = True
) -> EvalResult:
    return EvalResult(
        prompt_pair=PromptPair(prompt="test", response="test", source_model="openai/gpt-4o"),
        new_response="test",
        scores={"exact_match": score},
        composite_score=score,
        is_win=is_win,
        details={"output_type": output_type},
    )


# ---------------------------------------------------------------------------
# Pipeline routing tests
# ---------------------------------------------------------------------------


class TestDataSourceRouting:
    def test_redis_url_selects_redis_adapter(self, tmp_path):
        """When config.redis_url is set, RedisAdapter is used."""
        from rosettastone.core.pipeline import load_and_split_data

        config = _make_config(tmp_path, redis_url="redis://localhost:6379")
        with patch("rosettastone.ingest.redis_adapter.RedisAdapter") as mock_redis:
            mock_redis.return_value.load.return_value = [
                PromptPair(prompt="q", response="a", source_model="openai/gpt-4o")
                for _ in range(20)
            ]
            load_and_split_data(config)
            mock_redis.assert_called_once_with("redis://localhost:6379", "openai/gpt-4o")

    def test_no_redis_url_selects_jsonl_adapter(self, tmp_path):
        """When config.redis_url is None, JSONLAdapter is used."""
        from rosettastone.core.pipeline import load_and_split_data

        config = _make_config(tmp_path)
        train, val, test = load_and_split_data(config)
        assert len(train) + len(val) + len(test) == 5

    def test_cluster_prompts_deduplicates_to_one_representative_per_cluster(self, tmp_path):
        """When cluster_prompts=True, pairs is replaced with one representative per cluster plus noise."""
        from unittest.mock import MagicMock

        from rosettastone.cluster.types import ClusterResult, PromptCluster
        from rosettastone.core.pipeline import load_and_split_data

        pairs = [
            PromptPair(prompt=f"q{i}", response=f"a{i}", source_model="openai/gpt-4o")
            for i in range(6)
        ]

        noise_pair = PromptPair(prompt="noise", response="noise_resp", source_model="openai/gpt-4o")

        cluster_result = ClusterResult(
            clusters=[
                PromptCluster(cluster_id=0, label="group-a", pairs=pairs[0:3]),
                PromptCluster(cluster_id=1, label="group-b", pairs=pairs[3:6]),
            ],
            noise_pairs=[noise_pair],
            n_clusters=2,
            silhouette_score=0.75,
        )

        mock_clusterer = MagicMock()
        mock_clusterer.cluster.return_value = cluster_result

        config = _make_config(tmp_path, cluster_prompts=True)
        with patch("rosettastone.cluster.embedder.PromptClusterer", return_value=mock_clusterer):
            train, val, test = load_and_split_data(config)

        all_pairs = train + val + test
        # 2 clusters → 2 representatives + 1 noise = 3 total
        assert len(all_pairs) == 3
        # Representatives are the first pair from each cluster
        assert pairs[0] in all_pairs
        assert pairs[3] in all_pairs
        assert noise_pair in all_pairs
        # Non-representative cluster members must NOT be present
        for dropped in pairs[1:3] + pairs[4:6]:
            assert dropped not in all_pairs

    def test_cluster_prompts_import_error_falls_back_gracefully(self, tmp_path):
        """When clustering deps are missing, load_and_split_data proceeds without clustering."""
        import sys

        from rosettastone.core.pipeline import load_and_split_data

        config = _make_config(tmp_path, cluster_prompts=True)
        # Remove the embedder module so the lazy import inside load_and_split_data raises ImportError
        saved = sys.modules.pop("rosettastone.cluster.embedder", None)
        try:
            import builtins

            real_import = builtins.__import__

            def _block_embedder(name, *args, **kwargs):
                if name == "rosettastone.cluster.embedder":
                    raise ImportError("no module")
                return real_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=_block_embedder):
                train, val, test = load_and_split_data(config)
        finally:
            if saved is not None:
                sys.modules["rosettastone.cluster.embedder"] = saved

        # Should still return all 5 original pairs unmodified
        assert len(train) + len(val) + len(test) == 5


class TestOptimizerRouting:
    def test_mipro_auto_selects_mipro_optimizer(self, tmp_path):
        """When config.mipro_auto is set, MIPROv2Optimizer is used."""
        from rosettastone.core.pipeline import optimize_prompt

        config = _make_config(tmp_path, mipro_auto="light")
        pairs = [
            PromptPair(prompt="q", response="a", source_model="openai/gpt-4o") for _ in range(3)
        ]
        with patch("rosettastone.optimize.mipro.MIPROv2Optimizer") as mock_mipro:
            mock_mipro.return_value.optimize.return_value = "optimized"
            result = optimize_prompt(pairs, pairs, config)
            mock_mipro.assert_called_once()
            assert result == "optimized"

    def test_no_mipro_auto_selects_gepa_optimizer(self, tmp_path):
        """When config.mipro_auto is None, GEPAOptimizer is used."""
        from rosettastone.core.pipeline import optimize_prompt

        config = _make_config(tmp_path)
        pairs = [
            PromptPair(prompt="q", response="a", source_model="openai/gpt-4o") for _ in range(3)
        ]
        with patch("rosettastone.optimize.gepa.GEPAOptimizer") as mock_gepa:
            mock_gepa.return_value.optimize.return_value = "optimized"
            result = optimize_prompt(pairs, pairs, config)
            mock_gepa.assert_called_once()
            assert result == "optimized"


# ---------------------------------------------------------------------------
# Safety pipeline tests
# ---------------------------------------------------------------------------


class TestPIIScanPipeline:
    def test_pii_scan_adds_warnings_to_context(self):
        """PII findings from scan_pairs are added to context.safety_warnings."""
        from rosettastone.core.pipeline import run_pii_scan

        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=Path("/tmp/fake.jsonl"),
        )
        pairs = [
            PromptPair(
                prompt="Contact john@example.com",
                response="SSN: 123-45-6789",
                source_model="openai/gpt-4o",
            )
        ]
        ctx = PipelineContext()
        run_pii_scan(pairs, ctx, config)
        assert len(ctx.safety_warnings) >= 1
        types = {w.warning_type for w in ctx.safety_warnings}
        assert "pii" in types

    def test_pii_scan_disabled_when_config_says_no(self):
        """When config.pii_scan=False, no warnings are added."""
        from rosettastone.core.pipeline import run_pii_scan

        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=Path("/tmp/fake.jsonl"),
            pii_scan=False,
        )
        pairs = [
            PromptPair(
                prompt="SSN: 123-45-6789",
                response="ok",
                source_model="openai/gpt-4o",
            )
        ]
        ctx = PipelineContext()
        run_pii_scan(pairs, ctx, config)
        assert len(ctx.safety_warnings) == 0


class TestPromptAuditPipeline:
    def test_prompt_audit_detects_leakage(self):
        """Prompt audit finds training data in optimized prompt."""
        from rosettastone.core.pipeline import run_prompt_audit

        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=Path("/tmp/fake.jsonl"),
        )
        # Response must be at least 31 chars for the auditor's substring check
        leaked_response = (
            "This is a very specific and unique training response"
            " that contains enough characters to trigger the auditor"
        )
        # Need 15+ pairs so the boilerplate threshold (10%) is > 1.
        # Only 1 pair has the leaked text, so its count=1 < threshold=1.5 → not boilerplate.
        pairs = [PromptPair(prompt="q", response=leaked_response, source_model="openai/gpt-4o")] + [
            PromptPair(prompt=f"q{i}", response=f"generic answer {i}", source_model="openai/gpt-4o")
            for i in range(14)
        ]
        # The optimized prompt contains the training response verbatim
        optimized = f"Instructions: {leaked_response} end of instructions"
        ctx = PipelineContext()
        run_prompt_audit(optimized, pairs, ctx, config)
        assert len(ctx.safety_warnings) >= 1

    def test_prompt_audit_disabled(self):
        """When config.prompt_audit=False, no audit is performed."""
        from rosettastone.core.pipeline import run_prompt_audit

        config = MigrationConfig(
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-sonnet-4",
            data_path=Path("/tmp/fake.jsonl"),
            prompt_audit=False,
        )
        ctx = PipelineContext()
        run_prompt_audit("anything", [], ctx, config)
        assert len(ctx.safety_warnings) == 0


class TestPIIScanText:
    def test_high_severity_pii_in_prompt_creates_blocker(self):
        """HIGH-severity PII in optimized prompt text creates a warning."""
        from rosettastone.core.pipeline import run_pii_scan_text

        ctx = PipelineContext()
        run_pii_scan_text("Contact at 123-45-6789 for details", ctx)
        high_warnings = [w for w in ctx.safety_warnings if w.severity == SafetySeverity.HIGH]
        assert len(high_warnings) >= 1


# ---------------------------------------------------------------------------
# Recommendation pipeline tests
# ---------------------------------------------------------------------------


class TestRecommendationPipeline:
    def test_go_recommendation(self, tmp_path):
        """All types meeting thresholds produces GO (requires MIN_RELIABLE_SAMPLES=30).

        Uses long_text output type (threshold=0.75) so that the Wilson CI lower bound
        for 30/30 wins (~0.887) exceeds the threshold and GO is returned.
        """
        from rosettastone.core.pipeline import make_recommendation

        config = _make_config(tmp_path)
        results = [_make_eval_result(output_type="long_text", score=0.95) for _ in range(30)]
        ctx = PipelineContext()
        rec, reasoning, per_type = make_recommendation(results, ctx, config)
        assert rec == "GO"

    def test_no_go_with_high_safety(self, tmp_path):
        """HIGH-severity safety warning produces NO_GO."""
        from rosettastone.core.context import SafetyWarning
        from rosettastone.core.pipeline import make_recommendation

        config = _make_config(tmp_path)
        results = [_make_eval_result(score=0.95) for _ in range(15)]
        ctx = PipelineContext()
        ctx.safety_warnings.append(
            SafetyWarning(
                warning_type="pii",
                severity=SafetySeverity.HIGH,
                message="SSN found",
            )
        )
        rec, reasoning, _ = make_recommendation(results, ctx, config)
        assert rec == "NO_GO"

    def test_conditional_with_low_win_rate(self, tmp_path):
        """Output type below threshold produces CONDITIONAL."""
        from rosettastone.core.pipeline import make_recommendation

        config = _make_config(tmp_path)
        results = [_make_eval_result(score=0.5, is_win=False) for _ in range(15)]
        ctx = PipelineContext()
        rec, reasoning, _ = make_recommendation(results, ctx, config)
        assert rec in ("CONDITIONAL", "NO_GO")


# ---------------------------------------------------------------------------
# build_result integration
# ---------------------------------------------------------------------------


class TestBuildResult:
    def test_build_result_with_context(self, tmp_path):
        """build_result populates safety_warnings from context."""
        from rosettastone.core.context import SafetyWarning
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        ctx = PipelineContext()
        ctx.safety_warnings.append(
            SafetyWarning(
                warning_type="pii",
                severity=SafetySeverity.MEDIUM,
                message="Email found",
            )
        )
        ctx.costs["optimize"] = 1.50

        result = build_result(config, "optimized", [], [], 10.0, ctx)
        assert len(result.safety_warnings) == 1
        assert result.cost_usd == 1.50

    def test_build_result_without_context(self, tmp_path):
        """build_result works without context for backward compatibility."""
        from rosettastone.core.pipeline import build_result

        config = _make_config(tmp_path)
        result = build_result(config, "optimized", [], [], 10.0)
        assert result.safety_warnings == []
        assert result.cost_usd == 0.0
