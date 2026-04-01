"""Tests for pipeline_config.py — YAML loading and DAG validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rosettastone.optimize.pipeline_config import (
    PipelineConfig,
    PipelineModuleConfig,
    load_pipeline_config,
    validate_dag,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_YAML = textwrap.dedent("""\
    pipeline:
      name: test_pipeline
      source_model: openai/gpt-4o
      target_model: anthropic/claude-haiku-4-5
      modules:
        - name: step1
          prompt_template: "You are a helpful assistant."
          input_fields: [prompt]
          output_fields: [result]
          depends_on: []
        - name: step2
          prompt_template: "Refine the result."
          input_fields: [result]
          output_fields: [final_answer]
          depends_on: [step1]
""")


def _make_module(
    name: str,
    input_fields: list[str] | None = None,
    output_fields: list[str] | None = None,
    depends_on: list[str] | None = None,
) -> PipelineModuleConfig:
    return PipelineModuleConfig(
        name=name,
        prompt_template=f"Template for {name}",
        input_fields=input_fields or ["prompt"],
        output_fields=output_fields or [f"{name}_out"],
        depends_on=depends_on or [],
    )


def _make_config(modules: list[PipelineModuleConfig]) -> PipelineConfig:
    return PipelineConfig(
        name="test",
        modules=modules,
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-haiku-4-5",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadPipelineConfig:
    def test_load_pipeline_config_from_yaml(self, tmp_path: Path) -> None:
        """Parse a YAML file and verify all fields are correctly loaded."""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(_SIMPLE_YAML)

        config = load_pipeline_config(yaml_file)

        assert config.name == "test_pipeline"
        assert config.source_model == "openai/gpt-4o"
        assert config.target_model == "anthropic/claude-haiku-4-5"
        assert len(config.modules) == 2
        assert config.modules[0].name == "step1"
        assert config.modules[1].name == "step2"
        assert config.modules[0].input_fields == ["prompt"]
        assert config.modules[1].depends_on == ["step1"]

    def test_load_pipeline_config_top_level_yaml(self, tmp_path: Path) -> None:
        """YAML without a 'pipeline' key should also parse (top-level dict)."""
        yaml_content = textwrap.dedent("""\
            name: bare_pipeline
            source_model: openai/gpt-4o
            target_model: anthropic/claude-haiku-4-5
            modules:
              - name: mod1
                prompt_template: "Hello"
                input_fields: [x]
                output_fields: [y]
        """)
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml_content)

        config = load_pipeline_config(yaml_file)
        assert config.name == "bare_pipeline"
        assert len(config.modules) == 1


class TestValidateDag:
    def test_validate_dag_simple_linear(self) -> None:
        """3-module linear chain returns correct topological order."""
        modules = [
            _make_module("a", input_fields=["prompt"], output_fields=["a_out"], depends_on=[]),
            _make_module("b", input_fields=["a_out"], output_fields=["b_out"], depends_on=["a"]),
            _make_module("c", input_fields=["b_out"], output_fields=["c_out"], depends_on=["b"]),
        ]
        config = _make_config(modules)
        order = validate_dag(config)

        assert order == ["a", "b", "c"]

    def test_validate_dag_parallel_branches(self) -> None:
        """Two modules depending on the same root — root comes first."""
        modules = [
            _make_module(
                "root", input_fields=["prompt"], output_fields=["root_out"], depends_on=[]
            ),
            _make_module(
                "branch1",
                input_fields=["root_out"],
                output_fields=["b1_out"],
                depends_on=["root"],
            ),
            _make_module(
                "branch2",
                input_fields=["root_out"],
                output_fields=["b2_out"],
                depends_on=["root"],
            ),
        ]
        config = _make_config(modules)
        order = validate_dag(config)

        assert order[0] == "root"
        assert set(order[1:]) == {"branch1", "branch2"}

    def test_validate_dag_detects_cycle(self) -> None:
        """A→B→C→A circular dependency raises ValueError with 'Circular'."""
        modules = [
            _make_module("a", output_fields=["a_out"], depends_on=["c"]),
            _make_module("b", output_fields=["b_out"], depends_on=["a"]),
            _make_module("c", output_fields=["c_out"], depends_on=["b"]),
        ]
        config = _make_config(modules)

        with pytest.raises(ValueError, match="[Cc]ircular"):
            validate_dag(config)

    def test_validate_dag_missing_dependency(self) -> None:
        """Module depends on nonexistent module raises ValueError."""
        modules = [
            _make_module("a", depends_on=["nonexistent"]),
        ]
        config = _make_config(modules)

        with pytest.raises(ValueError, match="nonexistent"):
            validate_dag(config)

    def test_validate_dag_field_not_produced_by_upstream(self) -> None:
        """Module consumes a field from a non-upstream sibling raises ValueError.

        Topology: root → branch1, root → branch2
        branch1 depends only on root but requests branch2_out (produced by branch2,
        which is NOT in branch1's depends_on chain). This must raise.
        """
        modules = [
            _make_module("root", input_fields=["prompt"], output_fields=["root_out"]),
            _make_module(
                "branch1",
                input_fields=["branch2_out"],  # branch2 is NOT a dependency
                output_fields=["b1_out"],
                depends_on=["root"],
            ),
            _make_module(
                "branch2",
                input_fields=["root_out"],
                output_fields=["branch2_out"],
                depends_on=["root"],
            ),
        ]
        config = _make_config(modules)

        with pytest.raises(ValueError, match="branch2_out"):
            validate_dag(config)

    def test_pipeline_config_validation_empty_modules(self) -> None:
        """Empty modules list — PipelineConfig parses successfully."""
        config = PipelineConfig(
            name="empty_pipeline",
            modules=[],
            source_model="openai/gpt-4o",
            target_model="anthropic/claude-haiku-4-5",
        )
        # validate_dag on empty config returns empty list
        order = validate_dag(config)
        assert order == []

    def test_pipeline_module_defaults(self) -> None:
        """`depends_on` defaults to [] when not specified."""
        module = PipelineModuleConfig(
            name="mod",
            prompt_template="template",
            input_fields=["in"],
            output_fields=["out"],
        )
        assert module.depends_on == []

    def test_validate_dag_single_module(self) -> None:
        """Single module with no dependencies returns [module_name]."""
        modules = [_make_module("only", input_fields=["prompt"], output_fields=["result"])]
        config = _make_config(modules)

        order = validate_dag(config)
        assert order == ["only"]
