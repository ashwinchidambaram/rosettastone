"""Pydantic models and loader for YAML-driven pipeline configuration with DAG validation."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from pydantic import BaseModel


class PipelineModuleConfig(BaseModel):
    """Configuration for a single module in the pipeline."""

    name: str
    prompt_template: str  # System prompt for this module
    input_fields: list[str]  # Fields this module reads
    output_fields: list[str]  # Fields this module produces
    depends_on: list[str] = []  # Names of modules this depends on


class PipelineConfig(BaseModel):
    """Full pipeline configuration."""

    name: str
    modules: list[PipelineModuleConfig]
    source_model: str
    target_model: str
    data_path: str | None = None
    gepa_auto: str = "light"
    reflection_model: str = "openai/gpt-4o"


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load and validate a pipeline config from a YAML file."""
    import yaml  # type: ignore[import-untyped]

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Parse the pipeline section
    pipeline_data = raw.get("pipeline", raw)
    return PipelineConfig(**pipeline_data)


def validate_dag(config: PipelineConfig) -> list[str]:
    """Validate the pipeline DAG. Returns topologically sorted module names.

    Raises ValueError on:
    - Circular dependencies
    - References to non-existent modules in depends_on
    - Output fields referenced as inputs that don't exist in any upstream module
    """
    module_names = {m.name for m in config.modules}

    # Validate all depends_on references exist
    for module in config.modules:
        for dep in module.depends_on:
            if dep not in module_names:
                raise ValueError(
                    f"Module '{module.name}' depends on '{dep}', "
                    f"which does not exist in the pipeline."
                )

    # Build adjacency list and in-degree map from depends_on
    # Edge: dep -> module (dep must come before module)
    in_degree: dict[str, int] = {m.name: 0 for m in config.modules}
    adjacency: dict[str, list[str]] = {m.name: [] for m in config.modules}

    for module in config.modules:
        for dep in module.depends_on:
            adjacency[dep].append(module.name)
            in_degree[module.name] += 1

    # Kahn's algorithm for topological sort
    queue: deque[str] = deque(name for name, degree in in_degree.items() if degree == 0)
    sorted_modules: list[str] = []

    while queue:
        current = queue.popleft()
        sorted_modules.append(current)
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_modules) != len(config.modules):
        # Find the cycle members for a more informative error
        cycle_members = [name for name, degree in in_degree.items() if degree > 0]
        raise ValueError(f"Circular dependency detected among modules: {cycle_members}")

    # Validate input fields: each input must either be a "raw" pipeline input
    # (not produced by any module) or produced by an upstream module.
    # Build a map of module name -> set of output fields
    module_outputs: dict[str, set[str]] = {m.name: set(m.output_fields) for m in config.modules}

    # All fields ever produced by any module
    all_produced: set[str] = set()
    for m in config.modules:
        all_produced.update(m.output_fields)

    # Process modules in topological order
    module_by_name = {m.name: m for m in config.modules}
    available_fields: set[str] = set()

    for module_name in sorted_modules:
        module = module_by_name[module_name]
        upstream_outputs: set[str] = set()
        for dep in module.depends_on:
            upstream_outputs.update(module_outputs[dep])

        for field in module.input_fields:
            # A field is valid if it's not produced by any module (raw input)
            # or if it's produced by an upstream (dependency) module.
            produced_but_not_upstream = field in all_produced and field not in upstream_outputs
            if produced_but_not_upstream and field not in available_fields:
                raise ValueError(
                    f"Module '{module.name}' requires input field '{field}', "
                    f"but it is not produced by any upstream module."
                )

        # After processing this module, its outputs become available for downstream modules
        available_fields.update(module.output_fields)

    return sorted_modules
