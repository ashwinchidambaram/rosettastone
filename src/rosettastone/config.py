from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EvalStrategy(StrEnum):
    AUTO = "auto"
    BERTSCORE = "bertscore"
    EMBEDDING = "embedding"
    EXACT = "exact"
    JSON = "json"


class OptimizerChoice(StrEnum):
    GEPA = "gepa"
    MIPRO = "mipro"  # Phase 2


class AdapterChoice(StrEnum):
    JSONL = "jsonl"
    REDIS = "redis"
    CSV = "csv"
    BRAINTRUST = "braintrust"
    LANGSMITH = "langsmith"
    OTEL = "otel"


class PIIEngine(StrEnum):
    REGEX = "regex"
    PRESIDIO = "presidio"


class MigrationConfig(BaseModel):
    # Required
    source_model: str
    target_model: str
    data_path: Path | None = None

    # Optional with smart defaults
    output_dir: Path = Path("./migration_output")
    eval_strategy: EvalStrategy = EvalStrategy.AUTO
    optimizer: OptimizerChoice = OptimizerChoice.GEPA
    train_split: float = Field(default=0.2, ge=0.1, le=0.5)
    val_split: float = Field(default=0.8, ge=0.5, le=0.9)
    min_pairs: int = 20
    recommended_pairs: int = 100

    # GEPA configuration
    gepa_auto: Literal["light", "medium", "heavy"] = "light"
    reflection_model: str = "openai/gpt-4o"
    num_threads: int = 4

    # Pre-flight
    dry_run: bool = False
    skip_preflight: bool = False
    max_context_usage: float = 0.75

    # Phase 2: Data sources
    local_only: bool = False
    redis_url: str | None = None

    # Phase 2: LLM-as-judge
    judge_model: str = "openai/gpt-4o"

    # Phase 2: Safety
    pii_scan: bool = True
    prompt_audit: bool = True

    # Phase 2: MIPROv2 optimizer
    mipro_auto: Literal["light", "medium", "heavy"] | None = None

    # Phase 2: Decision thresholds (per output type)
    win_thresholds: dict[str, float] = {
        "json": 0.95,
        "classification": 0.90,
        "short_text": 0.80,
        "long_text": 0.75,
    }

    # Phase 3: Web UI content storage
    store_prompt_content: bool = False

    # Phase 4: Data adapter selection
    adapter: AdapterChoice = AdapterChoice.JSONL

    # Phase 4: Adapter-specific options
    csv_delimiter: str | None = None
    csv_prompt_column: str | None = None
    csv_response_column: str | None = None
    braintrust_project: str | None = None
    langsmith_project: str | None = None
    langsmith_start_date: str | None = None
    langsmith_end_date: str | None = None
    otel_path: Path | None = None

    # Phase 4: PII engine selection
    pii_engine: PIIEngine = PIIEngine.REGEX

    # Phase 4: Clustering
    cluster_prompts: bool = False

    # Phase 4: Improvement mode
    improvement_objectives: list[dict[str, str | float]] | None = None

    # Phase 5: Known-issue weighting
    known_issue_weight: float = Field(
        default=2.0, gt=0.0, description="Score divisor applied to known-issue pairs in GEPA metric"
    )

    # Cost guardrail
    max_cost_usd: float | None = Field(
        default=None,
        description="Max migration cost in USD; aborts if preflight estimate exceeds this",
    )
