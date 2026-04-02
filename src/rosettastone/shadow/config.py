"""Shadow deployment configuration model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RollbackConfig(BaseModel):
    enabled: bool = True
    error_rate_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    latency_p99_ms: int = 5000


class EndpointsConfig(BaseModel):
    source: str = "http://localhost:8001/v1/chat/completions"
    target: str = "http://localhost:8002/v1/chat/completions"


class ShadowConfig(BaseModel):
    source_model: str
    target_model: str
    optimized_prompt: str = ""
    primary: Literal["source", "target"] = "source"
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    duration_hours: int = Field(default=72, ge=1)
    log_path: str = "./shadow_logs/"
    rollback: RollbackConfig = Field(default_factory=RollbackConfig)
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
