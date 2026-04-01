"""Test fixtures for server tests."""

from __future__ import annotations

import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.app import create_app
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord, TestCaseRecord


@pytest.fixture
def engine():
    """Create a test engine.

    When DATABASE_URL is set (e.g. in PostgreSQL CI), uses that URL.
    Otherwise falls back to an in-memory SQLite engine with StaticPool.
    """
    import os

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        eng = create_engine(database_url, echo=False)
    else:
        eng = create_engine(
            "sqlite://",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    SQLModel.metadata.create_all(eng)
    yield eng
    SQLModel.metadata.drop_all(eng)


@pytest.fixture
def session(engine) -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


@pytest.fixture
def sample_migration(session: Session) -> MigrationRecord:
    """Insert a sample migration record."""
    migration = MigrationRecord(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        status="complete",
        confidence_score=0.92,
        baseline_score=0.85,
        improvement=0.07,
        cost_usd=1.23,
        duration_seconds=45.6,
        recommendation="GO",
        recommendation_reasoning="All types pass thresholds.",
        config_json=json.dumps(
            {"source_model": "openai/gpt-4o", "target_model": "anthropic/claude-sonnet-4"}
        ),
        per_type_scores_json=json.dumps(
            {
                "json": {
                    "win_rate": 0.95,
                    "mean": 0.93,
                    "median": 0.94,
                    "p10": 0.88,
                    "p50": 0.94,
                    "p90": 0.98,
                    "min_score": 0.85,
                    "max_score": 1.0,
                    "sample_count": 20,
                    "confidence_interval": [0.88, 0.99],
                },
            }
        ),
        warnings_json=json.dumps(["Low sample count for classification"]),
        safety_warnings_json=json.dumps([]),
    )
    session.add(migration)
    session.commit()
    session.refresh(migration)
    return migration


@pytest.fixture
def sample_migration_with_cluster(session: Session) -> MigrationRecord:
    """Insert a sample migration record with cluster_summary in config."""
    migration = MigrationRecord(
        source_model="openai/gpt-4o",
        target_model="anthropic/claude-sonnet-4",
        status="complete",
        confidence_score=0.92,
        baseline_score=0.85,
        improvement=0.07,
        cost_usd=1.23,
        duration_seconds=45.6,
        recommendation="GO",
        recommendation_reasoning="All types pass thresholds.",
        config_json=json.dumps(
            {
                "source_model": "openai/gpt-4o",
                "target_model": "anthropic/claude-sonnet-4",
                "cluster_prompts": True,
                "cluster_summary": {
                    "n_clusters": 5,
                    "silhouette_score": 0.72,
                    "original_pairs": 100,
                    "representative_pairs": 25,
                },
            }
        ),
        per_type_scores_json=json.dumps(
            {
                "json": {
                    "win_rate": 0.95,
                    "mean": 0.93,
                    "median": 0.94,
                    "p10": 0.88,
                    "p50": 0.94,
                    "p90": 0.98,
                    "min_score": 0.85,
                    "max_score": 1.0,
                    "sample_count": 20,
                    "confidence_interval": [0.88, 0.99],
                },
            }
        ),
        warnings_json=json.dumps(["Low sample count for classification"]),
        safety_warnings_json=json.dumps([]),
    )
    session.add(migration)
    session.commit()
    session.refresh(migration)
    return migration


@pytest.fixture
def sample_test_cases(session: Session, sample_migration: MigrationRecord) -> list[TestCaseRecord]:
    """Insert sample test case records."""
    cases = []
    for i in range(5):
        tc = TestCaseRecord(
            migration_id=sample_migration.id,
            phase="validation",
            output_type="json",
            composite_score=0.85 + i * 0.03,
            is_win=True,
            scores_json=json.dumps({"bertscore": 0.9, "exact_match": 0.8 + i * 0.03}),
            details_json=json.dumps({"output_type": "json"}),
            response_length=100 + i * 10,
            new_response_length=95 + i * 10,
        )
        session.add(tc)
        cases.append(tc)
    session.commit()
    for tc in cases:
        session.refresh(tc)
    return cases


@pytest.fixture
def client(engine) -> TestClient:
    """Create a test client with in-memory database."""
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    return TestClient(app)
