"""Tests for migration endpoint IDOR vulnerability fix.

These tests verify that migration endpoints enforce ownership checks in multi-user mode:
1. In single-user mode, all migrations are accessible
2. In multi-user mode, non-admin users only see/modify their own migrations
3. In multi-user mode, admin users see/modify any migration
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from rosettastone.server.app import create_app
from rosettastone.server.database import get_session
from rosettastone.server.models import MigrationRecord, TestCaseRecord


@pytest.fixture
def engine():
    """Create a test engine with in-memory SQLite."""
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
def session(engine) -> Session:
    """Create a test session."""
    with Session(engine) as sess:
        yield sess


@pytest.fixture
def client(engine) -> TestClient:
    """Create a test client with in-memory database (single-user mode)."""
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    return TestClient(app)


_MULTI_USER_JWT_SECRET = "test-multi-user-secret-long-enough-for-hmac"


def _create_multi_user_client(
    engine, monkeypatch, user_id: int = 1, role: str = "admin"
) -> TestClient:
    """Create a multi-user test client with a specific user identity."""
    from rosettastone.server.auth_utils import create_jwt

    monkeypatch.setenv("ROSETTASTONE_MULTI_USER", "true")
    monkeypatch.setenv("ROSETTASTONE_JWT_SECRET", _MULTI_USER_JWT_SECRET)
    monkeypatch.delenv("ROSETTASTONE_API_KEY", raising=False)

    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session

    token = create_jwt(user_id=user_id, role=role, secret=_MULTI_USER_JWT_SECRET)
    return TestClient(app, headers={"Authorization": f"Bearer {token}"})


def _create_sample_migrations(session: Session) -> None:
    """Create sample migrations owned by different users."""
    # User 1's migration
    m1 = MigrationRecord(
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
        warnings_json=json.dumps([]),
        safety_warnings_json=json.dumps([]),
        owner_id=1,
    )
    session.add(m1)

    # User 2's migration
    m2 = MigrationRecord(
        source_model="openai/gpt-3.5-turbo",
        target_model="openai/gpt-4o",
        status="complete",
        confidence_score=0.88,
        baseline_score=0.80,
        improvement=0.08,
        cost_usd=0.98,
        duration_seconds=30.0,
        recommendation="GO",
        recommendation_reasoning="Good performance.",
        config_json=json.dumps(
            {"source_model": "openai/gpt-3.5-turbo", "target_model": "openai/gpt-4o"}
        ),
        per_type_scores_json=json.dumps(
            {
                "text": {
                    "win_rate": 0.88,
                    "mean": 0.87,
                    "median": 0.88,
                    "p10": 0.80,
                    "p50": 0.88,
                    "p90": 0.95,
                    "min_score": 0.75,
                    "max_score": 1.0,
                    "sample_count": 50,
                    "confidence_interval": [0.84, 0.92],
                },
            }
        ),
        warnings_json=json.dumps([]),
        safety_warnings_json=json.dumps([]),
        owner_id=2,
    )
    session.add(m2)

    # User 1's failed migration
    m3 = MigrationRecord(
        source_model="openai/gpt-3.5-turbo",
        target_model="anthropic/claude-opus",
        status="failed",
        confidence_score=None,
        baseline_score=None,
        improvement=None,
        cost_usd=0.5,
        duration_seconds=15.0,
        recommendation=None,
        recommendation_reasoning=None,
        config_json=json.dumps(
            {"source_model": "openai/gpt-3.5-turbo", "target_model": "anthropic/claude-opus"}
        ),
        per_type_scores_json=json.dumps({}),
        warnings_json=json.dumps([]),
        safety_warnings_json=json.dumps([]),
        owner_id=1,
        checkpoint_stage="validation",
    )
    session.add(m3)

    session.commit()


def _create_test_cases(session: Session, migration_id: int) -> list[TestCaseRecord]:
    """Create sample test cases for a migration."""
    cases = []
    for i in range(3):
        tc = TestCaseRecord(
            migration_id=migration_id,
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


# ---------------------------------------------------------------------------
# Single-User Mode Tests (baseline behavior)
# ---------------------------------------------------------------------------


def test_migration_detail_single_user_accessible(engine):
    """In single-user mode, any migration is accessible."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    # Fetch migration 1
    resp = client.get("/api/v1/migrations/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 1
    assert data["source_model"] == "openai/gpt-4o"


def test_migration_test_cases_single_user_accessible(engine):
    """In single-user mode, test cases are accessible."""
    with Session(engine) as session:
        _create_sample_migrations(session)
        _create_test_cases(session, 1)

    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    resp = client.get("/api/v1/migrations/1/test-cases")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 3


# ---------------------------------------------------------------------------
# Multi-User Mode Tests: Non-Admin User Access Control
# ---------------------------------------------------------------------------


def test_migration_detail_owner_can_access(engine, monkeypatch):
    """Multi-user: owner can access their own migration."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 1


def test_migration_detail_owner_forbidden(engine, monkeypatch):
    """Multi-user: non-owner gets 403 when accessing other user's migration."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    # User 1 (viewer role) tries to access User 2's migration (id=2)
    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/2")
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Access denied"


def test_migration_test_cases_owner_can_access(engine, monkeypatch):
    """Multi-user: owner can list test cases for their migration."""
    with Session(engine) as session:
        _create_sample_migrations(session)
        _create_test_cases(session, 1)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/1/test-cases")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 3


def test_migration_test_cases_owner_forbidden(engine, monkeypatch):
    """Multi-user: non-owner gets 403 when accessing other user's test cases."""
    with Session(engine) as session:
        _create_sample_migrations(session)
        _create_test_cases(session, 2)

    # User 1 tries to access User 2's test cases
    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/2/test-cases")
    assert resp.status_code == 403


def test_migration_test_case_detail_owner_can_access(engine, monkeypatch):
    """Multi-user: owner can get a specific test case."""
    with Session(engine) as session:
        _create_sample_migrations(session)
        test_cases = _create_test_cases(session, 1)
        tc_id = test_cases[0].id

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get(f"/api/v1/migrations/1/test-cases/{tc_id}")
    assert resp.status_code == 200


def test_migration_test_case_detail_owner_forbidden(engine, monkeypatch):
    """Multi-user: non-owner gets 403 when accessing other user's test case."""
    with Session(engine) as session:
        _create_sample_migrations(session)
        test_cases = _create_test_cases(session, 2)
        tc_id = test_cases[0].id

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get(f"/api/v1/migrations/2/test-cases/{tc_id}")
    assert resp.status_code == 403


def test_migration_regressions_owner_can_access(engine, monkeypatch):
    """Multi-user: owner can get regressions for their migration."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/1/regressions")
    assert resp.status_code == 200
    data = resp.json()
    assert "prompt_regressions" in data


def test_migration_regressions_owner_forbidden(engine, monkeypatch):
    """Multi-user: non-owner gets 403 when accessing other user's regressions."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/2/regressions")
    assert resp.status_code == 403


def test_migration_optimization_trace_owner_can_access(engine, monkeypatch):
    """Multi-user: owner can get optimization trace."""
    with Session(engine) as session:
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
            config_json=json.dumps({"source_model": "openai/gpt-4o"}),
            per_type_scores_json=json.dumps({}),
            warnings_json=json.dumps([]),
            safety_warnings_json=json.dumps([]),
            owner_id=1,
            optimization_score_history_json=json.dumps(
                [
                    {"iteration_num": 0, "mean_score": 0.85},
                    {"iteration_num": 1, "mean_score": 0.87},
                ]
            ),
        )
        session.add(migration)
        session.commit()

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/1/optimization-trace")
    assert resp.status_code == 200
    data = resp.json()
    assert "iterations" in data


def test_migration_optimization_trace_owner_forbidden(engine, monkeypatch):
    """Multi-user: non-owner gets 403 when accessing other user's optimization trace."""
    with Session(engine) as session:
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
            config_json=json.dumps({"source_model": "openai/gpt-4o"}),
            per_type_scores_json=json.dumps({}),
            warnings_json=json.dumps([]),
            safety_warnings_json=json.dumps([]),
            owner_id=2,
            optimization_score_history_json=json.dumps(
                [
                    {"iteration_num": 0, "mean_score": 0.85},
                    {"iteration_num": 1, "mean_score": 0.87},
                ]
            ),
        )
        session.add(migration)
        session.commit()

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/1/optimization-trace")
    assert resp.status_code == 403


def test_migration_diagnostics_owner_can_access(engine, monkeypatch):
    """Multi-user: owner can get diagnostics."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/1/diagnostics")
    assert resp.status_code == 200
    data = resp.json()
    assert "migration_id" in data


def test_migration_diagnostics_owner_forbidden(engine, monkeypatch):
    """Multi-user: non-owner gets 403 when accessing other user's diagnostics."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/2/diagnostics")
    assert resp.status_code == 403


def test_migration_stream_owner_can_access(engine, monkeypatch):
    """Multi-user: owner can stream migration progress."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/1/stream")
    # SSE endpoint returns 200 with streaming response
    assert resp.status_code == 200


def test_migration_stream_owner_forbidden(engine, monkeypatch):
    """Multi-user: non-owner gets 403 when streaming other user's migration."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/api/v1/migrations/2/stream")
    assert resp.status_code == 403


def test_migration_resume_owner_can_access(engine, monkeypatch):
    """Multi-user: owner can resume their own migration."""
    from unittest.mock import MagicMock

    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="editor")
    # Mock task_worker to avoid KeyError
    client.app.state.task_worker = MagicMock()

    resp = client.post("/api/v1/migrations/3/resume")
    # Migration 3 belongs to user 1 and is failed with checkpoint
    assert resp.status_code == 200


def test_migration_resume_owner_forbidden(engine, monkeypatch):
    """Multi-user: non-owner gets 403 when resuming other user's migration."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    # User 1 tries to resume a migration in failed state but with owner_id=1
    # Let's create a failed migration for user 2
    with Session(engine) as session:
        m_failed = MigrationRecord(
            source_model="openai/gpt-3.5-turbo",
            target_model="anthropic/claude-opus",
            status="failed",
            confidence_score=None,
            baseline_score=None,
            improvement=None,
            cost_usd=0.5,
            duration_seconds=15.0,
            recommendation=None,
            recommendation_reasoning=None,
            config_json=json.dumps({}),
            per_type_scores_json=json.dumps({}),
            warnings_json=json.dumps([]),
            safety_warnings_json=json.dumps([]),
            owner_id=2,
            checkpoint_stage="validation",
        )
        session.add(m_failed)
        session.commit()

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="editor")

    resp = client.post("/api/v1/migrations/4/resume")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Multi-User Mode Tests: Admin Bypass
# ---------------------------------------------------------------------------


def test_migration_detail_admin_can_access_any(engine, monkeypatch):
    """Multi-user: admin can access any migration."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=3, role="admin")

    # Admin accesses User 2's migration
    resp = client.get("/api/v1/migrations/2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 2


def test_migration_test_cases_admin_can_access_any(engine, monkeypatch):
    """Multi-user: admin can access test cases for any migration."""
    with Session(engine) as session:
        _create_sample_migrations(session)
        _create_test_cases(session, 2)

    client = _create_multi_user_client(engine, monkeypatch, user_id=3, role="admin")

    # Admin accesses User 2's test cases
    resp = client.get("/api/v1/migrations/2/test-cases")
    assert resp.status_code == 200


def test_migration_diagnostics_admin_can_access_any(engine, monkeypatch):
    """Multi-user: admin can access diagnostics for any migration."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=3, role="admin")

    # Admin accesses User 2's diagnostics
    resp = client.get("/api/v1/migrations/2/diagnostics")
    assert resp.status_code == 200


def test_migration_resume_admin_can_access_any(engine, monkeypatch):
    """Multi-user: admin can resume any migration."""
    from unittest.mock import MagicMock

    with Session(engine) as session:
        m_failed = MigrationRecord(
            source_model="openai/gpt-3.5-turbo",
            target_model="anthropic/claude-opus",
            status="failed",
            confidence_score=None,
            baseline_score=None,
            improvement=None,
            cost_usd=0.5,
            duration_seconds=15.0,
            recommendation=None,
            recommendation_reasoning=None,
            config_json=json.dumps({}),
            per_type_scores_json=json.dumps({}),
            warnings_json=json.dumps([]),
            safety_warnings_json=json.dumps([]),
            owner_id=2,  # User 2's migration
            checkpoint_stage="validation",
        )
        session.add(m_failed)
        session.commit()

    client = _create_multi_user_client(engine, monkeypatch, user_id=3, role="admin")
    # Mock task_worker to avoid KeyError
    client.app.state.task_worker = MagicMock()

    # Admin resumes User 2's migration
    resp = client.post("/api/v1/migrations/1/resume")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Multi-User Mode Tests: UI Endpoints
# ---------------------------------------------------------------------------


def test_migration_optimization_trace_fragment_owner_forbidden(engine, monkeypatch):
    """Multi-user: UI fragment endpoint enforces ownership."""
    with Session(engine) as session:
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
            config_json=json.dumps({"source_model": "openai/gpt-4o"}),
            per_type_scores_json=json.dumps({}),
            warnings_json=json.dumps([]),
            safety_warnings_json=json.dumps([]),
            owner_id=2,
            optimization_score_history_json=json.dumps(
                [
                    {"iteration_num": 0, "mean_score": 0.85},
                ]
            ),
        )
        session.add(migration)
        session.commit()

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/ui/migrations/1/optimization-trace-fragment")
    assert resp.status_code == 403


def test_migration_diagnostics_fragment_owner_forbidden(engine, monkeypatch):
    """Multi-user: UI diagnostics fragment enforces ownership."""
    with Session(engine) as session:
        _create_sample_migrations(session)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/ui/migrations/2/diagnostics-fragment")
    assert resp.status_code == 403


def test_migration_test_cases_table_owner_forbidden(engine, monkeypatch):
    """Multi-user: UI test cases table enforces ownership."""
    with Session(engine) as session:
        _create_sample_migrations(session)
        _create_test_cases(session, 2)

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/ui/migrations/2/test-cases-table")
    assert resp.status_code == 403


def test_migration_optimizer_history_fragment_owner_forbidden(engine, monkeypatch):
    """Multi-user: UI optimizer history fragment enforces ownership."""
    with Session(engine) as session:
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
            config_json=json.dumps({"source_model": "openai/gpt-4o"}),
            per_type_scores_json=json.dumps({}),
            warnings_json=json.dumps([]),
            safety_warnings_json=json.dumps([]),
            owner_id=2,
        )
        session.add(migration)
        session.commit()

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="viewer")

    resp = client.get("/ui/migrations/1/optimizer-history")
    assert resp.status_code == 403


def test_migration_ui_resume_owner_forbidden(engine, monkeypatch):
    """Multi-user: UI resume endpoint enforces ownership."""
    from unittest.mock import MagicMock

    with Session(engine) as session:
        m_failed = MigrationRecord(
            source_model="openai/gpt-3.5-turbo",
            target_model="anthropic/claude-opus",
            status="failed",
            confidence_score=None,
            baseline_score=None,
            improvement=None,
            cost_usd=0.5,
            duration_seconds=15.0,
            recommendation=None,
            recommendation_reasoning=None,
            config_json=json.dumps({}),
            per_type_scores_json=json.dumps({}),
            warnings_json=json.dumps([]),
            safety_warnings_json=json.dumps([]),
            owner_id=2,
            checkpoint_stage="validation",
        )
        session.add(m_failed)
        session.commit()

    client = _create_multi_user_client(engine, monkeypatch, user_id=1, role="editor")
    # Mock task_worker to avoid KeyError
    client.app.state.task_worker = MagicMock()

    resp = client.post("/ui/migrations/1/resume")
    assert resp.status_code == 403
