"""Tests for cost aggregation API endpoints and UI integration."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session  # noqa: E402

from rosettastone.server.api.costs import _compute_costs, _generate_opportunities  # noqa: E402
from rosettastone.server.models import MigrationRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_migration(
    session: Session,
    source: str = "openai/gpt-4o",
    target: str = "anthropic/claude-sonnet-4",
    cost: float = 1.00,
    status: str = "complete",
    recommendation: str | None = None,
    confidence_score: float | None = None,
) -> MigrationRecord:
    record = MigrationRecord(
        source_model=source,
        target_model=target,
        status=status,
        cost_usd=cost,
        recommendation=recommendation,
        confidence_score=confidence_score,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


# ---------------------------------------------------------------------------
# _compute_costs unit tests
# ---------------------------------------------------------------------------


class TestComputeCosts:
    def test_returns_none_when_no_migrations(self, session: Session) -> None:
        result = _compute_costs(session)
        assert result is None

    def test_returns_dict_with_migrations(self, session: Session) -> None:
        _insert_migration(session, cost=2.00)
        result = _compute_costs(session)
        assert result is not None
        assert "total_month" in result
        assert "potential_savings" in result
        assert "after_optimization" in result
        assert "by_model" in result
        assert "opportunities" in result

    def test_total_aggregates_all_costs(self, session: Session) -> None:
        _insert_migration(session, target="openai/gpt-4o-mini", cost=1.00)
        _insert_migration(session, target="openai/gpt-4o-mini", cost=2.00)
        result = _compute_costs(session)
        assert result is not None
        assert result["total_month"] == "$3.00"

    def test_cost_attributed_to_target_model(self, session: Session) -> None:
        _insert_migration(session, source="openai/gpt-4o", target="anthropic/claude-sonnet-4", cost=5.00)
        result = _compute_costs(session)
        assert result is not None
        by_model = result["by_model"]
        model_names = [item["model"] for item in by_model]
        assert "claude-sonnet-4" in model_names
        # Source model should NOT appear in by_model
        assert "gpt-4o" not in model_names

    def test_by_model_sorted_by_cost_descending(self, session: Session) -> None:
        _insert_migration(session, target="anthropic/claude-sonnet-4", cost=1.00)
        _insert_migration(session, target="openai/gpt-4o", cost=5.00)
        result = _compute_costs(session)
        assert result is not None
        by_model = result["by_model"]
        assert len(by_model) == 2
        assert by_model[0]["model"] == "gpt-4o"
        assert by_model[1]["model"] == "claude-sonnet-4"

    def test_by_model_pct_sums_near_100(self, session: Session) -> None:
        _insert_migration(session, target="openai/gpt-4o", cost=3.00)
        _insert_migration(session, target="anthropic/claude-sonnet-4", cost=1.00)
        result = _compute_costs(session)
        assert result is not None
        total_pct = sum(item["pct"] for item in result["by_model"])
        # Rounding may cause slight deviation
        assert abs(total_pct - 100) <= 2

    def test_zero_cost_migrations_no_division_by_zero(self, session: Session) -> None:
        _insert_migration(session, cost=0.0)
        result = _compute_costs(session)
        assert result is not None
        assert result["total_month"] == "$0.00"
        assert result["by_model"][0]["pct"] == 0

    def test_potential_savings_is_25_percent(self, session: Session) -> None:
        _insert_migration(session, cost=4.00)
        result = _compute_costs(session)
        assert result is not None
        # 25% of $4.00 = $1.00
        assert result["potential_savings"] == "$1.00"

    def test_after_optimization_equals_total_minus_savings(self, session: Session) -> None:
        _insert_migration(session, cost=8.00)
        result = _compute_costs(session)
        assert result is not None
        # 8.00 - 2.00 (25%) = 6.00
        assert result["after_optimization"] == "$6.00"

    def test_cost_formatted_with_comma_for_thousands(self, session: Session) -> None:
        _insert_migration(session, cost=1500.00)
        result = _compute_costs(session)
        assert result is not None
        assert "$1,500.00" == result["total_month"]

    def test_multiple_migrations_same_target_aggregated(self, session: Session) -> None:
        for _ in range(3):
            _insert_migration(session, target="openai/gpt-4o", cost=2.00)
        result = _compute_costs(session)
        assert result is not None
        assert len(result["by_model"]) == 1
        assert result["by_model"][0]["cost"] == "$6.00"

    def test_model_short_name_used_when_slash_present(self, session: Session) -> None:
        _insert_migration(session, target="openai/gpt-4o-mini", cost=1.00)
        result = _compute_costs(session)
        assert result is not None
        model_names = [item["model"] for item in result["by_model"]]
        assert "gpt-4o-mini" in model_names
        assert "openai/gpt-4o-mini" not in model_names

    def test_model_id_without_slash_used_as_is(self, session: Session) -> None:
        record = MigrationRecord(
            source_model="gpt-4",
            target_model="gpt-4-turbo",
            status="complete",
            cost_usd=1.00,
        )
        session.add(record)
        session.commit()
        result = _compute_costs(session)
        assert result is not None
        model_names = [item["model"] for item in result["by_model"]]
        assert "gpt-4-turbo" in model_names


# ---------------------------------------------------------------------------
# _generate_opportunities unit tests
# ---------------------------------------------------------------------------


class TestGenerateOpportunities:
    def test_go_migration_generates_opportunity(self, session: Session) -> None:
        _insert_migration(
            session,
            source="openai/gpt-4o",
            target="anthropic/claude-sonnet-4",
            cost=10.00,
            status="complete",
            recommendation="GO",
            confidence_score=0.95,
        )
        model_costs = {"gpt-4o": 5.00}
        opps = _generate_opportunities(model_costs, session)
        assert len(opps) >= 1
        assert "claude-sonnet-4" in opps[0]["title"]
        assert "95% parity" == opps[0]["confidence"]

    def test_no_go_migration_does_not_generate_opportunity(self, session: Session) -> None:
        _insert_migration(
            session,
            source="openai/gpt-4o",
            target="anthropic/claude-sonnet-4",
            recommendation="NO_GO",
            status="complete",
        )
        model_costs = {"gpt-4o": 5.00}
        opps = _generate_opportunities(model_costs, session)
        # Should fall through to generic opportunity since source not in model_costs result
        # or produce generic if model_costs is present
        # Either way there should be at most 1 generic opportunity
        assert len(opps) <= 1

    def test_fallback_generic_opportunity_when_no_go_migrations(self, session: Session) -> None:
        _insert_migration(session, recommendation=None, status="pending")
        model_costs = {"gpt-4o": 10.00}
        opps = _generate_opportunities(model_costs, session)
        assert len(opps) == 1
        assert "gpt-4o" in opps[0]["title"]
        assert "Requires evaluation" == opps[0]["confidence"]

    def test_opportunities_capped_at_3(self, session: Session) -> None:
        for i in range(6):
            _insert_migration(
                session,
                source=f"openai/model-{i}",
                target=f"anthropic/target-{i}",
                recommendation="GO",
                status="complete",
                confidence_score=0.90,
            )
        # Give all source models a cost entry so they all qualify
        model_costs = {f"model-{i}": 5.00 for i in range(6)}
        opps = _generate_opportunities(model_costs, session)
        assert len(opps) <= 3

    def test_empty_model_costs_returns_empty_list(self, session: Session) -> None:
        opps = _generate_opportunities({}, session)
        assert opps == []


# ---------------------------------------------------------------------------
# GET /api/v1/costs
# ---------------------------------------------------------------------------


class TestGetCostsEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/costs")
        assert resp.status_code == 200

    def test_no_migrations_returns_empty_defaults(self, client: TestClient) -> None:
        resp = client.get("/api/v1/costs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_month"] == "$0.00"
        assert data["potential_savings"] == "$0.00"
        assert data["after_optimization"] == "$0.00"
        assert data["by_model"] == []
        assert data["opportunities"] == []

    def test_with_migrations_returns_aggregated_data(
        self, client: TestClient, engine
    ) -> None:
        with Session(engine) as session:
            _insert_migration(session, target="openai/gpt-4o", cost=3.00)
            _insert_migration(session, target="anthropic/claude-sonnet-4", cost=1.00)

        resp = client.get("/api/v1/costs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_month"] == "$4.00"
        assert data["potential_savings"] == "$1.00"
        assert data["after_optimization"] == "$3.00"
        assert len(data["by_model"]) == 2

    def test_response_shape(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _insert_migration(session, cost=1.00)

        resp = client.get("/api/v1/costs")
        data = resp.json()
        assert "total_month" in data
        assert "potential_savings" in data
        assert "after_optimization" in data
        assert "by_model" in data
        assert "opportunities" in data

    def test_by_model_item_shape(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _insert_migration(session, target="openai/gpt-4o", cost=2.00)

        resp = client.get("/api/v1/costs")
        data = resp.json()
        item = data["by_model"][0]
        assert "model" in item
        assert "cost" in item
        assert "pct" in item


# ---------------------------------------------------------------------------
# GET /api/v1/costs/by-model
# ---------------------------------------------------------------------------


class TestGetCostsByModelEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/costs/by-model")
        assert resp.status_code == 200

    def test_no_migrations_returns_empty_list(self, client: TestClient) -> None:
        resp = client.get("/api/v1/costs/by-model")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_migrations_returns_breakdown(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _insert_migration(session, target="openai/gpt-4o", cost=3.00)
            _insert_migration(session, target="anthropic/claude-sonnet-4", cost=1.00)

        resp = client.get("/api/v1/costs/by-model")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        model_names = [item["model"] for item in data]
        assert "gpt-4o" in model_names
        assert "claude-sonnet-4" in model_names

    def test_sorted_by_cost_descending(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _insert_migration(session, target="anthropic/claude-sonnet-4", cost=1.00)
            _insert_migration(session, target="openai/gpt-4o", cost=5.00)

        resp = client.get("/api/v1/costs/by-model")
        data = resp.json()
        assert data[0]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# UI /ui/costs — integration tests
# ---------------------------------------------------------------------------


class TestCostsUIPage:
    def test_costs_page_returns_200(self, client: TestClient) -> None:
        resp = client.get("/ui/costs")
        assert resp.status_code == 200

    def test_costs_page_falls_back_to_dummy_when_no_data(self, client: TestClient) -> None:
        """With no migrations, the page should render dummy data."""
        resp = client.get("/ui/costs")
        assert resp.status_code == 200
        # Dummy data total is $1,247
        assert "$1,247" in resp.text

    def test_costs_page_shows_real_total_when_data_exists(
        self, client: TestClient, engine
    ) -> None:
        with Session(engine) as session:
            _insert_migration(session, target="openai/gpt-4o", cost=10.00)

        resp = client.get("/ui/costs")
        assert resp.status_code == 200
        assert "$10.00" in resp.text
        # Dummy total should NOT appear
        assert "$1,247" not in resp.text

    def test_costs_page_shows_model_breakdown(self, client: TestClient, engine) -> None:
        with Session(engine) as session:
            _insert_migration(session, target="openai/gpt-4o", cost=7.50)
            _insert_migration(session, target="anthropic/claude-sonnet-4", cost=2.50)

        resp = client.get("/ui/costs")
        assert resp.status_code == 200
        assert "gpt-4o" in resp.text
        assert "claude-sonnet-4" in resp.text

    def test_costs_page_shows_opportunities_section(
        self, client: TestClient, engine
    ) -> None:
        with Session(engine) as session:
            _insert_migration(
                session,
                source="openai/gpt-4o",
                target="anthropic/claude-sonnet-4",
                cost=5.00,
                status="complete",
                recommendation="GO",
                confidence_score=0.90,
            )

        resp = client.get("/ui/costs")
        assert resp.status_code == 200
        assert "Optimization opportunities" in resp.text

    def test_costs_page_active_nav_is_costs(self, client: TestClient) -> None:
        resp = client.get("/ui/costs")
        assert resp.status_code == 200
        # The active nav item should be highlighted for "costs"
        # The base template marks the active nav, so "costs" should appear
        assert "costs" in resp.text.lower()

    def test_costs_page_with_zero_cost_migrations(
        self, client: TestClient, engine
    ) -> None:
        """Migrations with zero cost should not crash (no division by zero)."""
        with Session(engine) as session:
            _insert_migration(session, cost=0.0)

        resp = client.get("/ui/costs")
        assert resp.status_code == 200
        assert "$0.00" in resp.text
