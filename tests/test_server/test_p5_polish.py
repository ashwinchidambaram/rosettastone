"""Tests for P5 polish features: error pages, security headers, mobile nav, accessibility."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

from rosettastone.server.app import create_app  # noqa: E402
from rosettastone.server.database import get_session  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    app = create_app()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Error pages — HTML for /ui/, JSON for /api/
# ---------------------------------------------------------------------------


class TestErrorPages:
    def test_ui_404_returns_branded_html(self, client: TestClient) -> None:
        resp = client.get("/ui/nonexistent-page")
        assert resp.status_code == 404
        body = resp.text
        assert "404" in body
        assert "Back to Dashboard" in body
        assert "RosettaStone" in body

    def test_api_404_returns_json(self, client: TestClient) -> None:
        resp = client.get("/api/v1/migrations/999999")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data

    def test_ui_404_is_html_content_type(self, client: TestClient) -> None:
        resp = client.get("/ui/does-not-exist")
        assert resp.status_code == 404
        assert "text/html" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    def test_x_content_type_options(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_x_xss_protection(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_referrer_policy(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_headers_on_ui_routes(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"


# ---------------------------------------------------------------------------
# Mobile nav elements
# ---------------------------------------------------------------------------


class TestMobileNav:
    def test_hamburger_button_present(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations")
        assert resp.status_code == 200
        body = resp.text
        assert 'id="mobile-menu-btn"' in body
        assert "menu" in body  # material icon name

    def test_mobile_nav_drawer_present(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations")
        body = resp.text
        assert 'id="mobile-nav-drawer"' in body
        assert 'id="mobile-nav-backdrop"' in body

    def test_mobile_nav_has_all_links(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations")
        body = resp.text
        assert 'href="/ui/"' in body
        assert 'href="/ui/migrations"' in body
        assert 'href="/ui/costs"' in body
        assert 'href="/ui/alerts"' in body


# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------


class TestAccessibility:
    def test_skip_to_content_link(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations")
        body = resp.text
        assert 'href="#main-content"' in body
        assert "Skip to content" in body

    def test_main_content_id(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations")
        body = resp.text
        assert 'id="main-content"' in body

    def test_nav_role_attribute(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations")
        body = resp.text
        assert 'role="navigation"' in body

    def test_main_role_attribute(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations")
        body = resp.text
        assert 'role="main"' in body

    def test_aria_labels_on_icon_buttons(self, client: TestClient) -> None:
        resp = client.get("/ui/migrations")
        body = resp.text
        assert 'aria-label="Toggle dark/light theme"' in body
        assert 'aria-label="Settings"' in body
        assert 'aria-label="Open navigation menu"' in body
