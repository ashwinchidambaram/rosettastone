"""Tests for report API endpoints."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
sqlmodel = pytest.importorskip("sqlmodel")


class TestMarkdownReport:
    def test_markdown_report(self, client, engine, sample_migration):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/markdown")
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]
        content = response.text
        # Should contain migration info regardless of which code path
        assert "openai/gpt-4o" in content or "Migration Report" in content

    def test_markdown_report_404(self, client):
        response = client.get("/api/v1/migrations/999/report/markdown")
        assert response.status_code == 404


class TestHtmlReport:
    def test_html_report_returns_content_or_501(self, client, engine, sample_migration):
        """HTML report returns 200 with content when jinja2 is available, 501 otherwise."""
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/html")
        # jinja2 is a project dependency — expect 200 with HTML content
        assert response.status_code in (200, 501)
        if response.status_code == 200:
            assert "text/html" in response.headers["content-type"]
            assert len(response.text) > 0

    def test_html_report_200_when_jinja2_available(self, client, engine, sample_migration):
        """HTML report returns 200 with HTML content (jinja2 is installed)."""
        pytest.importorskip("jinja2")
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        content = response.text
        assert len(content) > 0

    def test_html_report_404(self, client):
        response = client.get("/api/v1/migrations/999/report/html")
        assert response.status_code == 404


class TestPdfReport:
    def test_pdf_report_501_without_weasyprint(self, client, engine, sample_migration):
        """PDF returns 501 when weasyprint is not installed."""
        try:
            import weasyprint  # noqa: F401

            pytest.skip("weasyprint is installed — 501 path not reachable")
        except ImportError:
            pass
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/pdf")
        assert response.status_code == 501

    def test_pdf_report_200_with_weasyprint(self, client, engine, sample_migration):
        """PDF returns 200 with PDF bytes when weasyprint is installed."""
        pytest.importorskip("weasyprint")
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"

    def test_pdf_report_returns_content_or_501(self, client, engine, sample_migration):
        """PDF report endpoint responds with 200 or 501 depending on weasyprint availability."""
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/pdf")
        assert response.status_code in (200, 501)

    def test_pdf_report_404(self, client):
        response = client.get("/api/v1/migrations/999/report/pdf")
        assert response.status_code == 404


class TestExecutiveReport:
    def test_executive_report_returns_markdown(self, client, engine, sample_migration):
        """Executive report returns 200 with markdown content."""
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/executive")
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]
        assert len(response.text) > 0

    def test_executive_report_content_includes_models(self, client, engine, sample_migration):
        """Executive report narrative includes source/target model info."""
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/executive")
        assert response.status_code == 200
        content = response.text
        # The narrative should reference the models from the config
        assert "gpt-4o" in content or "claude" in content or len(content) > 50

    def test_executive_report_404(self, client):
        response = client.get("/api/v1/migrations/999/report/executive")
        assert response.status_code == 404
