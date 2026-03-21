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


class TestPdfReport:
    def test_pdf_report_501(self, client, engine, sample_migration):
        """PDF should return 501 when weasyprint is not installed."""
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/pdf")
        assert response.status_code == 501

    def test_pdf_report_404(self, client):
        response = client.get("/api/v1/migrations/999/report/pdf")
        assert response.status_code == 404


class TestHtmlReport:
    def test_html_report_501(self, client, engine, sample_migration):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/html")
        assert response.status_code == 501

    def test_html_report_404(self, client):
        response = client.get("/api/v1/migrations/999/report/html")
        assert response.status_code == 404


class TestExecutiveReport:
    def test_executive_report_501(self, client, engine, sample_migration):
        response = client.get(f"/api/v1/migrations/{sample_migration.id}/report/executive")
        assert response.status_code == 501

    def test_executive_report_404(self, client):
        response = client.get("/api/v1/migrations/999/report/executive")
        assert response.status_code == 404
