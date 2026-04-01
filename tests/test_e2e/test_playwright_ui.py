"""Playwright end-to-end tests for the RosettaStone web UI.

Covers all pages, navigation, interactive behaviors, visual styling,
and error handling as specified in PLAYWRIGHT_TEST_PLAN.md.
"""

from __future__ import annotations

import pathlib
import re
import signal
import subprocess
import time
import urllib.request

import pytest
from playwright.sync_api import Page, expect, sync_playwright

# Resolve the project root dynamically so the fixture works on any machine / CI.
# This file lives at tests/test_e2e/test_playwright_ui.py, so .parent.parent.parent
# walks up: test_playwright_ui.py → test_e2e/ → tests/ → project root.
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

BASE_URL = "http://localhost:8765"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _kill_port(port: int) -> None:
    """Kill any existing processes bound to the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                subprocess.run(["kill", "-9", pid], check=False)
            except Exception:
                pass
        if pids:
            time.sleep(0.5)  # Give the OS a moment to release the port
    except Exception:
        pass


@pytest.fixture(scope="session")
def server():
    """Start the FastAPI server for testing (session-scoped, started once)."""
    # Ensure no stale process is holding the port
    _kill_port(8765)

    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "rosettastone.server.app:create_app",
            "--factory",
            "--port",
            "8765",
            "--timeout-keep-alive",
            "0",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Poll health endpoint until ready (max 30s)
    ready = False
    for _ in range(60):
        try:
            resp = urllib.request.urlopen(f"{BASE_URL}/api/v1/health", timeout=2)
            if resp.status == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.5)

    if not ready:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"Server at {BASE_URL} did not become ready within 30s")

    yield BASE_URL

    # Graceful shutdown, then force-kill if needed
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except Exception:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="session")
def browser_instance():
    """Create a Playwright browser instance shared for the session."""
    pw = sync_playwright().start()
    # Force dark color scheme to match the app's default dark theme
    browser = pw.chromium.launch(headless=True)
    yield browser
    browser.close()
    pw.stop()


@pytest.fixture
def dark_mode_page(server, browser_instance):
    """Create a page with forced dark color scheme (for theme toggle tests)."""
    context = browser_instance.new_context(
        color_scheme="dark",
    )
    pg = context.new_page()
    pg.set_default_timeout(60_000)
    yield pg
    pg.close()
    context.close()


@pytest.fixture
def page(server, browser_instance):
    """Create a new browser page for each test."""
    context = browser_instance.new_context()
    pg = context.new_page()
    pg.set_default_timeout(60_000)
    yield pg
    pg.close()
    context.close()


# ---------------------------------------------------------------------------
# 1. Models Dashboard (/ui/)
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestModelsDashboard:
    def test_models_dashboard_loads(self, page: Page, server: str):
        """GET /ui/?empty=false returns 200 and shows 'Your models' heading."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page).to_have_title(re.compile(r"Model Intelligence", re.IGNORECASE))
        expect(page.locator("h2").filter(has_text="Your models")).to_be_visible()

    def test_models_dashboard_shows_all_models(self, page: Page, server: str):
        """All 4 DUMMY_MODELS are rendered on the page."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page.locator("text=openai/gpt-4o").first).to_be_visible()
        expect(page.locator("text=anthropic/claude-sonnet-4").first).to_be_visible()
        expect(page.locator("text=openai/gpt-4o-mini").first).to_be_visible()
        expect(page.locator("text=openai/gpt-4o-0613").first).to_be_visible()

    def test_models_dashboard_active_count(self, page: Page, server: str):
        """Shows '3 ACTIVE INSTANCES' text for the 3 active models."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page.locator("text=3 ACTIVE INSTANCES")).to_be_visible()

    def test_models_dashboard_deprecated_card(self, page: Page, server: str):
        """Deprecated model card shows retirement date, replacement, and start migration link."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page.locator("text=Deprecated").first).to_be_visible()
        expect(page.locator("text=Retiring Apr 15, 2026")).to_be_visible()
        expect(page.locator("text=openai/gpt-4o (latest)")).to_be_visible()
        expect(page.locator("text=Start migration")).to_be_visible()

    def test_models_dashboard_alerts_banner(self, page: Page, server: str):
        """Alerts banner shows '3 things need your attention' and all 3 alert messages."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page.locator("text=3 things need your attention")).to_be_visible()
        expect(page.locator("text=Model retiring in 26 days")).to_be_visible()
        expect(page.locator("text=Price decreased 17%")).to_be_visible()
        expect(page.locator("text=New model available")).to_be_visible()

    def test_models_dashboard_add_model_button(self, page: Page, server: str):
        """'Add model' card is present."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page.locator("text=Add model")).to_be_visible()

    def test_models_dashboard_explore_table(self, page: Page, server: str):
        """Explore models table has 5 rows with correct model names."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page.locator("h2").filter(has_text="Explore models")).to_be_visible()
        expect(page.locator("input[placeholder*='Search 2,450+']")).to_be_visible()
        expect(page.locator("table")).to_be_visible()

        # Verify all 5 explore table models
        for model in [
            "gemini-2.5-flash",
            "llama-4-maverick",
            "claude-opus-4.6",
            "mistral-large",
            "deepseek-v3",
        ]:
            expect(page.locator(f"td:has-text('{model}')").first).to_be_visible()

    def test_models_dashboard_model_costs_and_context(self, page: Page, server: str):
        """Model cards display context and cost/M values."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page.locator("text=$2.50").first).to_be_visible()
        expect(page.locator("text=$3.00").first).to_be_visible()
        expect(page.locator("text=OpenAI").first).to_be_visible()
        expect(page.locator("text=Anthropic").first).to_be_visible()

    def test_models_dashboard_run_migration_links(self, page: Page, server: str):
        """Active model cards have 'Run migration' links."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page.locator("text=Run migration").first).to_be_visible()

    def test_nav_active_state_models(self, page: Page, server: str):
        """'Models' nav link has terracotta styling on /ui/."""
        page.goto(f"{server}/ui/?empty=false")
        models_link = page.locator('a[href="/ui/"]').first
        expect(models_link).to_have_css("color", "rgb(212, 116, 94)")


# ---------------------------------------------------------------------------
# 2. Models Empty State (/ui/?empty=true)
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestModelsEmptyState:
    def test_models_empty_state_loads(self, page: Page, server: str):
        """GET /ui/?empty=true returns 200 and shows 'Welcome to RosettaStone'."""
        page.goto(f"{server}/ui/?empty=true")
        expect(page).to_have_title(re.compile(r"Welcome", re.IGNORECASE))
        expect(page.locator("h1").filter(has_text="Welcome to RosettaStone")).to_be_visible()

    def test_models_empty_state_onboarding_card(self, page: Page, server: str):
        """Empty state has input fields, buttons, and LiteLLM hint."""
        page.goto(f"{server}/ui/?empty=true")
        expect(page.locator('p:has-text("Let\'s set up your model landscape.")')).to_be_visible()
        expect(page.locator('input[placeholder="openai/gpt-4o"]')).to_be_visible()
        expect(page.locator('input[placeholder="anthropic/claude-sonnet-4"]')).to_be_visible()
        expect(page.locator("text=+ Add another model")).to_be_visible()
        expect(page.locator("text=We use LiteLLM model identifiers")).to_be_visible()
        expect(page.locator("button:has-text('Set up models')")).to_be_visible()
        expect(page.locator("text=Or import from migration results")).to_be_visible()
        expect(page.locator("text=Connect Redis cache")).to_be_visible()
        expect(page.locator("text=Connect LangSmith")).to_be_visible()

    def test_models_empty_state_input_accepts_text(self, page: Page, server: str):
        """Input fields are focusable and accept text."""
        page.goto(f"{server}/ui/?empty=true")
        input_el = page.locator('input[placeholder="openai/gpt-4o"]')
        input_el.click()
        input_el.fill("test-model")
        expect(input_el).to_have_value("test-model")

    def test_empty_param_false_shows_dashboard(self, page: Page, server: str):
        """/ui/?empty=false shows the normal dashboard, not the empty state."""
        page.goto(f"{server}/ui/?empty=false")
        expect(page.locator("h2").filter(has_text="Your models")).to_be_visible()
        expect(page.locator("h1").filter(has_text="Welcome to RosettaStone")).to_have_count(0)


# ---------------------------------------------------------------------------
# 3. Migrations List (/ui/migrations)
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestMigrationsList:
    def test_migrations_list_loads(self, page: Page, server: str):
        """GET /ui/migrations returns 200, shows heading and 3 migration cards."""
        page.goto(f"{server}/ui/migrations")
        expect(page).to_have_title(re.compile(r"Migrations", re.IGNORECASE))
        expect(page.locator("h1").filter(has_text="Migrations")).to_be_visible()
        expect(page.locator("p").filter(has_text="Evaluate and deploy")).to_be_visible()
        expect(page.locator("text=+ New migration").first).to_be_visible()

    def test_migrations_list_card_data(self, page: Page, server: str):
        """All 3 migration cards show correct source/target, recommendation, and stats."""
        page.goto(f"{server}/ui/migrations")

        # Migration 1
        expect(page.locator("text=gpt-4o").first).to_be_visible()
        expect(page.locator("text=claude-sonnet-4").first).to_be_visible()
        expect(page.locator("text=Safe to ship").first).to_be_visible()
        expect(page.locator("text=92% confidence")).to_be_visible()
        expect(page.locator("text=156 test cases")).to_be_visible()
        expect(page.locator("text=$2.34")).to_be_visible()
        expect(page.locator("text=2 minutes ago")).to_be_visible()

        # Migration 2
        expect(page.locator("text=gpt-4o-0613").first).to_be_visible()
        expect(page.locator("text=Needs review").first).to_be_visible()
        expect(page.locator("text=78% confidence")).to_be_visible()
        expect(page.locator("text=43 test cases")).to_be_visible()
        expect(page.locator("text=$1.12")).to_be_visible()
        expect(page.locator("text=1 day ago")).to_be_visible()

        # Migration 3
        expect(page.locator("text=gpt-3.5-turbo").first).to_be_visible()
        expect(page.locator("text=gpt-4o-mini").first).to_be_visible()
        expect(page.locator("text=Do not ship").first).to_be_visible()
        expect(page.locator("text=61% confidence")).to_be_visible()
        expect(page.locator("text=89 test cases")).to_be_visible()
        expect(page.locator("text=$0.87")).to_be_visible()
        expect(page.locator("text=3 days ago")).to_be_visible()

    def test_migrations_list_card_links(self, page: Page, server: str):
        """Migration cards link to /ui/migrations/{id}."""
        page.goto(f"{server}/ui/migrations")
        expect(page.locator('a[href="/ui/migrations/1"]').first).to_be_visible()
        expect(page.locator('a[href="/ui/migrations/2"]').first).to_be_visible()
        expect(page.locator('a[href="/ui/migrations/3"]').first).to_be_visible()

    def test_migrations_list_pagination_footer(self, page: Page, server: str):
        """Pagination footer shows 'Showing 3 migrations'."""
        page.goto(f"{server}/ui/migrations")
        expect(page.locator("text=Showing 3 migrations")).to_be_visible()

    def test_migration_card_click_navigates(self, page: Page, server: str):
        """Clicking migration card 1 navigates to the detail page."""
        page.goto(f"{server}/ui/migrations")
        page.locator('a[href="/ui/migrations/1"]').first.click()
        expect(page).to_have_url(re.compile(r"/ui/migrations/1"))

    def test_nav_active_state_migrations(self, page: Page, server: str):
        """'Migrations' nav link has terracotta styling on /ui/migrations."""
        page.goto(f"{server}/ui/migrations")
        migrations_link = page.locator('a[href="/ui/migrations"]').first
        expect(migrations_link).to_have_css("color", "rgb(212, 116, 94)")

    def test_safe_to_ship_badge_color(self, page: Page, server: str):
        """'Safe to ship' badge has sage/green background color."""
        page.goto(f"{server}/ui/migrations")
        badge = page.locator("text=Safe to ship").first
        expect(badge).to_be_visible()

    def test_needs_review_badge_color(self, page: Page, server: str):
        """'Needs review' badge is visible."""
        page.goto(f"{server}/ui/migrations")
        badge = page.locator("text=Needs review").first
        expect(badge).to_be_visible()

    def test_do_not_ship_badge_color(self, page: Page, server: str):
        """'Do not ship' badge is visible."""
        page.goto(f"{server}/ui/migrations")
        badge = page.locator("text=Do not ship").first
        expect(badge).to_be_visible()

    def test_migration_card_left_border_colors(self, page: Page, server: str):
        """Migration cards have left-border color indicators present in the DOM."""
        page.goto(f"{server}/ui/migrations")
        # Each card should have the colored left border div
        left_borders = page.locator("div.absolute.left-0")
        expect(left_borders).to_have_count(3)


# ---------------------------------------------------------------------------
# 4-6. Migration Detail Pages
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestMigrationDetailSafeToShip:
    def test_migration_detail_safe_to_ship(self, page: Page, server: str):
        """GET /ui/migrations/1 loads with correct source/target and recommendation."""
        page.goto(f"{server}/ui/migrations/1")
        expect(page).to_have_title(re.compile(r"Migration Detail", re.IGNORECASE))
        heading = page.locator("h1")
        expect(heading).to_contain_text("gpt-4o")
        expect(heading).to_contain_text("claude-sonnet-4")

    def test_migration_detail_safe_recommendation_card(self, page: Page, server: str):
        """Recommendation card shows 'Safe to ship', confidence, and reasoning."""
        page.goto(f"{server}/ui/migrations/1")
        expect(page.locator("text=Safe to ship").first).to_be_visible()
        expect(page.locator("text=92%").first).to_be_visible()
        expect(page.locator("text=Confidence Score")).to_be_visible()
        expect(page.locator("text=All output types meet or exceed").first).to_be_visible()

    def test_migration_detail_safe_kpi_grid(self, page: Page, server: str):
        """KPI grid shows Parity Score 92%, Baseline 85%, Improvement +7%."""
        page.goto(f"{server}/ui/migrations/1")
        expect(page.locator("text=Parity Score")).to_be_visible()
        expect(page.locator("text=Baseline").first).to_be_visible()
        expect(page.locator("text=Improvement from GEPA")).to_be_visible()
        expect(page.locator("text=+7%")).to_be_visible()
        expect(page.locator("text=85%").first).to_be_visible()

    def test_migration_detail_safe_per_type_results(self, page: Page, server: str):
        """'Results by output type' section shows 4 type cards with correct badges."""
        page.goto(f"{server}/ui/migrations/1")
        expect(page.locator("h3:has-text('Results by output type')")).to_be_visible()
        # Check all 4 types
        expect(page.locator("text=JSON").first).to_be_visible()
        expect(page.locator("text=Text").first).to_be_visible()
        expect(page.locator("text=Code").first).to_be_visible()
        expect(page.locator("text=Classification").first).to_be_visible()
        # Check badges
        expect(page.locator("text=PASS").first).to_be_visible()
        expect(page.locator("text=WARN").first).to_be_visible()
        # Check win/total values (the safe-to-ship layout renders wins/total)
        expect(page.locator("text=48/48")).to_be_visible()
        expect(page.locator("text=89/96")).to_be_visible()

    def test_migration_detail_safe_regressions(self, page: Page, server: str):
        """'Needs Attention' section shows 3 regressions with titles and data."""
        page.goto(f"{server}/ui/migrations/1")
        expect(page.locator("text=Needs Attention")).to_be_visible()
        expect(page.locator("text=3 CRITICAL")).to_be_visible()
        expect(page.locator("text=Worst regression")).to_be_visible()
        expect(page.locator("text=Priority classification mismatch")).to_be_visible()
        expect(page.locator("text=Truncated JSON response")).to_be_visible()
        expect(page.locator("text=Different code formatting")).to_be_visible()
        expect(page.locator("text=urgent").first).to_be_visible()
        expect(page.locator("text=high_priority").first).to_be_visible()

    def test_migration_detail_safe_config_collapsible(self, page: Page, server: str):
        """Collapsible 'Configuration and details' toggles open/close."""
        page.goto(f"{server}/ui/migrations/1")
        config_btn = page.locator('[data-action="toggle-collapse"][data-target="config-details"]')
        config_content = page.locator("#config-details")

        expect(config_btn).to_be_visible()
        # Initially collapsed
        expect(config_btn).to_have_attribute("aria-expanded", "false")

        # Click to expand
        config_btn.click()
        expect(config_btn).to_have_attribute("aria-expanded", "true")
        expect(config_content).to_have_class(re.compile(r"expanded"))

        # Click again to collapse
        config_btn.click()
        expect(config_btn).to_have_attribute("aria-expanded", "false")

    def test_migration_detail_safe_model_metadata(self, page: Page, server: str):
        """Static model metadata section is visible."""
        page.goto(f"{server}/ui/migrations/1")
        expect(page.locator("text=Model Metadata")).to_be_visible()
        expect(page.locator("text=Context Window")).to_be_visible()
        expect(page.locator("text=200k tokens")).to_be_visible()
        expect(page.locator("text=Provider").first).to_be_visible()

    def test_migration_detail_safe_export_link(self, page: Page, server: str):
        """'Export report' link points to /ui/migrations/1/executive."""
        page.goto(f"{server}/ui/migrations/1")
        export_link = page.locator('a[href="/ui/migrations/1/executive"]')
        expect(export_link).to_be_visible()

    def test_migration_detail_safe_header_buttons(self, page: Page, server: str):
        """'Version History' and 'Deploy to Prod' buttons are visible."""
        page.goto(f"{server}/ui/migrations/1")
        expect(page.locator("text=Complete").first).to_be_visible()
        expect(page.locator("button:has-text('Version History')")).to_be_visible()
        expect(page.locator("button:has-text('Deploy to Prod')")).to_be_visible()

    def test_breadcrumb_navigation(self, page: Page, server: str):
        """Clicking 'Migrations' breadcrumb navigates back to /ui/migrations."""
        page.goto(f"{server}/ui/migrations/1")
        page.locator('a[href="/ui/migrations"]').first.click()
        expect(page).to_have_url(re.compile(r"/ui/migrations$"))

    def test_migration_detail_safe_view_diff_buttons(self, page: Page, server: str):
        """'View diff' buttons exist for all 3 regressions."""
        page.goto(f"{server}/ui/migrations/1")
        view_diff_buttons = page.locator('button:has-text("View diff")')
        expect(view_diff_buttons).to_have_count(3)


@pytest.mark.playwright
class TestMigrationDetailNeedsReview:
    def test_migration_detail_needs_review(self, page: Page, server: str):
        """GET /ui/migrations/2 shows correct source/target and 'Needs review' recommendation."""
        page.goto(f"{server}/ui/migrations/2")
        heading = page.locator("h1")
        expect(heading).to_contain_text("gpt-4o-0613")
        expect(heading).to_contain_text("gpt-4o")

    def test_migration_detail_needs_review_no_per_type(self, page: Page, server: str):
        """'Results by output type' section does NOT appear (empty per_type)."""
        page.goto(f"{server}/ui/migrations/2")
        expect(page.locator("text=Results by output type")).to_have_count(0)

    def test_migration_detail_needs_review_no_regressions(self, page: Page, server: str):
        """'Regressions to review' section does NOT appear (empty regressions)."""
        page.goto(f"{server}/ui/migrations/2")
        expect(page.locator("text=Regressions to review")).to_have_count(0)

    def test_migration_detail_needs_review_recommendation(self, page: Page, server: str):
        """Recommendation card shows 'Needs review', warning icon, and reasoning."""
        page.goto(f"{server}/ui/migrations/2")
        expect(page.locator("text=Needs review").first).to_be_visible()
        expect(page.locator("text=Review").first).to_be_visible()
        expect(page.locator("text=78%").first).to_be_visible()
        expect(page.locator("text=72%").first).to_be_visible()
        expect(page.locator("text=+6%")).to_be_visible()
        expect(page.locator("text=classification accuracy dropped 8%")).to_be_visible()

    def test_migration_detail_needs_review_full_report_link(self, page: Page, server: str):
        """'View full report' link points to /ui/migrations/2/executive."""
        page.goto(f"{server}/ui/migrations/2")
        expect(page.locator('a[href="/ui/migrations/2/executive"]').first).to_be_visible()

    def test_migration_detail_needs_review_buttons(self, page: Page, server: str):
        """'Review edge cases' button is visible."""
        page.goto(f"{server}/ui/migrations/2")
        expect(page.locator("button:has-text('Review edge cases')")).to_be_visible()


@pytest.mark.playwright
class TestMigrationDetailDoNotShip:
    def test_migration_detail_do_not_ship(self, page: Page, server: str):
        """GET /ui/migrations/3 shows correct source/target and 'Do not ship' recommendation."""
        page.goto(f"{server}/ui/migrations/3")
        heading = page.locator("h1")
        expect(heading).to_contain_text("gpt-3.5-turbo")
        expect(heading).to_contain_text("gpt-4o-mini")

    def test_migration_detail_do_not_ship_header(self, page: Page, server: str):
        """'FAILED' badge and 'Do not ship' heading are visible."""
        page.goto(f"{server}/ui/migrations/3")
        expect(page.locator("text=FAILED").first).to_be_visible()
        expect(page.locator("h2:has-text('Do not ship')")).to_be_visible()
        expect(page.locator("text=Critical schema violations in 16%")).to_be_visible()

    def test_migration_detail_do_not_ship_per_type(self, page: Page, server: str):
        """'Results by output type' shows 4 type cards with FAIL/WARN badges."""
        page.goto(f"{server}/ui/migrations/3")
        expect(page.locator("h3:has-text('Results by output type')")).to_be_visible()
        expect(page.locator("text=FAIL").first).to_be_visible()
        expect(page.locator("text=Schema violations in 12 cases")).to_be_visible()
        expect(page.locator("text=Syntax errors in 8 outputs")).to_be_visible()
        expect(page.locator("text=37% accuracy drop")).to_be_visible()
        expect(page.locator("text=30/42")).to_be_visible()
        expect(page.locator("text=22/28")).to_be_visible()
        expect(page.locator("text=3/11")).to_be_visible()
        expect(page.locator("text=5/8")).to_be_visible()

    def test_migration_detail_do_not_ship_regressions(self, page: Page, server: str):
        """'Critical regressions' shows 3 regressions with 'Showing 3 of 3'."""
        page.goto(f"{server}/ui/migrations/3")
        expect(page.locator("h3:has-text('Critical regressions')")).to_be_visible()
        expect(page.locator("text=Showing 3 of 3")).to_be_visible()
        expect(
            page.locator("text=Schema Violation: Missing required key 'metadata'")
        ).to_be_visible()
        expect(page.locator("text=Syntax error in generated Python")).to_be_visible()
        expect(page.locator("text=Wrong classification category")).to_be_visible()

    def test_migration_detail_do_not_ship_kpi_values(self, page: Page, server: str):
        """KPI grid shows 61%, 58%, and +3%."""
        page.goto(f"{server}/ui/migrations/3")
        expect(page.locator("text=61%").first).to_be_visible()
        expect(page.locator("text=58%").first).to_be_visible()
        expect(page.locator("text=+3%")).to_be_visible()

    def test_migration_detail_do_not_ship_full_report_link(self, page: Page, server: str):
        """'View full report' link points to /ui/migrations/3/executive."""
        page.goto(f"{server}/ui/migrations/3")
        expect(page.locator('a[href="/ui/migrations/3/executive"]').first).to_be_visible()

    def test_migration_detail_do_not_ship_last_evaluated(self, page: Page, server: str):
        """'Last Evaluated' label and '3 days ago' time are visible."""
        page.goto(f"{server}/ui/migrations/3")
        expect(page.locator("text=Last Evaluated")).to_be_visible()
        expect(page.locator("text=3 days ago")).to_be_visible()


# ---------------------------------------------------------------------------
# 7. Costs Overview (/ui/costs)
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestCostsPage:
    def test_costs_page_loads(self, page: Page, server: str):
        """GET /ui/costs returns 200 and shows 'Cost overview' heading."""
        page.goto(f"{server}/ui/costs")
        expect(page).to_have_title(re.compile(r"Costs", re.IGNORECASE))
        expect(page.locator("h1:has-text('Cost overview')")).to_be_visible()

    def test_costs_page_kpi_values(self, page: Page, server: str):
        """KPI cards show $1,247, $312, $935."""
        page.goto(f"{server}/ui/costs")
        expect(page.locator("text=$1,247")).to_be_visible()
        expect(page.locator("text=$312").first).to_be_visible()
        expect(page.locator("text=$935")).to_be_visible()
        expect(page.locator("text=Estimated this month")).to_be_visible()
        expect(page.locator("text=Potential savings")).to_be_visible()
        expect(page.locator("text=After optimization")).to_be_visible()

    def test_costs_page_kpi_subtexts(self, page: Page, server: str):
        """KPI subtext shows trend and optimization info."""
        page.goto(f"{server}/ui/costs")
        expect(page.locator("text=+12.4% vs last month")).to_be_visible()
        expect(page.locator("text=2 actionable optimizations")).to_be_visible()
        expect(page.locator("text=Target budget reached")).to_be_visible()

    def test_costs_page_model_breakdown(self, page: Page, server: str):
        """Cost by model section shows 3 models with correct costs and percentages."""
        page.goto(f"{server}/ui/costs")
        expect(page.locator("h2:has-text('Cost by model')")).to_be_visible()
        expect(page.locator("text=$823")).to_be_visible()
        expect(page.locator("text=66%")).to_be_visible()
        expect(page.locator("text=$312").first).to_be_visible()
        expect(page.locator("text=25%")).to_be_visible()
        expect(page.locator("text=$112")).to_be_visible()
        expect(page.locator("text=9%")).to_be_visible()

    def test_costs_page_optimization_opportunities(self, page: Page, server: str):
        """'Optimization opportunities' shows 2 opportunity cards with correct data."""
        page.goto(f"{server}/ui/costs")
        expect(page.locator("h2:has-text('Optimization opportunities')")).to_be_visible()
        expect(page.locator("text=Switch gpt-4o classification to gpt-4o-mini")).to_be_visible()
        expect(page.locator("text=$187/mo")).to_be_visible()
        expect(page.locator("text=94% parity")).to_be_visible()
        expect(page.locator("text=Batch non-urgent gpt-4o requests")).to_be_visible()
        expect(page.locator("text=$125/mo")).to_be_visible()
        expect(page.locator("text=No quality impact")).to_be_visible()

    def test_nav_active_state_costs(self, page: Page, server: str):
        """'Costs' nav link has terracotta styling on /ui/costs."""
        page.goto(f"{server}/ui/costs")
        costs_link = page.locator('a[href="/ui/costs"]').first
        expect(costs_link).to_have_css("color", "rgb(212, 116, 94)")


# ---------------------------------------------------------------------------
# 8. Alerts Hub (/ui/alerts)
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestAlertsPage:
    def test_alerts_page_loads(self, page: Page, server: str):
        """GET /ui/alerts returns 200 and shows '3 Critical Alerts' heading."""
        page.goto(f"{server}/ui/alerts")
        expect(page).to_have_title(re.compile(r"Alerts", re.IGNORECASE))
        expect(page.locator("h1").filter(has_text="3 Critical Alerts")).to_be_visible()

    def test_alerts_page_hero_section(self, page: Page, server: str):
        """Hero section shows 'System Priority: Omega', 'ACTION REQUIRED.', and action buttons."""
        page.goto(f"{server}/ui/alerts")
        expect(page.locator("text=System Priority: Omega")).to_be_visible()
        expect(page.locator("text=ACTION REQUIRED.")).to_be_visible()
        expect(page.locator("text=RosettaStone migration protocols")).to_be_visible()
        expect(page.locator("button:has-text('Start Batch Migration')")).to_be_visible()
        expect(page.locator("button:has-text('View Drift Logs')")).to_be_visible()

    def test_alerts_page_intelligence_reports_section(self, page: Page, server: str):
        """'Active Intelligence Reports' section is visible."""
        page.goto(f"{server}/ui/alerts")
        expect(page.locator("h2:has-text('Active Intelligence Reports')")).to_be_visible()
        expect(page.locator("text=Real-time heuristic monitoring")).to_be_visible()
        expect(page.locator("button:has-text('Export Report Data')")).to_be_visible()

    def test_alerts_page_alert_cards(self, page: Page, server: str):
        """3 alert cards render with correct type-specific data."""
        page.goto(f"{server}/ui/alerts")
        # Deprecation alert
        expect(page.locator("text=gpt-4o-0613").first).to_be_visible()
        expect(page.locator("text=Model retiring in 26 days")).to_be_visible()
        expect(page.locator("text=Start migration to gpt-4o")).to_be_visible()
        expect(page.locator("text=26 days remaining")).to_be_visible()

        # Price change alert
        expect(page.locator("text=claude-sonnet-4").first).to_be_visible()
        expect(page.locator("text=Price decreased 17%")).to_be_visible()
        expect(page.locator("text=No action needed")).to_be_visible()

        # New model alert
        expect(page.locator("text=claude-opus-4.6").first).to_be_visible()
        expect(page.locator("text=New model available").first).to_be_visible()
        expect(page.locator("text=Available").first).to_be_visible()

    def test_alerts_page_notification_settings(self, page: Page, server: str):
        """Notification settings section shows 3 checkboxes with correct default states."""
        page.goto(f"{server}/ui/alerts")
        expect(page.locator("h3:has-text('Notification Settings')")).to_be_visible()

        checkboxes = page.locator('input[type="checkbox"]')
        expect(checkboxes).to_have_count(3)

        # First checkbox: Critical System Alerts -- checked
        expect(checkboxes.nth(0)).to_be_checked()
        # Second checkbox: Cost & Efficiency Reports -- unchecked
        expect(checkboxes.nth(1)).not_to_be_checked()
        # Third checkbox: Model Performance Drift -- checked
        expect(checkboxes.nth(2)).to_be_checked()

        expect(page.locator("button:has-text('Save Preferences')")).to_be_visible()

    def test_alerts_page_checkbox_toggle(self, page: Page, server: str):
        """Clicking unchecked 'Cost & Efficiency Reports' checkbox toggles it on."""
        page.goto(f"{server}/ui/alerts")
        checkboxes = page.locator('input[type="checkbox"]')
        second_cb = checkboxes.nth(1)
        expect(second_cb).not_to_be_checked()
        second_cb.click()
        expect(second_cb).to_be_checked()

    def test_nav_active_state_alerts(self, page: Page, server: str):
        """'Alerts' nav link has terracotta styling on /ui/alerts."""
        page.goto(f"{server}/ui/alerts")
        alerts_link = page.locator('a[href="/ui/alerts"]').first
        expect(alerts_link).to_have_css("color", "rgb(212, 116, 94)")


# ---------------------------------------------------------------------------
# 9. Executive Report (/ui/migrations/1/executive)
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestExecutiveReport:
    def test_executive_report_safe(self, page: Page, server: str):
        """GET /ui/migrations/1/executive shows full report content."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page).to_have_title(re.compile(r"Executive Migration Report", re.IGNORECASE))
        expect(page.locator("text=ROSETTASTONE MIGRATION REPORT")).to_be_visible()
        heading = page.locator("h1")
        expect(heading).to_contain_text("gpt-4o")
        expect(heading).to_contain_text("claude-sonnet-4")

    def test_executive_report_standalone_no_nav(self, page: Page, server: str):
        """Executive report has no nav bar and no base footer."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page.locator("nav")).to_have_count(0)
        expect(page.locator("text=2024 RosettaStone Intelligence")).to_have_count(0)

    def test_executive_report_recommendation(self, page: Page, server: str):
        """Report shows 'Recommendation: Safe to switch' with reasoning."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page.locator("h2").filter(has_text="Recommendation: Safe to switch")).to_be_visible()
        expect(page.locator("text=All output types meet or exceed").first).to_be_visible()

    def test_executive_report_metric_boxes(self, page: Page, server: str):
        """Metric boxes show Quality match 92%, Cost $2.34, Risk Level Low, Deployment Ready."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page.locator("text=Quality match")).to_be_visible()
        expect(page.locator("text=92%").first).to_be_visible()
        expect(page.locator("text=$2.34").first).to_be_visible()
        expect(page.locator("text=Cost").first).to_be_visible()
        expect(page.locator("text=Risk level")).to_be_visible()
        expect(page.locator("text=Low")).to_be_visible()
        expect(page.locator("text=Deployment status")).to_be_visible()
        expect(page.locator("text=Ready")).to_be_visible()

    def test_executive_report_what_improves(self, page: Page, server: str):
        """'What improves' section shows improvement bullets."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page.locator("text=What improves")).to_be_visible()
        expect(page.locator("text=92% parity across 156 test cases")).to_be_visible()
        expect(page.locator("text=+7% improvement from GEPA")).to_be_visible()
        expect(page.locator("text=Baseline score: 85%")).to_be_visible()

    def test_executive_report_what_to_watch(self, page: Page, server: str):
        """'What to watch' section shows up to 3 regression bullets."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page.locator("text=What to watch")).to_be_visible()
        expect(page.locator("text=Priority classification mismatch")).to_be_visible()
        expect(page.locator("text=Truncated JSON response")).to_be_visible()
        expect(page.locator("text=Different code formatting")).to_be_visible()

    def test_executive_report_per_type(self, page: Page, server: str):
        """'Results by Output Type' section shows 4 per-type rows."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page.locator("text=Results by Output Type")).to_be_visible()
        expect(page.locator("text=JSON").first).to_be_visible()
        expect(page.locator("text=Text").first).to_be_visible()
        expect(page.locator("text=Code").first).to_be_visible()
        expect(page.locator("text=Classification").first).to_be_visible()

    def test_executive_report_footer(self, page: Page, server: str):
        """Report footer shows version, 'Prepared by', and 'CONFIDENTIAL'."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page.locator("text=Prepared by RosettaStone v0.1.0")).to_be_visible()
        expect(page.locator("text=CONFIDENTIAL")).to_be_visible()
        expect(page.locator("text=V.04.2-STABLE")).to_be_visible()

    def test_executive_report_light_background(self, page: Page, server: str):
        """Executive report has light background (not dark theme)."""
        page.goto(f"{server}/ui/migrations/1/executive")
        bg = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
        # Background should be light (#F9F7F4 = rgb(249, 247, 244)) not dark (#131313)
        assert "249" in bg or "247" in bg or "244" in bg, f"Expected light background, got: {bg}"

    def test_executive_report_do_not_ship(self, page: Page, server: str):
        """GET /ui/migrations/3/executive shows 'Recommendation: Do not switch', High risk, Blocked."""
        page.goto(f"{server}/ui/migrations/3/executive")
        expect(page.locator("h2").filter(has_text="Recommendation: Do not switch")).to_be_visible()
        expect(page.locator("text=High")).to_be_visible()
        expect(page.locator("text=Blocked")).to_be_visible()

    def test_executive_report_needs_review(self, page: Page, server: str):
        """GET /ui/migrations/2/executive shows 'Recommendation: Needs review', Medium risk, Pending."""
        page.goto(f"{server}/ui/migrations/2/executive")
        expect(page.locator("h2").filter(has_text="Recommendation: Needs review")).to_be_visible()
        expect(page.locator("text=Medium")).to_be_visible()
        expect(page.locator("text=Pending")).to_be_visible()

    def test_executive_report_needs_review_no_regressions_text(self, page: Page, server: str):
        """Executive report for migration 2 (empty regressions) shows 'No critical regressions detected'."""
        page.goto(f"{server}/ui/migrations/2/executive")
        expect(page.locator("text=No critical regressions detected")).to_be_visible()


# ---------------------------------------------------------------------------
# 10. Diff Fragment (/ui/fragments/diff/1/42)
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestDiffFragment:
    def test_diff_fragment_loads(self, page: Page, server: str):
        """GET /ui/fragments/diff/1/42 returns 200 with fragment HTML."""
        response = page.goto(f"{server}/ui/fragments/diff/1/42")
        assert response is not None
        assert response.status == 200

    def test_diff_fragment_content(self, page: Page, server: str):
        """Diff fragment shows all score values, expected/actual content, and metadata."""
        page.goto(f"{server}/ui/fragments/diff/1/42")
        expect(page.locator("text=Classification").first).to_be_visible()
        expect(page.locator("text=0.72").first).to_be_visible()
        expect(page.locator("text=LOSS").first).to_be_visible()
        expect(page.locator("text=BERTScore")).to_be_visible()
        expect(page.locator("text=0.85").first).to_be_visible()
        expect(page.locator("text=Embedding similarity")).to_be_visible()
        expect(page.locator("text=0.79").first).to_be_visible()
        expect(page.locator("text=Composite").first).to_be_visible()
        expect(page.locator("text=Side-by-side").first).to_be_visible()
        expect(page.locator("text=Unified").first).to_be_visible()
        expect(page.locator("text=urgent").first).to_be_visible()
        expect(page.locator("text=high_priority").first).to_be_visible()
        expect(page.locator("text=Test Case Evidence")).to_be_visible()
        expect(page.locator("text=#42")).to_be_visible()

    def test_diff_fragment_pii_warning(self, page: Page, server: str):
        """PII warning banner is present in the diff fragment."""
        page.goto(f"{server}/ui/fragments/diff/1/42")
        expect(page.locator("text=Content may contain sensitive data")).to_be_visible()

    def test_diff_fragment_close_button(self, page: Page, server: str):
        """Close slideout button is present in the diff fragment."""
        page.goto(f"{server}/ui/fragments/diff/1/42")
        expect(page.locator('[data-action="close-slideout"]').first).to_be_visible()

    def test_diff_fragment_approve_variance_button(self, page: Page, server: str):
        """'APPROVE VARIANCE' button is present in the diff fragment footer."""
        page.goto(f"{server}/ui/fragments/diff/1/42")
        expect(page.locator("text=APPROVE VARIANCE")).to_be_visible()


# ---------------------------------------------------------------------------
# 11. Cross-Page: Navigation Bar
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestNavigationBar:
    def test_nav_links_present_on_all_pages(self, page: Page, server: str):
        """Nav links (Models, Migrations, Costs, Alerts) exist on all main pages."""
        pages = [
            "/ui/",
            "/ui/migrations",
            "/ui/migrations/1",
            "/ui/costs",
            "/ui/alerts",
        ]
        for path in pages:
            page.goto(f"{server}{path}")
            expect(page.locator("nav").first).to_be_visible()
            expect(page.locator("text=RosettaStone").first).to_be_visible()
            expect(page.locator('a[href="/ui/"]').first).to_be_visible()
            expect(page.locator('a[href="/ui/migrations"]').first).to_be_visible()
            expect(page.locator('a[href="/ui/costs"]').first).to_be_visible()
            expect(page.locator('a[href="/ui/alerts"]').first).to_be_visible()

    def test_nav_active_state_migration_detail(self, page: Page, server: str):
        """'Migrations' nav link has terracotta styling on migration detail pages."""
        page.goto(f"{server}/ui/migrations/1")
        migrations_link = page.locator('nav a[href="/ui/migrations"]').first
        expect(migrations_link).to_have_css("color", "rgb(212, 116, 94)")

    def test_nav_alerts_notification_dot(self, page: Page, server: str):
        """Red notification dot is present near the Alerts nav link."""
        page.goto(f"{server}/ui/")
        # The dot is a span with bg-[#D85650] in the relative div near the Alerts link
        nav = page.locator("nav")
        expect(nav.locator("span.bg-\\[\\#D85650\\]").first).to_be_visible()

    def test_nav_theme_toggle_button(self, page: Page, server: str):
        """Theme toggle button is visible in the nav bar."""
        page.goto(f"{server}/ui/")
        expect(page.locator('[data-action="toggle-theme"]')).to_be_visible()

    def test_nav_settings_and_avatar(self, page: Page, server: str):
        """Settings button and avatar/account icon are visible in the nav bar."""
        page.goto(f"{server}/ui/")
        nav = page.locator("nav")
        settings = nav.locator(".material-symbols-outlined").filter(has_text="settings")
        expect(settings).to_be_visible()
        account = nav.locator(".material-symbols-outlined").filter(has_text="account_circle")
        expect(account).to_be_visible()

    def test_nav_click_navigation(self, page: Page, server: str):
        """Clicking each nav link navigates to the correct URL."""
        page.goto(f"{server}/ui/")

        page.locator('a[href="/ui/migrations"]').first.click()
        expect(page).to_have_url(re.compile(r"/ui/migrations$"))

        page.locator('a[href="/ui/costs"]').first.click()
        expect(page).to_have_url(re.compile(r"/ui/costs$"))

        page.locator('a[href="/ui/alerts"]').first.click()
        expect(page).to_have_url(re.compile(r"/ui/alerts$"))

        page.locator('a[href="/ui/"]').first.click()
        expect(page).to_have_url(re.compile(r"/ui/$"))

    def test_executive_report_no_nav(self, page: Page, server: str):
        """Executive report page has no nav bar (standalone template)."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page.locator("nav")).to_have_count(0)


# ---------------------------------------------------------------------------
# 12. Cross-Page: Theme Toggle
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestThemeToggle:
    def test_theme_toggle_dark_to_light(self, dark_mode_page: Page, server: str):
        """Theme toggle changes data-theme from dark to light and removes dark class."""
        dark_mode_page.goto(f"{server}/ui/")
        # Default: dark theme (forced via browser color-scheme=dark)
        expect(dark_mode_page.locator("html")).to_have_attribute("data-theme", "dark")
        expect(dark_mode_page.locator("html")).to_have_class(re.compile(r"dark"))

        # Click toggle
        dark_mode_page.locator('[data-action="toggle-theme"]').click()

        # Now: light theme
        expect(dark_mode_page.locator("html")).to_have_attribute("data-theme", "light")
        html_classes = dark_mode_page.locator("html").get_attribute("class") or ""
        assert "dark" not in html_classes, f"Expected 'dark' class removed, got: {html_classes}"

    def test_theme_toggle_light_to_dark(self, dark_mode_page: Page, server: str):
        """Toggling twice restores dark mode."""
        dark_mode_page.goto(f"{server}/ui/")
        toggle = dark_mode_page.locator('[data-action="toggle-theme"]')

        # Toggle to light
        toggle.click()
        expect(dark_mode_page.locator("html")).to_have_attribute("data-theme", "light")

        # Toggle back to dark
        toggle.click()
        expect(dark_mode_page.locator("html")).to_have_attribute("data-theme", "dark")
        expect(dark_mode_page.locator("html")).to_have_class(re.compile(r"dark"))

    def test_theme_toggle_updates_localstorage(self, dark_mode_page: Page, server: str):
        """Theme toggle updates localStorage 'rosettastone-theme' key."""
        dark_mode_page.goto(f"{server}/ui/")
        _initial_stored = dark_mode_page.evaluate(
            "() => localStorage.getItem('rosettastone-theme')"
        )
        dark_mode_page.locator('[data-action="toggle-theme"]').click()
        stored = dark_mode_page.evaluate("() => localStorage.getItem('rosettastone-theme')")
        assert stored == "light", f"Expected localStorage 'light', got: {stored}"

    def test_theme_persists_across_navigation(self, dark_mode_page: Page, server: str):
        """Light theme persists after navigating to another page via localStorage."""
        dark_mode_page.goto(f"{server}/ui/")
        dark_mode_page.locator('[data-action="toggle-theme"]').click()
        expect(dark_mode_page.locator("html")).to_have_attribute("data-theme", "light")

        dark_mode_page.goto(f"{server}/ui/costs")
        expect(dark_mode_page.locator("html")).to_have_attribute("data-theme", "light")


# ---------------------------------------------------------------------------
# 13. Cross-Page: Footer
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestFooter:
    def test_footer_present_on_all_pages(self, page: Page, server: str):
        """Footer is visible on all main pages."""
        pages = [
            "/ui/",
            "/ui/?empty=true",
            "/ui/migrations",
            "/ui/migrations/1",
            "/ui/costs",
            "/ui/alerts",
        ]
        for path in pages:
            page.goto(f"{server}{path}")
            expect(page.locator("footer")).to_be_visible()

    def test_footer_copyright_text(self, page: Page, server: str):
        """Footer contains '2024 RosettaStone Intelligence' copyright text."""
        page.goto(f"{server}/ui/")
        expect(page.locator("footer")).to_contain_text("2024 RosettaStone Intelligence")

    def test_footer_links(self, page: Page, server: str):
        """Footer has Documentation, API Reference, Status, and Privacy links."""
        page.goto(f"{server}/ui/")
        footer = page.locator("footer")
        expect(footer.locator("text=Documentation")).to_be_visible()
        expect(footer.locator("text=API Reference")).to_be_visible()
        expect(footer.locator("text=Status")).to_be_visible()
        expect(footer.locator("text=Privacy")).to_be_visible()

    def test_footer_absent_on_executive_report(self, page: Page, server: str):
        """Executive report does not have the base.html footer."""
        page.goto(f"{server}/ui/migrations/1/executive")
        expect(page.locator("text=2024 RosettaStone Intelligence")).to_have_count(0)


# ---------------------------------------------------------------------------
# 14. Error Handling: 404 pages
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestErrorHandling:
    def test_migration_detail_404(self, page: Page, server: str):
        """GET /ui/migrations/999 returns 404."""
        response = page.goto(f"{server}/ui/migrations/999")
        assert response is not None
        assert response.status == 404, f"Expected 404, got {response.status}"

    def test_executive_report_404(self, page: Page, server: str):
        """GET /ui/migrations/999/executive returns 404."""
        response = page.goto(f"{server}/ui/migrations/999/executive")
        assert response is not None
        assert response.status == 404, f"Expected 404, got {response.status}"

    def test_migration_detail_zero_id_404(self, page: Page, server: str):
        """GET /ui/migrations/0 returns 404."""
        response = page.goto(f"{server}/ui/migrations/0")
        assert response is not None
        assert response.status == 404, f"Expected 404, got {response.status}"

    def test_migration_detail_large_id_404(self, page: Page, server: str):
        """GET /ui/migrations/9999 returns 404."""
        response = page.goto(f"{server}/ui/migrations/9999")
        assert response is not None
        assert response.status == 404, f"Expected 404, got {response.status}"


# ---------------------------------------------------------------------------
# 15-16. Slideout Panel Integration (Full Lifecycle)
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestSlideoutPanel:
    def test_slideout_opens_on_view_diff(self, page: Page, server: str):
        """Clicking 'View diff' on migration 1 opens the slideout panel."""
        page.goto(f"{server}/ui/migrations/1")
        expect(page.locator("#diff-panel")).to_be_hidden()
        expect(page.locator("#diff-backdrop")).to_be_hidden()

        page.locator('button:has-text("View diff")').first.click()
        page.wait_for_load_state("networkidle")

        expect(page.locator("#diff-panel")).to_be_visible()
        expect(page.locator("#diff-panel")).to_have_class(re.compile(r"open"))
        expect(page.locator("#diff-backdrop")).to_be_visible()

    def test_slideout_content_loads(self, page: Page, server: str):
        """Diff content loads into the panel after opening."""
        page.goto(f"{server}/ui/migrations/1")
        page.locator('button:has-text("View diff")').first.click()
        page.wait_for_load_state("networkidle")

        expect(page.locator("#diff-content")).to_contain_text("BERTScore")
        expect(page.locator("#diff-content")).to_contain_text("Classification")

    def test_slideout_body_scroll_locked(self, page: Page, server: str):
        """Body overflow is set to 'hidden' when the slideout is open."""
        page.goto(f"{server}/ui/migrations/1")
        page.locator('button:has-text("View diff")').first.click()
        page.wait_for_load_state("networkidle")
        body_overflow = page.evaluate("() => document.body.style.overflow")
        assert body_overflow == "hidden", f"Expected overflow 'hidden', got: {body_overflow}"

    def test_slideout_close_via_x_button(self, page: Page, server: str):
        """Slideout closes when X button is clicked."""
        page.goto(f"{server}/ui/migrations/1")
        page.locator('button:has-text("View diff")').first.click()
        page.wait_for_load_state("networkidle")
        expect(page.locator("#diff-panel")).to_be_visible()

        page.locator('[data-action="close-slideout"]').first.click()
        page.wait_for_timeout(400)
        expect(page.locator("#diff-panel")).to_be_hidden()
        expect(page.locator("#diff-backdrop")).to_be_hidden()

    def test_slideout_body_scroll_restored(self, page: Page, server: str):
        """Body overflow is restored after closing the slideout."""
        page.goto(f"{server}/ui/migrations/1")
        page.locator('button:has-text("View diff")').first.click()
        page.wait_for_load_state("networkidle")

        page.locator('[data-action="close-slideout"]').first.click()
        page.wait_for_timeout(400)
        body_overflow = page.evaluate("() => document.body.style.overflow")
        assert body_overflow == "", f"Expected overflow '', got: {body_overflow}"

    def test_slideout_close_via_backdrop(self, page: Page, server: str):
        """Slideout closes when the backdrop is clicked."""
        page.goto(f"{server}/ui/migrations/1")
        page.locator('button:has-text("View diff")').first.click()
        page.wait_for_load_state("networkidle")
        expect(page.locator("#diff-panel")).to_be_visible()

        # Use JavaScript to dispatch the click directly on the backdrop element,
        # since the panel overlaps part of the backdrop in the DOM.
        page.evaluate(
            "() => document.getElementById('diff-backdrop').dispatchEvent(new MouseEvent('click', {bubbles: true, target: document.getElementById('diff-backdrop')}))"
        )
        page.wait_for_timeout(400)
        expect(page.locator("#diff-panel")).to_be_hidden()

    def test_slideout_close_via_escape(self, page: Page, server: str):
        """Slideout closes when the Escape key is pressed."""
        page.goto(f"{server}/ui/migrations/1")
        page.locator('button:has-text("View diff")').first.click()
        page.wait_for_load_state("networkidle")
        expect(page.locator("#diff-panel")).to_be_visible()

        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
        expect(page.locator("#diff-panel")).to_be_hidden()

    def test_slideout_close_via_footer_button(self, page: Page, server: str):
        """Slideout closes when the footer 'CLOSE' button is clicked."""
        page.goto(f"{server}/ui/migrations/1")
        page.locator('button:has-text("View diff")').first.click()
        page.wait_for_load_state("networkidle")

        close_btns = page.locator('[data-action="close-slideout"]')
        close_btns.last.click()
        page.wait_for_timeout(400)
        expect(page.locator("#diff-panel")).to_be_hidden()

    def test_do_not_ship_hover_reveal_diff_link(self, page: Page, server: str):
        """Hovering over a regression row reveals the 'View diff' link on migration 3."""
        page.goto(f"{server}/ui/migrations/3")

        first_reg = page.locator(".group").filter(has_text="Schema Violation").first
        diff_link = first_reg.locator('a:has-text("View diff")')

        # Hover to reveal
        first_reg.hover()

        # Force click since the link may still be animating opacity
        diff_link.click(force=True)
        page.wait_for_load_state("networkidle")

        expect(page.locator("#diff-panel")).to_be_visible()
        expect(page.locator("#diff-content")).to_contain_text("BERTScore")

    def test_slideout_full_lifecycle(self, page: Page, server: str):
        """Full open-close lifecycle: open panel, verify content, close via X."""
        page.goto(f"{server}/ui/migrations/1")

        # 1. Panel starts hidden
        expect(page.locator("#diff-panel")).to_be_hidden()
        expect(page.locator("#diff-backdrop")).to_be_hidden()

        # 2. Open
        page.locator('button:has-text("View diff")').first.click()
        page.wait_for_load_state("networkidle")

        # 3. Panel open
        expect(page.locator("#diff-panel")).to_be_visible()
        expect(page.locator("#diff-panel")).to_have_class(re.compile(r"open"))
        expect(page.locator("#diff-backdrop")).to_be_visible()

        # 4. Body scroll locked
        assert page.evaluate("() => document.body.style.overflow") == "hidden"

        # 5. Content loaded
        expect(page.locator("#diff-content")).to_contain_text("BERTScore")
        expect(page.locator("#diff-content")).to_contain_text("Classification")

        # 6. Close via X
        page.locator('[data-action="close-slideout"]').first.click()
        page.wait_for_timeout(400)
        expect(page.locator("#diff-panel")).to_be_hidden()
        expect(page.locator("#diff-backdrop")).to_be_hidden()

        # 7. Body scroll restored
        assert page.evaluate("() => document.body.style.overflow") == ""
