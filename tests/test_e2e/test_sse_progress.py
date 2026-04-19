"""Playwright end-to-end tests for SSE live-progress feature (PR #5).

Validates that SSE events injected through the test app's /test/* endpoints
correctly update DOM elements on the migration detail page.

Uses port 8766 (distinct from test_playwright_ui.py's 8765) and starts
tests.test_e2e.sse_test_app:create_test_app (the real app + test injection
endpoints).
"""

from __future__ import annotations

import pathlib
import signal
import subprocess
import time
import urllib.request

import httpx
import pytest
from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

BASE_URL = "http://localhost:8766"


# ---------------------------------------------------------------------------
# Helpers
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
            time.sleep(0.5)
    except Exception:
        pass


def _create_running_migration(base_url: str, **kwargs: object) -> int:
    """Create a migration record in 'running' state via test endpoint."""
    body: dict[str, object] = {
        "source_model": "test/mock-source",
        "target_model": "test/mock-target",
        "current_stage": "preflight",
    }
    body.update(kwargs)
    resp = httpx.post(f"{base_url}/test/create-running-migration", json=body, timeout=10)
    resp.raise_for_status()
    return resp.json()["id"]


class SSEEventInjector:
    """Injects SSE events through test endpoints and handles synchronization."""

    def __init__(self, base_url: str, migration_id: int) -> None:
        self.base_url = base_url
        self.migration_id = migration_id
        self._client = httpx.Client(timeout=10)

    def wait_for_sse_connection(self, timeout: float = 10.0) -> None:
        """Poll until at least one SSE client is connected for this migration."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = self._client.get(
                f"{self.base_url}/test/sse-clients/{self.migration_id}"
            )
            if resp.json()["count"] > 0:
                return
            time.sleep(0.05)  # 50ms polling — condition-based, not fixed wait
        raise TimeoutError(
            f"No SSE client connected for migration {self.migration_id} within {timeout}s"
        )

    def emit(self, event: dict) -> None:  # type: ignore[type-arg]
        """Inject an SSE event into the progress hub."""
        self._client.post(
            f"{self.base_url}/test/emit-event/{self.migration_id}", json=event
        )

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def server():
    """Start the SSE test app (session-scoped, started once at port 8766)."""
    _kill_port(8766)

    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "tests.test_e2e.sse_test_app:create_test_app",
            "--factory",
            "--port",
            "8766",
            "--timeout-keep-alive",
            "0",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

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
    browser = pw.chromium.launch(headless=True)
    yield browser
    browser.close()
    pw.stop()


@pytest.fixture
def page(server, browser_instance):
    """Create a new browser page for each test."""
    context = browser_instance.new_context()
    pg = context.new_page()
    pg.set_default_timeout(30_000)
    yield pg
    pg.close()
    context.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.playwright
class TestSSELiveProgress:
    # -----------------------------------------------------------------------
    # Test 1: progress bar exists on load
    # -----------------------------------------------------------------------

    def test_progress_bar_exists_on_load(self, page: Page, server: str) -> None:
        """#overall-progress element is present with initial width of 0%."""
        mid = _create_running_migration(server)
        page.goto(f"{server}/ui/migrations/{mid}")

        progress_bar = page.locator("#overall-progress")
        # Element starts at width=0% so Playwright considers it "not visible";
        # wait for "attached" (present in DOM) instead.
        progress_bar.wait_for(state="attached")

        width = page.evaluate(
            "() => document.getElementById('overall-progress').style.width"
        )
        assert width == "0%" or width == "", (
            f"Expected initial width '0%' or '', got '{width}'"
        )

    # -----------------------------------------------------------------------
    # Test 2: progress bar width strictly increases
    # -----------------------------------------------------------------------

    def test_progress_bar_width_strictly_increases(
        self, page: Page, server: str
    ) -> None:
        """Progress bar width strictly increases as events are emitted."""
        mid = _create_running_migration(server)
        page.goto(f"{server}/ui/migrations/{mid}")
        page.locator("#overall-progress").wait_for(state="attached")

        injector = SSEEventInjector(server, mid)
        try:
            injector.wait_for_sse_connection()

            # Capture initial width
            initial_width = page.evaluate(
                "() => parseFloat(document.getElementById('overall-progress').style.width) || 0"
            )

            widths = [initial_width]

            for progress_val, expected_pct in [(0.3, 30), (0.7, 70), (1.0, 100)]:
                injector.emit(
                    {
                        "type": "progress",
                        "overall_progress": progress_val,
                        "current_stage": "gepa_optimization",
                        "migration_id": mid,
                    }
                )
                expected = expected_pct
                page.wait_for_function(
                    f"() => parseFloat(document.getElementById('overall-progress').style.width) >= {expected}"
                )
                current = page.evaluate(
                    "() => parseFloat(document.getElementById('overall-progress').style.width)"
                )
                widths.append(current)

            # Assert strictly increasing: 0 < 30 < 70 < 100
            assert widths[0] < widths[1] < widths[2] < widths[3], (
                f"Expected strictly increasing widths, got: {widths}"
            )
        finally:
            injector.close()

    # -----------------------------------------------------------------------
    # Test 3: current stage text changes
    # -----------------------------------------------------------------------

    def test_current_stage_text_changes(self, page: Page, server: str) -> None:
        """#current-stage text updates to reflect each emitted stage."""
        mid = _create_running_migration(server)
        page.goto(f"{server}/ui/migrations/{mid}")
        page.locator("#overall-progress").wait_for(state="attached")

        injector = SSEEventInjector(server, mid)
        try:
            injector.wait_for_sse_connection()

            stages = ["baseline_eval", "gepa_optimization", "validation_eval"]
            seen_texts: list[str] = []

            for stage in stages:
                # JS replaces underscores with spaces
                expected_text = stage.replace("_", " ")
                injector.emit(
                    {
                        "type": "progress",
                        "overall_progress": 0.3,
                        "current_stage": stage,
                        "migration_id": mid,
                    }
                )
                page.wait_for_function(
                    f"() => document.getElementById('current-stage').textContent.trim() === '{expected_text}'"
                )
                current_text = page.evaluate(
                    "() => document.getElementById('current-stage').textContent.trim()"
                )
                seen_texts.append(current_text)

            # Assert at least 2 distinct stage texts were seen
            distinct = set(seen_texts)
            assert len(distinct) >= 2, (
                f"Expected at least 2 distinct stage texts, got: {seen_texts}"
            )
        finally:
            injector.close()

    # -----------------------------------------------------------------------
    # Test 4: ETA display updates
    # -----------------------------------------------------------------------

    def test_eta_display_updates(self, page: Page, server: str) -> None:
        """#eta-display updates to show minutes remaining when eta_seconds is set."""
        mid = _create_running_migration(server)
        page.goto(f"{server}/ui/migrations/{mid}")
        page.locator("#overall-progress").wait_for(state="attached")

        injector = SSEEventInjector(server, mid)
        try:
            injector.wait_for_sse_connection()

            injector.emit(
                {
                    "type": "progress",
                    "overall_progress": 0.2,
                    "current_stage": "gepa_optimization",
                    "eta_seconds": 120,
                    "migration_id": mid,
                }
            )

            # Wait until eta-display is non-empty
            page.wait_for_function(
                "() => document.getElementById('eta-display').textContent.trim().length > 0"
            )

            eta_text = page.evaluate(
                "() => document.getElementById('eta-display').textContent.trim()"
            )
            assert "min remaining" in eta_text, (
                f"Expected 'min remaining' in eta text, got: '{eta_text}'"
            )
        finally:
            injector.close()

    # -----------------------------------------------------------------------
    # Test 5: GEPA sparkline receives data
    # -----------------------------------------------------------------------

    def test_gepa_sparkline_receives_data(self, page: Page, server: str) -> None:
        """gepa_iteration events unhide panel, add bars to sparkline, update counter."""
        mid = _create_running_migration(server)
        page.goto(f"{server}/ui/migrations/{mid}")
        page.locator("#overall-progress").wait_for(state="attached")

        # Panel starts hidden
        panel_classes = page.evaluate(
            "() => document.getElementById('gepa-progress-panel').className"
        )
        assert "hidden" in panel_classes, (
            f"Expected #gepa-progress-panel to be hidden initially, classes: {panel_classes}"
        )

        injector = SSEEventInjector(server, mid)
        try:
            injector.wait_for_sse_connection()

            for i, score in [(1, 0.6), (2, 0.7), (3, 0.8)]:
                injector.emit(
                    {
                        "type": "gepa_iteration",
                        "iteration": i,
                        "total_iterations": 5,
                        "running_mean_score": score,
                        "migration_id": mid,
                    }
                )
                # Wait for the counter text to reflect this iteration
                page.wait_for_function(
                    f"() => document.getElementById('gepa-iteration-counter').textContent.includes('{i} / 5')"
                )

            # Panel should no longer be hidden
            panel_classes_after = page.evaluate(
                "() => document.getElementById('gepa-progress-panel').className"
            )
            assert "hidden" not in panel_classes_after, (
                f"Expected panel to be visible, but classes still contain 'hidden': {panel_classes_after}"
            )

            # Sparkline should have at least 3 child elements
            bar_count = page.evaluate(
                "() => document.getElementById('gepa-sparkline').children.length"
            )
            assert bar_count >= 3, (
                f"Expected at least 3 sparkline bars, got: {bar_count}"
            )

            # Counter text should show "3 / 5"
            counter_text = page.evaluate(
                "() => document.getElementById('gepa-iteration-counter').textContent.trim()"
            )
            assert "3 / 5" in counter_text, (
                f"Expected counter to contain '3 / 5', got: '{counter_text}'"
            )
        finally:
            injector.close()

    # -----------------------------------------------------------------------
    # Test 6: eval_pair counter updates
    # -----------------------------------------------------------------------

    def test_eval_pair_counter_updates(self, page: Page, server: str) -> None:
        """eval_pair events unhide panel and update counter text."""
        mid = _create_running_migration(server, current_stage="baseline_eval")
        page.goto(f"{server}/ui/migrations/{mid}")
        page.locator("#overall-progress").wait_for(state="attached")

        injector = SSEEventInjector(server, mid)
        try:
            injector.wait_for_sse_connection()

            for i in [1, 2, 3]:
                injector.emit(
                    {
                        "type": "eval_pair",
                        "pair_index": i,
                        "total_pairs": 5,
                        "migration_id": mid,
                    }
                )
                # Wait for counter to update to this pair index
                page.wait_for_function(
                    f"() => document.getElementById('eval-pair-counter').textContent.trim() === '{i}/5'"
                )

            # Panel should be visible
            panel_classes = page.evaluate(
                "() => document.getElementById('eval-pair-panel').className"
            )
            assert "hidden" not in panel_classes, (
                f"Expected #eval-pair-panel to be visible, but classes: {panel_classes}"
            )

            # Counter should show "3/5"
            counter_text = page.evaluate(
                "() => document.getElementById('eval-pair-counter').textContent.trim()"
            )
            assert counter_text == "3/5", (
                f"Expected '3/5', got: '{counter_text}'"
            )
        finally:
            injector.close()

    # -----------------------------------------------------------------------
    # Test 7: no console errors
    # -----------------------------------------------------------------------

    def test_no_console_errors(self, page: Page, server: str) -> None:
        """No JS console errors are produced when processing a full SSE event sequence.

        Note: CSP inline-style warnings are pre-existing browser behaviour for
        ``element.style.width = …`` when the app's ``style-src`` directive lacks
        ``'unsafe-inline'``.  These warnings do not block DOM updates (the other
        8 tests verify the updates happen) and are therefore excluded from this
        check.  Only genuine JavaScript errors (uncaught exceptions, undefined
        variable references, etc.) are considered failures.
        """
        errors: list[str] = []

        def _collect_error(msg) -> None:  # type: ignore[no-untyped-def]
            if msg.type == "error" and "Content Security Policy" not in msg.text:
                errors.append(msg.text)

        page.on("console", _collect_error)

        mid = _create_running_migration(server)
        page.goto(f"{server}/ui/migrations/{mid}")
        page.locator("#overall-progress").wait_for(state="attached")

        injector = SSEEventInjector(server, mid)
        try:
            injector.wait_for_sse_connection()

            # Emit a full event sequence
            injector.emit(
                {
                    "type": "progress",
                    "overall_progress": 0.4,
                    "current_stage": "gepa_optimization",
                    "eta_seconds": 60,
                    "migration_id": mid,
                }
            )
            # Wait for progress bar to update (confirms the event was processed)
            page.wait_for_function(
                "() => parseFloat(document.getElementById('overall-progress').style.width) >= 40"
            )

            injector.emit(
                {
                    "type": "gepa_iteration",
                    "iteration": 1,
                    "total_iterations": 5,
                    "running_mean_score": 0.75,
                    "migration_id": mid,
                }
            )
            page.wait_for_function(
                "() => document.getElementById('gepa-iteration-counter').textContent.includes('1 / 5')"
            )

            injector.emit(
                {
                    "type": "eval_pair",
                    "pair_index": 1,
                    "total_pairs": 10,
                    "migration_id": mid,
                }
            )
            page.wait_for_function(
                "() => document.getElementById('eval-pair-counter').textContent.trim() === '1/10'"
            )

        finally:
            injector.close()

        assert errors == [], f"Expected zero console errors, got: {errors}"

    # -----------------------------------------------------------------------
    # Test 8: terminal state triggers reload
    # -----------------------------------------------------------------------

    def test_terminal_state_triggers_reload(self, page: Page, server: str) -> None:
        """Emitting a terminal status event causes window.location.reload()."""
        mid = _create_running_migration(server)
        page.goto(f"{server}/ui/migrations/{mid}")
        page.locator("#overall-progress").wait_for(state="attached")

        injector = SSEEventInjector(server, mid)
        try:
            injector.wait_for_sse_connection()

            with page.expect_navigation(timeout=15_000):
                injector.emit(
                    {
                        "type": "progress",
                        "status": "complete",
                        "overall_progress": 1.0,
                        "current_stage": "complete",
                        "migration_id": mid,
                    }
                )
            # If we reach here, navigation (reload) was triggered successfully
        finally:
            injector.close()

    # -----------------------------------------------------------------------
    # Test 9: progress never decreases
    # -----------------------------------------------------------------------

    def test_progress_never_decreases(self, page: Page, server: str) -> None:
        """Progress bar width values are non-decreasing across all emitted events."""
        mid = _create_running_migration(server)
        page.goto(f"{server}/ui/migrations/{mid}")
        page.locator("#overall-progress").wait_for(state="attached")

        injector = SSEEventInjector(server, mid)
        try:
            injector.wait_for_sse_connection()

            # Install MutationObserver to record every style change
            page.evaluate("""
                () => {
                    window.__progressWidths = [];
                    var el = document.getElementById('overall-progress');
                    if (el) {
                        window.__progressWidths.push(parseFloat(el.style.width) || 0);
                        new MutationObserver(function() {
                            window.__progressWidths.push(parseFloat(el.style.width) || 0);
                        }).observe(el, {attributes: true, attributeFilter: ['style']});
                    }
                }
            """)

            # Emit progress events in ascending order
            for progress_val, expected_pct in [
                (0.1, 10),
                (0.3, 30),
                (0.5, 50),
                (0.7, 70),
                (0.9, 90),
            ]:
                injector.emit(
                    {
                        "type": "progress",
                        "overall_progress": progress_val,
                        "current_stage": "gepa_optimization",
                        "migration_id": mid,
                    }
                )
                # Wait for DOM to reflect this specific width before next emit
                page.wait_for_function(
                    f"() => parseFloat(document.getElementById('overall-progress').style.width) >= {expected_pct}"
                )

            # Retrieve all recorded widths
            all_widths: list[float] = page.evaluate("() => window.__progressWidths")

            # Assert non-decreasing — each value must be >= the previous
            for i in range(1, len(all_widths)):
                assert all_widths[i] >= all_widths[i - 1], (
                    f"Progress decreased at index {i}: {all_widths[i - 1]} → {all_widths[i]}. "
                    f"Full sequence: {all_widths}"
                )
        finally:
            injector.close()
