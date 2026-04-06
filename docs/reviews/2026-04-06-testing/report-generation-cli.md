# Report Generation & CLI — Testing Review

**Scope:** `src/rosettastone/report/`, `src/rosettastone/cli/`, server-side Jinja2 templates, and CLI E2E.
**Date:** 2026-04-06
**Author:** Testing Lead (Report Generation & CLI)

---

## 1. Boundary Map

### Inside (we test it)

| Component | Path |
|-----------|------|
| Markdown report generator | `report/markdown.py` |
| HTML report generator | `report/html_generator.py` |
| PDF report generator | `report/pdf_generator.py` |
| Executive narrative generator | `report/narrative.py` |
| Executive prompt builder | `report/executive_prompt.py` |
| Markdown report template | `report/templates/report.md.jinja` |
| HTML report template | `report/templates/report.html.jinja` |
| Executive summary template | `report/templates/executive.md.jinja` |
| CLI commands (Typer app) | `cli/main.py` |
| Rich display components | `cli/display.py` |
| CI output formatters | `cli/ci_output.py` |
| `_stats_to_dict` helper (3 copies) | `report/markdown.py`, `html_generator.py`, `pdf_generator.py` |
| `_build_sample_comparisons` | `report/markdown.py` |

### Outside (we mock/stub)

| Dependency | Strategy |
|------------|----------|
| **Jinja2** (template engine) | Real — it is pure computation, fast, deterministic. No mock. |
| **weasyprint** (PDF rendering) | Mock — optional dep with system-level C libraries. `patch.dict("sys.modules")` to test import guard and call shape. |
| **LiteLLM** (narrative LLM call) | Mock — external API call. Mock `litellm.completion` to test LLM path and fallback path. |
| **Migrator / pipeline** | Mock — CLI tests mock `Migrator` class entirely. |
| **uvicorn** | Mock — `serve` command tests mock `uvicorn.run`. |
| **File system** | Real via `tmp_path` — pytest fixtures handle cleanup. |
| **Rich Console** | Real with `StringIO` capture — deterministic output testing. |
| **Chart.js bundle** | Real file read — `_CHARTJS_PATH` reads from `server/static/js/chart.min.js`. |
| **FastAPI server** (Playwright) | Real subprocess — session-scoped server fixture. |
| **Chromium** (Playwright) | Real headless browser via Playwright. |

### On the fence

| Item | Issue |
|------|-------|
| **Server templates** (`server/templates/*.html`) | Rendered by FastAPI routes, tested via Playwright. Not directly tested as Jinja2 templates with unit tests. Template correctness depends on E2E. |
| **`_CHARTJS_PATH`** resolution in `html_generator.py` | Uses a hardcoded relative path that walks from `report/` to `server/static/js/`. Fragile across installs. |
| **`MigrationResult` construction** in test fixtures | Duplicated across 6+ test files with slight variations. Not a shared conftest fixture. |

### Dependency diagram

```
CLI (Typer app)
  |
  +-- migrate ------> Migrator.run() [MOCKED]
  |     |
  |     +-- display.py (Rich tables/panels)
  |     +-- report/markdown.py ---------> Jinja2 (report.md.jinja)
  |     +-- report/html_generator.py ----> Jinja2 (report.html.jinja) + Chart.js
  |     +-- report/pdf_generator.py -----> Jinja2 (report.html.jinja) + weasyprint [MOCKED]
  |     +-- report/narrative.py ---------> LiteLLM [MOCKED] | Jinja2 (executive.md.jinja)
  |           +-- report/executive_prompt.py (prompt formatting)
  |
  +-- preflight ----> Migrator.run() [MOCKED]
  +-- evaluate -----> pipeline.load_and_split_data + evaluate_baseline [MOCKED]
  +-- batch --------> batch.load_manifest + run_batch [UNTESTED in CLI tests]
  +-- ci-report ----> ci_output.py (format_ci_json, format_pr_comment, format_quality_diff) [UNTESTED]
  +-- serve --------> uvicorn.run [MOCKED]
  +-- score-shadow -> shadow.evaluator [UNTESTED in CLI tests]
  +-- calibrate ----> calibration.calibrator [UNTESTED in CLI tests]

Server templates (server/templates/*.html)
  |
  +-- Tested only via Playwright E2E (test_playwright_ui.py)
```

---

## 2. Current Coverage Audit

### `tests/test_report/test_markdown.py`

**Covers:**
- File creation (markdown + optimized_prompt.txt)
- Output dir auto-creation
- Empty results rendering
- Optimized prompt content written correctly
- GO / NO_GO / CONDITIONAL recommendation rendering
- Per-type table presence and values
- Safety warnings (empty and populated)
- Cost summary section
- Worst regressions section

**Misses:**
- `_build_sample_comparisons` is not directly unit-tested (only indirectly via full report render)
- `_stats_to_dict` is not tested in the markdown module (tested only in `test_pdf_generator.py`)
- No test for config dict vs Pydantic model as `result.config`
- No test for very long optimized prompts (truncation at 600 chars in template)
- No test for `cost_breakdown` section (phased cost breakdown)
- No test for `stage_timing` / Pipeline Timing (covered in `test_report_rendering.py` instead)
- No negative test for malformed `per_type_scores` values (e.g. missing keys)
- No test for the `variance_flag_threshold` config access pattern: `result.config.get("variance_flag_threshold", 0.1)` -- will fail if `result.config` is a Pydantic model, not a dict

**Brittle tests:** None identified. Tests use content assertions against generated markdown which is reasonable.

**False-confidence risks:**
- `test_report_renders_go_recommendation` asserts `"GO" in content` -- this matches the word "GO" anywhere, including "NO_GO". Assertion is too loose (though it passes because "GO" does appear in the badge). Not a false positive today, but fragile.

### `tests/test_report/test_html_generator.py`

**Covers:**
- HTML file creation and naming
- Structural validity (DOCTYPE, html/head/body/style tags)
- Key sections present (Executive Summary, Recommendation, Key Metrics, Score Distribution, Per-Case Results)
- Model names in output
- Formatted metric values
- Warnings in output
- PII not in output (prompt/response content excluded) -- strong assertion
- Empty results handling
- Output directory creation
- Chart.js presence
- Color coding classes

**Misses:**
- No test for `per_type_scores` rendering in HTML
- No test for `recommendation_reasoning` rendering
- No test for `prompt_regressions` / regression analysis section
- No test for `non_deterministic_count` / eval reliability section
- No test for `cost_breakdown` rendering
- No test for `embed_mode` vs non-embed rendering differences
- No test for Chart.js fallback when `chart.min.js` is missing
- No test for `autoescape=True` XSS protection (e.g. `<script>` in warning text)

**Dead tests:** None.

### `tests/test_report/test_narrative.py`

**Covers:**
- `_format_per_type`: empty, dict stats, dataclass stats, unknown types
- `_format_safety`: empty, dict, object, string, mixed warnings
- `generate_executive_narrative`: local_only mode, metrics in output, empty results, LLM failure fallback
- `_basic_summary`: readable output with all fields, zero results, missing config fields
- `EXECUTIVE_PROMPT`: no-PII instruction, placeholder fields

**Misses:**
- No test for successful LLM path (litellm.completion returning a valid response)
- No test for `format_executive_prompt` integration with `generate_executive_narrative` (it calls `executive_prompt.format_executive_prompt` internally but that code path is only tested when LLM is not mocked away)
- No test for `_config_get` helper with Pydantic model vs dict
- No test for `_template_fallback` when template file is missing (falls back to `_basic_summary`)
- The LLM fallback test (`test_llm_failure_falls_back_to_template`) patches `sys.modules` which is fragile -- it replaces the entire module dict entry, which can cause import side effects

### `tests/test_report/test_pdf_generator.py`

**Covers:**
- `_stats_to_dict`: dict passthrough, dataclass conversion, unknown types
- ImportError when weasyprint missing
- Mock weasyprint call shape verification
- Output directory creation
- PII not in generated HTML

**Misses:**
- No actual PDF generation test (requires weasyprint installed)
- No test that the PDF output is a valid PDF file
- No test for `embed_mode=True` being passed to template
- The weasyprint mock test verifies call shape but not that the HTML content is correct for PDF layout

### `tests/test_report/test_executive_prompt.py`

**Covers:**
- `format_executive_prompt` returns correct message list structure
- Message list length matches system + few-shot + user
- `SYSTEM_PROMPT` contains no-raw-content instruction
- Few-shot examples: count, scenarios cover GO/CONDITIONAL/NO_GO, no PII patterns, required keys
- `_format_per_type_block`: dict data, empty, dataclass, missing CI
- `_format_safety_block`: dict, object, string, empty warnings
- `_format_warnings_block`: populated and empty
- User prompt content integrity: model names, metrics, no-raw-content reminder
- Zero test cases (no division error)

**Misses:**
- No test for malformed `confidence_interval` (e.g. tuple of 1 element, or None)
- No test for very large `per_type_scores` dict (10+ output types)

**Strong tests.** This is the best-covered module in the scope.

### `tests/test_report/test_report_rendering.py`

**Covers:**
- Pipeline Timing section: renders when populated, absent when empty or default
- 95% CI column: valid CI values, dash when (0.0, 0.0)
- Evaluation Reliability section: renders when `eval_runs > 1` or `non_deterministic_count > 0`, absent at baseline
- Regression section with metric deltas: significant deltas shown, small deltas filtered
- Regression section with empty metric_deltas: shows dash
- Sample Improvements: Top Metric column populated, dash when no scores
- Skipped Pairs section: renders with failure_reason, absent without

**Misses:**
- No test for Non-Deterministic Prompts section (requires `is_non_deterministic` attribute on EvalResult and `score_std`/`run_scores` -- these are template features tested only by assertion of section header)

**Strong tests.** Good coverage of observability features.

### `tests/test_cli/test_display.py`

**Covers:**
- `_score_color`: all threshold boundaries (green/yellow/red)
- `MigrationDisplay.__init__`: default and custom console
- `create_progress`: returns Progress, uses same console
- `show_summary_table`: output type rendering, sample count, color thresholds (green/yellow/red), empty inputs, Overall row for multi-type, missing threshold key
- `show_recommendation`: GO/NO_GO/CONDITIONAL text, case insensitivity, unknown recommendation
- `show_cost_summary`: phase names, Total row, sum calculation, empty costs, decimal formatting
- `show_safety_warnings`: empty list (no output), plain string, dict with HIGH/MEDIUM/LOW, object with attributes, multiple warnings, panel title, case-insensitive severity
- Smoke test: all methods called in sequence

**Misses:**
- `show_timing_table` -- **zero tests**
- `show_prompt_evolution` -- **zero tests**
- `show_variance_warning` -- **zero tests**
- No test for CI column rendering (confidence_interval in per_type stats)

### `tests/test_cli/test_commands.py`

**Covers:**
- `migrate`: calls Migrator.run(), output contains scores, warnings printing, no warnings when empty, config construction (source/target/data_path), dry-run flag
- `preflight`: calls Migrator with dry_run=True, displays warnings
- `evaluate`: calls load_and_split_data + evaluate_baseline, win rate calculation, skip_preflight set
- Error cases: missing --data, --from, --to
- `--from` keyword alias works
- `serve`: registered, default options, custom host/port, reload flag, app factory reference
- `--lm-extra-kwargs`: valid JSON, multiple keys, invalid JSON, omitted
- `--gepa-timeout-seconds`: set, different value, omitted

**Misses:**
- `batch` command -- **zero tests**
- `ci-report` command -- **zero tests**
- `score-shadow` command -- **zero tests**
- `calibrate` command -- **zero tests**
- `--pipeline` flag -- **zero tests**
- `--adapter`, `--pii-engine`, `--cluster-prompts`, `--csv-*`, `--braintrust-*`, `--langsmith-*`, `--otel-path`, `--improvement-objectives`, `--num-threads` flags -- **zero tests**
- No test for `--auto` flag (GEPA intensity)
- No test for `--local-only` flag
- No test for `--no-pii-scan`, `--no-prompt-audit` flags
- No test for `--optimizer` mipro mode
- No test for `--judge-model`, `--reflection-model` flags

### `tests/test_cli/` -- MISSING file: `test_ci_output.py`

`ci_output.py` has **zero test coverage**. Three public functions (`format_ci_json`, `format_pr_comment`, `format_quality_diff`) are completely untested.

### `tests/test_e2e/test_playwright_ui.py`

**Covers (in scope):**
- Server template rendering for models dashboard, migrations list, migration detail pages
- Navigation, badges, collapsible sections, dark mode
- Executive report page rendering

**Misses (in scope):**
- No Playwright test for the report file generators (markdown/HTML/PDF) -- these are offline generators, not served pages
- Template rendering for 404/500 error pages is not tested
- No test for the diff slideout fragment

**Notes:**
- Playwright tests are gated behind `@pytest.mark.playwright` -- they only run when Playwright and Chromium are available
- Session-scoped server fixture kills port 8765 before starting -- CI portability concern (requires `lsof`)

---

## 3. Risk Ranking

| # | Failure Mode | Manifestation | Existing Tests Catch It? | Severity |
|---|-------------|---------------|--------------------------|----------|
| 1 | `ci_output.py` produces malformed CI JSON or broken PR comment markdown | CI/CD pipeline fails silently, broken GitHub PR comments, bad quality gates | **No -- zero tests** | **CRITICAL** |
| 2 | `result.config` is a Pydantic model instead of dict; `.get()` calls in templates fail | `AttributeError` at report generation time; users see crash instead of report | **No** -- all test fixtures use plain dicts for config | **CRITICAL** |
| 3 | `batch`, `ci-report`, `score-shadow`, `calibrate` CLI commands crash on invocation | Users hit unhandled exceptions from untested code paths | **No -- zero CLI tests** | **HIGH** |
| 4 | `show_timing_table`, `show_prompt_evolution`, `show_variance_warning` produce garbled Rich output | Unreadable CLI output for timing data, prompt evolution, and variance warnings | **No -- zero tests** | **HIGH** |
| 5 | HTML template XSS vulnerability: user-controlled warning text with `<script>` tags | Jinja2 autoescape is on for HTML but not tested; if disabled, XSS in self-hosted reports | **No** | **HIGH** |
| 6 | `_CHARTJS_PATH` resolution fails in installed package (non-editable install) | HTML report renders without charts; silent degradation | **No** -- only tested implicitly via `"chart.js" in content.lower()` | **MEDIUM** |
| 7 | PDF report generator produces invalid/empty PDF when weasyprint is actually installed | PDF file exists but is corrupted or blank | **No** -- weasyprint is always mocked | **MEDIUM** |
| 8 | Narrative LLM call succeeds but returns empty/None content | Falls back to template, but the log message wrongly says "using LLM" before falling back | **Partial** -- test covers exception fallback but not empty-response fallback | **MEDIUM** |
| 9 | Template rendering with 100+ validation results causes huge report / OOM | Report file grows unbounded; no pagination or truncation for per-case section | **No** | **LOW** |
| 10 | `_build_sample_comparisons` crashes when `baseline_results` and `validation_results` have different lengths | `zip()` silently truncates; score comparison is wrong | **No** | **LOW** |

---

## 4. Test Plan by Tier

### Unit Tests

| Test | Target | Assertions | Write Time | Status |
|------|--------|------------|------------|--------|
| `test_format_ci_json_structure` | `ci_output.format_ci_json` | Valid JSON, all expected keys present, floats rounded to 4dp | S | **MISSING** |
| `test_format_ci_json_with_all_fields` | `ci_output.format_ci_json` | safety_warnings, per_type_scores serialized correctly | S | **MISSING** |
| `test_format_pr_comment_markdown_valid` | `ci_output.format_pr_comment` | Contains header, source/target, recommendation emoji, scores table, cost | S | **MISSING** |
| `test_format_pr_comment_with_warnings` | `ci_output.format_pr_comment` | Warnings section present when warnings exist | S | **MISSING** |
| `test_format_pr_comment_with_safety` | `ci_output.format_pr_comment` | Safety section present when safety_warnings exist | S | **MISSING** |
| `test_format_quality_diff_current_only` | `ci_output.format_quality_diff` | Shows current scores table when baseline is None | S | **MISSING** |
| `test_format_quality_diff_with_baseline` | `ci_output.format_quality_diff` | Shows comparison table with deltas when baseline provided | S | **MISSING** |
| `test_format_quality_diff_per_type` | `ci_output.format_quality_diff` | Per-type breakdown appears when per_type_scores populated | S | **MISSING** |
| `test_show_timing_table_renders_stages` | `display.show_timing_table` | Stage names, durations, percentage shares visible | S | **MISSING** |
| `test_show_timing_table_empty_skips` | `display.show_timing_table` | No output when stage_timing is empty | S | **MISSING** |
| `test_show_timing_table_single_stage` | `display.show_timing_table` | Works with single stage (100% share) | S | **MISSING** |
| `test_show_prompt_evolution_renders` | `display.show_prompt_evolution` | Before/After scores, delta, prompt panel, truncation at 400 chars | M | **MISSING** |
| `test_show_prompt_evolution_with_comparisons` | `display.show_prompt_evolution` | Top improvements table rendered with sample_comparisons | M | **MISSING** |
| `test_show_prompt_evolution_negative_improvement` | `display.show_prompt_evolution` | Negative delta shows red styling, minus sign | S | **MISSING** |
| `test_show_variance_warning_renders` | `display.show_variance_warning` | Panel text with count, singular/plural | S | **MISSING** |
| `test_show_variance_warning_zero_skips` | `display.show_variance_warning` | No output when count is 0 or negative | S | **MISSING** |
| `test_build_sample_comparisons_top_n` | `markdown._build_sample_comparisons` | Returns top N by delta, sorted descending | S | **MISSING** |
| `test_build_sample_comparisons_empty` | `markdown._build_sample_comparisons` | Returns empty list for empty inputs | S | **MISSING** |
| `test_build_sample_comparisons_mismatched_lengths` | `markdown._build_sample_comparisons` | Handles len(baseline) != len(validation) via zip truncation | S | **MISSING** |
| `test_stats_to_dict_shared_behavior` | All three `_stats_to_dict` copies | Verify identical behavior (or refactor to shared function) | S | **PARTIAL** (only pdf_generator tested) |
| `test_html_autoescape_xss` | `html_generator.generate_html_report` | Warning text with `<script>alert(1)</script>` is escaped in output | S | **MISSING** |
| `test_config_get_pydantic_vs_dict` | `narrative._config_get` | Works with both dict and Pydantic BaseModel | S | **MISSING** |
| `test_recommendation_emoji_mapping` | `ci_output._RECOMMENDATION_EMOJI` | All 4 keys (GO, CONDITIONAL, NO_GO, None) map to emoji strings | S | **MISSING** |

### Contract Tests

| Test | Target | Assertions | Write Time | Status |
|------|--------|------------|------------|--------|
| `test_all_generators_accept_migration_result` | `generate_markdown_report`, `generate_html_report`, `generate_pdf_report` | All accept the same `MigrationResult` shape and produce a `Path` | M | **MISSING** |
| `test_generators_share_template_variables` | All 3 generators | The set of context variables passed to templates is consistent across markdown/html/pdf | M | **MISSING** |
| `test_narrative_returns_string` | `generate_executive_narrative` | Always returns `str`, never `None`, for both LLM and fallback paths | S | **EXISTS** |
| `test_ci_output_functions_return_string` | `format_ci_json`, `format_pr_comment`, `format_quality_diff` | All return `str` | S | **MISSING** |

### Integration Tests

| Test | Target | Assertions | Write Time | Status |
|------|--------|------------|------------|--------|
| `test_full_markdown_report_pipeline` | `generate_markdown_report` with realistic MigrationResult | Report contains all sections, no Jinja2 errors, all optional sections conditional | M | **PARTIAL** (exists but doesn't cover all sections) |
| `test_full_html_report_pipeline` | `generate_html_report` with realistic MigrationResult | Valid HTML with all sections, chart data embedded | M | **PARTIAL** |
| `test_cli_migrate_e2e_subprocess` | `rosettastone migrate --dry-run` via `subprocess.run` | Exit code 0, output contains expected text | L | **MISSING** |
| `test_cli_ci_report_json_format` | `rosettastone ci-report --format json` via CliRunner | Valid JSON output with correct structure | M | **MISSING** |
| `test_cli_ci_report_pr_comment_format` | `rosettastone ci-report --format pr-comment` | Valid markdown with all sections | M | **MISSING** |
| `test_cli_ci_report_quality_diff_format` | `rosettastone ci-report --format quality-diff` | Valid markdown quality report | M | **MISSING** |
| `test_cli_batch_command` | `rosettastone batch --manifest` via CliRunner | Calls load_manifest + run_batch, prints summary | M | **MISSING** |
| `test_cli_serve_wildcard_host_warning` | `serve --host 0.0.0.0` | Prints warning about network exposure | S | **MISSING** |

### Property-Based Tests

| Test | Target | Assertions | Write Time | Status |
|------|--------|------------|------------|--------|
| `test_markdown_template_never_crashes` | `generate_markdown_report` | Hypothesis-generated MigrationResult with varied shapes (empty lists, None optionals, extreme floats) never raises | L | **MISSING** |
| `test_html_template_never_crashes` | `generate_html_report` | Same strategy as above for HTML | L | **MISSING** |
| `test_ci_json_always_valid` | `format_ci_json` | Any valid MigrationResult produces parseable JSON | M | **MISSING** |
| `test_pr_comment_no_unclosed_markdown` | `format_pr_comment` | Output has balanced markdown table delimiters | M | **MISSING** |
| `test_score_color_covers_all_floats` | `_score_color` | For any float 0.0-1.0, returns one of green/yellow/red | S | **MISSING** |

### End-to-End Tests

| Test | Target | Assertions | Write Time | Status |
|------|--------|------------|------------|--------|
| Playwright: migration detail executive report | `/ui/migrations/{id}/executive` | Page loads, contains recommendation, metrics, no raw content | M | **EXISTS** (in test_playwright_ui.py) |
| Playwright: 404 page template | `/ui/nonexistent` | Returns 404, renders error template | S | **MISSING** |
| Playwright: 500 error handling | Forced server error | Returns 500, renders error template | M | **MISSING** |
| CLI subprocess: `migrate --help` | Exit code 0, all flags documented | S | **MISSING** |
| CLI subprocess: `ci-report --help` | Exit code 0, format options listed | S | **MISSING** |

---

## 5. Synthetic Data Generation Strategy

### MigrationResult Factory

Create a `tests/fixtures/migration_result_factory.py` with a builder that:

1. Accepts overrides for any field
2. Generates internally consistent data (e.g., `confidence_score` matches `validation_results` win rate)
3. Supports presets: `minimal`, `go_recommendation`, `no_go_recommendation`, `conditional`, `with_regressions`, `with_safety_warnings`, `large_dataset`

```
Factory presets:
- minimal: 0 validation results, all zeros
- happy_path: 50 results, 92% win rate, GO recommendation
- mixed: 50 results, 78% win rate, CONDITIONAL, 2 regressions
- no_go: 30 results, 55% win rate, HIGH safety warning
- large: 500 results, diverse output types (stress test for templates)
```

### EvalResult Builder

Create an `EvalResult` builder that generates realistic score distributions:
- Output type (JSON/SHORT_TEXT/LONG_TEXT/CODE/CLASSIFICATION)
- Score distribution centered at a configurable mean with configurable spread
- `is_win` derived from composite_score vs threshold
- Realistic `scores` dict with 2-4 metric keys

### Stability and Rot Prevention

- Factory lives in `tests/fixtures/` and is imported by all test files -- no more per-file `_base_result()` / `_make_result()` duplication
- Factory validates its own output against `MigrationResult` model -- catches schema drift immediately
- Pin factory presets as snapshot tests: generate once, assert structure matches expectation
- Cost: ~M to build, saves time on every future test

---

## 6. Fixtures, Fakes, and Mocks

### New Fixtures Needed

| Fixture | Location | Description | Shared? |
|---------|----------|-------------|---------|
| `migration_result_factory` | `tests/fixtures/migration_result_factory.py` | Builder for `MigrationResult` with presets | **Cross-subagent** -- usable by server, API, and pipeline tests |
| `eval_result_builder` | `tests/fixtures/migration_result_factory.py` | Builder for `EvalResult` with configurable scores | **Cross-subagent** |
| `prompt_regression_builder` | `tests/fixtures/migration_result_factory.py` | Builder for `PromptRegression` | **Cross-subagent** |
| `captured_console` | `tests/test_cli/conftest.py` | Returns `(Console, StringIO)` pair for CLI output capture | CLI-local |
| `mock_migrator` | `tests/test_cli/conftest.py` | Pre-configured `MagicMock` for `Migrator` with `_make_migration_result` defaults | CLI-local |
| `tmp_result_json` | `tests/test_cli/conftest.py` | Writes a valid `MigrationResult` JSON to `tmp_path` for `ci-report` tests | CLI-local |

### Existing Fixtures to Refactor

The following per-file helpers should be replaced by the shared factory:
- `test_markdown.py::_base_result`
- `test_html_generator.py::_make_result`
- `test_narrative.py::_make_result`
- `test_pdf_generator.py::_make_result`
- `test_report_rendering.py::_base_result`
- `test_commands.py::_make_migration_result`

### Fakes

| Fake | Purpose |
|------|---------|
| `FakeWeasyprint` | A minimal fake that writes a known byte string to the output path, for integration tests that want to verify the PDF file exists without installing weasyprint |

### Mocks (existing, keep)

- `patch("rosettastone.core.migrator.Migrator")` -- CLI command tests
- `patch.dict("sys.modules", {"weasyprint": ...})` -- PDF import guard
- `patch.dict("sys.modules", {"litellm": ...})` -- narrative LLM fallback
- `patch("uvicorn.run")` -- serve command

### Cross-Subagent Shared Fixtures Flag

The `migration_result_factory` is the highest-value shared fixture. The following test suites outside this scope also construct `MigrationResult` by hand:
- `tests/test_server/test_api_migrations.py`
- `tests/test_server/test_api_comparisons.py`
- `tests/test_server/test_negative_stress.py`

Coordinate with the Server API testing lead to use a shared factory.

---

## 7. Gaps You Can't Close

| Gap | Why |
|-----|-----|
| **Actual PDF output validation** | Requires `weasyprint` installed with system-level dependencies (Pango, Cairo, GDK-PixBuf). Cannot be reliably tested in CI without a dedicated Docker image. Mark as integration-only with `@pytest.mark.requires_weasyprint`. |
| **Chart.js visual correctness** | Chart rendering happens in a browser's Canvas API. Asserting that the bar chart looks right requires screenshot comparison or Canvas introspection. Out of scope for unit/integration tests. Playwright visual comparison could cover this but adds flakiness. |
| **LLM narrative quality** | The executive narrative LLM path calls GPT-4o. Testing that the returned text is "good" requires human judgment or an LLM-as-judge loop, which adds cost and non-determinism. Can only test the call shape and fallback behavior. |
| **Playwright CI portability** | The `_kill_port` function uses `lsof` (macOS/Linux). Windows CI would fail. The session-scoped server fixture assumes port 8765 is free. **NEEDS_HUMAN_REVIEW:** decide if Playwright tests are CI-required or dev-only. |
| **Template visual regression** | Markdown and HTML templates evolve; asserting specific strings like `"## Pipeline Timing"` is fragile to heading changes. No way to test "does the report look right" without visual snapshot testing. |

---

## 8. Cost and Time Estimate

| Tier | Test Count | Write Time | Effort |
|------|-----------|------------|--------|
| **Unit: ci_output.py** (CRITICAL gap) | 8 tests | S each | 2-3 hours |
| **Unit: display.py missing methods** | 8 tests | S each | 2-3 hours |
| **Unit: _build_sample_comparisons** | 3 tests | S each | 1 hour |
| **Unit: XSS / autoescape** | 1 test | S | 30 min |
| **Contract: generator interface** | 2 tests | M each | 2 hours |
| **Integration: CLI ci-report/batch** | 5 tests | M each | 3-4 hours |
| **Integration: full report pipeline** | 2 tests | M each | 2 hours |
| **Property-based: template fuzzing** | 3 tests | L each | 4-5 hours |
| **Shared factory + fixture refactor** | 1 module | M | 3-4 hours |
| **Playwright: 404/500 pages** | 2 tests | S each | 1 hour |
| **Total** | ~34 tests | | **~20-24 hours** |

Priority order for maximum risk reduction:
1. `ci_output.py` unit tests (2-3h) -- closes CRITICAL gap
2. Display missing method tests (2-3h) -- closes HIGH gap
3. CLI `ci-report` / `batch` integration (3-4h) -- closes HIGH gap
4. Shared factory (3-4h) -- reduces future test authoring time
5. Everything else

---

## 9. Path to Production

### Current Readiness Level

**Report generation: 65%.** Core markdown and HTML report generation is well-tested. Template rendering for all major sections (recommendations, per-type, regressions, timing) has solid coverage via `test_report_rendering.py`. PDF is tested at the interface level with mocked weasyprint. Executive prompt is the best-tested module.

**CLI: 50%.** `migrate`, `preflight`, `evaluate`, and `serve` commands are tested. But 4 commands (`batch`, `ci-report`, `score-shadow`, `calibrate`) have zero CLI test coverage. Many flags are untested. The `ci_output.py` module has zero tests -- this is the single biggest gap for production CI/CD workflows.

**Display: 70%.** Most Rich display methods are well-tested, but 3 methods (`show_timing_table`, `show_prompt_evolution`, `show_variance_warning`) have zero tests despite being called in the `migrate` command's output path.

### Gap to Production-Hardened

1. **CRITICAL: `ci_output.py` has zero tests.** This module formats output for CI/CD pipelines (GitHub PR comments, JSON quality gates). If it produces malformed output, CI pipelines silently pass or break. This is the #1 blocker.

2. **HIGH: 4 CLI commands untested.** `batch`, `ci-report`, `score-shadow`, `calibrate` could crash on basic invocation. At minimum need smoke tests.

3. **HIGH: Display methods untested.** `show_timing_table`, `show_prompt_evolution`, `show_variance_warning` could produce garbled output.

4. **MEDIUM: Fixture duplication.** 6+ files duplicate `_make_result()` / `_base_result()` with subtle differences. This causes drift and makes tests fragile to `MigrationResult` schema changes.

5. **MEDIUM: `config` dict vs Pydantic model ambiguity.** Templates call `config.get(...)` which works for dicts but fails for Pydantic models. No test catches this. **NEEDS_HUMAN_REVIEW:** is `MigrationResult.config` always a dict in production, or can it be a Pydantic model?

6. **LOW: Playwright CI portability.** `lsof`-based port cleanup is macOS/Linux only. Need to decide on CI strategy.

### Gates (Ordered Sequence)

```
Gate 1: ci_output unit tests pass                    [blocks CI/CD integration]
Gate 2: CLI smoke tests for all 8 commands           [blocks release]
Gate 3: Display method tests complete                [blocks CLI quality]
Gate 4: Shared fixture factory merged                [blocks efficient test authoring]
Gate 5: Property-based template fuzzing passes       [blocks confidence in edge cases]
Gate 6: Playwright 404/500 tests pass                [blocks error handling confidence]
```

### Smallest Next Slice

Write `tests/test_cli/test_ci_output.py` with 8 unit tests for `format_ci_json`, `format_pr_comment`, and `format_quality_diff`. This closes the #1 CRITICAL gap in ~2-3 hours and requires no new fixtures (just construct `MigrationResult` inline). No external dependencies, no mocks needed, fully deterministic.

### Dependencies on Other Subagents

- **Server API testing lead:** Coordinate on shared `migration_result_factory` fixture
- **Evaluate/Pipeline testing lead:** `_build_sample_comparisons` consumes `EvalResult` objects -- ensure eval result fixtures are compatible
- **Infrastructure:** Playwright CI runner needs Chromium and port availability
