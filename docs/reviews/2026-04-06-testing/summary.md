# Testing Review Summary — 2026-04-06

## Top 5 Highest-Leverage Actions

1. **Write column-level schema parity test (database-persistence).** The existing Alembic parity test only compares table names — a column added to `models.py` but missing from Alembic passes all tests silently. This is a false-confidence test protecting the most dangerous boundary. ~2 hours.

2. **Write IDOR tests and fix source (server-http-security).** Audit log, comparisons, reports, shadow config, and SSE endpoints have zero ownership checks. Any authenticated user can access any migration's data in multi-user mode. Tests will fail → fix source → tests pass. ~1 dev-day.

3. **Add PII adversarial/evasion tests (safety-observability-decision).** The PII scanner has zero tests for Unicode homoglyphs, zero-width character insertion, or space-padded digits. For a safety module that gates production data access, this is the widest gap between responsibility and coverage. ~1.5 days.

4. **Write `ci_output.py` tests (report-generation-cli).** Three public functions that format output for CI/CD pipelines (GitHub PR comments, JSON quality gates) have zero test coverage. A bug here means broken CI silently passing. ~3 hours.

5. **Write A/B runner background tests (background-orchestration).** `run_ab_test_background()` is never called from any test. The entire A/B execution path — simulation, live mode, batch commits, significance computation — has zero coverage. Batch commit partial failure would silently corrupt results. ~3 hours.

## Coordination Points Needing Human Input

- **Empty-string prompt/response:** Should `PromptPairInput` reject `""` as valid? CSV adapter skips empty values, but the schema accepts them. Decision affects ingest, evaluate, and report boundaries.
- **SQLite foreign key enforcement:** Should `PRAGMA foreign_keys=ON` be enabled? Currently off, which means orphaned records pass in SQLite but would fail in Postgres.
- **`_migrate_add_columns` deprecation:** This safety-net migration code is a second source of truth alongside Alembic. Should it be removed now that Alembic covers all schema changes?
- **`prompt_auditor.py` scope:** Described as "injection detection" but only performs training data leakage detection. Is a separate injection detection module planned?
- **`majority_label` tie-breaking:** Undefined behavior when vote counts are equal. Should be explicitly defined and tested.
- **Credit card regex false positives:** No Luhn validation. Acceptable for production, or should Presidio be required?
- **GEPA background thread lifetime:** `executor.shutdown(wait=False)` means the optimization thread is never killed after timeout. Accepted limitation or needs a kill mechanism?

## Report Reading Order

1. **server-http-security.md** — Real security bugs (IDOR, JWT default secret, CORS). Read first.
2. **database-persistence.md** — False-confidence test exposing schema drift risk. Read second.
3. **background-orchestration.md** — Silent data corruption in A/B batch commits. Production resilience gaps.
4. **safety-observability-decision.md** — PII evasion, Wilson CI single test case, calibration gaps.
5. **evaluation-strategies.md** — Fallback chain degradation, JSON fence stripping for non-OpenAI models.
6. **ingest-adapters.md** — Redis mixed-format detection, OTel nondeterminism.
7. **optimization-engines.md** — DSPy version coupling, cost tracking absence.
8. **report-generation-cli.md** — CI output gaps, CLI command coverage.

Consolidation documents:
- **path-to-production.md** — The headline deliverable. Sequenced plan from current state to production.
- **synthetic-data-strategy.md** — Unified test data approach across all boundaries.
