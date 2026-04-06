# Path to Production — Testing Strategy

## Current State Snapshot

| Boundary | Readiness | Tests | Key Gap |
|---|---|---|---|
| Server HTTP & Security | **Not production-ready** | ~200+ | IDOR on 5 endpoint groups, JWT default secret, CORS |
| Database & Persistence | **Alpha** | ~20 | Schema parity test is false-confidence (names only, not columns) |
| Background Orchestration | **Beta (partial)** | ~40 | A/B runner background path completely untested, crash recovery gaps |
| Safety & Observability | **Mixed (Red to Green)** | ~150 | PII evasion untested, Wilson CI single case, collector zero tests |
| Evaluation Strategies | **Beta (3/5)** | ~175 | Fallback chain silent degradation, JSON fence stripping for CoT models |
| Ingest Adapters | **Beta** | ~139 | Redis mixed-format auto-detect, OTel nondeterminism |
| Optimization Engines | **Level 2/5** | ~130 | No real LLM test, cost tracking absent, DSPy version coupling |
| Report Generation & CLI | **50-70%** | ~37 | ci_output.py zero tests, 4 CLI commands untested |

**Total existing tests:** ~900+ across all boundaries (1663 full suite including overlaps).
**CRITICAL risks identified:** 8 across all boundaries.
**HIGH risks identified:** 18 across all boundaries.
**Estimated total effort to production-hardened:** 160-220 hours (~20-28 person-days).

## Production Readiness Definition

For RosettaStone specifically, "production-ready" means:

1. **No silent bad data.** A migration recommendation (GO/NO_GO/CONDITIONAL) must be based on correctly computed scores, correctly applied thresholds, and correctly routed evaluation strategies. If any step fails, the failure must be visible.

2. **No data leakage across users.** In multi-user mode, user A cannot access user B's migrations, test cases, reports, or audit entries through any API endpoint.

3. **No PII in outputs.** Prompt content (which may contain PII) must never appear in logs, metrics, or error messages. The PII scanner must catch standard patterns and resist basic evasion.

4. **Schema changes don't break running deployments.** Alembic migrations must preserve existing data. The schema defined in code must match the schema applied by Alembic.

5. **Long-running operations survive interruption.** A 45-minute migration can be checkpointed, resumed after server restart, and produce the same result.

6. **Cost is bounded.** Optimization runs have enforced cost caps. Users know what they'll spend before they start.

7. **CI catches regressions.** Every PR runs fast tests (<60s). Slow integration tests run nightly or on-demand. API contract changes are detected automatically.

## Gate Structure

### Stage 0: Safety Net (eliminate false confidence)

**Entry:** Current state.
**Exit:** All false-confidence tests fixed, CRITICAL security issues have failing tests that expose them.

| Work Item | Boundary | Effort | Risk Addressed |
|---|---|---|---|
| Fix schema parity test to compare columns, types, and constraints | database | 2h | Schema drift undetected |
| Write IDOR tests for audit-log, comparisons, reports, shadow, SSE | server | 4h | Cross-user data access |
| Fix 4 false-confidence tests in prompt_auditor and pii_scanner | safety | 4h | Assertions that always pass |
| Write JWT default secret transition test | server | 2h | Auth bypass on config change |

**Total: ~12 hours. Produces immediately failing tests that expose real bugs.**

### Stage 1: Contract Coverage

**Entry:** Stage 0 complete, false-confidence tests fixed.
**Exit:** Every pluggable interface has contract tests. API schema snapshots exist.

| Work Item | Boundary | Effort |
|---|---|---|
| Column-level Alembic schema parity (C1-C4) | database | 6h |
| API schema snapshot tests for top 10 endpoints | server | 8h |
| Evaluator key ↔ METRIC_WEIGHTS alignment test | evaluation | 4h |
| DataAdapter contract tests (output shape, empty source) | ingest | 4h |
| Report generator contract tests (accept same MigrationResult) | report-cli | 4h |
| Shared `migration_result_factory` to replace 6+ duplicates | shared | 3h |

**Total: ~29 hours.**

### Stage 2: Integration Coverage

**Entry:** Stage 1 complete, contracts verified.
**Exit:** Every external service boundary has at least one integration test against a real local instance or a contract-verified fake.

| Work Item | Boundary | Effort |
|---|---|---|
| Redis adapter integration test with Docker + mixed formats | ingest | 6h |
| Alembic upgrade/downgrade with data preservation | database | 4h |
| Full pipeline integration test with mocked LLM | orchestration | 6h |
| A/B runner simulation mode end-to-end | orchestration | 4h |
| Evaluation fallback chain (BERTScore→embedding→exact_match) | evaluation | 4h |
| Playwright tests running in CI (fix hardcoded paths) | server | 4h |
| Presidio real analyzer integration | safety | 3h |
| VCR cassette infrastructure + first 3 cassettes | shared | 6h |
| ci_output.py unit tests (8 tests) | report-cli | 3h |
| Missing CLI command smoke tests (batch, ci-report, score-shadow, calibrate) | report-cli | 3h |
| Missing display method tests (show_timing_table, show_prompt_evolution, show_variance_warning) | report-cli | 3h |

**Total: ~46 hours.**

### Stage 3: Property-Based & Statistical Hardening

**Entry:** Stage 2 complete, integration tests passing.
**Exit:** Mathematical functions have property-based tests. Statistical decisions are verified against invariants.

| Work Item | Boundary | Effort |
|---|---|---|
| Wilson CI property tests (invariants, boundary values, monotonicity) | evaluation/decision | 4h |
| Splitter dedup/split invariants (Hypothesis) | ingest | 3h |
| JSON fence stripping for thinking-prefix models | evaluation | 2h |
| PII regex adversarial testing (homoglyphs, zero-width, space-padding) | safety | 4h |
| PII regex Hypothesis fuzzing (emails, arbitrary Unicode) | safety | 3h |
| Calibration ROC accuracy with varied distributions | safety | 3h |
| chi2 approximation accuracy vs scipy | decision | 2h |
| Metric blended score with improvement objectives | optimization | 3h |
| Batch BERTScore index alignment verification | evaluation | 3h |

**Total: ~27 hours.**

### Stage 4: Resilience & Operations

**Entry:** Stage 3 complete, statistical confidence established.
**Exit:** System handles failures gracefully. Monitoring produces actionable data.

| Work Item | Boundary | Effort |
|---|---|---|
| Fix IDOR source bugs (add check_resource_owner to 5 endpoint groups) | server | 8h |
| Fix CORS origin validation (reject * with credentials) | server | 2h |
| Concurrent task claiming test (Postgres) | orchestration | 4h |
| A/B batch commit partial failure handling | orchestration | 3h |
| GEPA timeout intermediate result capture | optimization | 3h |
| Cost cap enforcement during GEPA optimization | optimization | 3h |
| Prometheus metric recording verification | safety | 2h |
| SSE progress queue overflow behavior | orchestration | 2h |
| Postgres CI job setup | database | 4h |
| SQLite WAL concurrent write test | database | 3h |

**Total: ~34 hours.**

### Stage 5: Production Sign-Off

**Entry:** Stage 4 complete.
**Exit:** All gates pass. E2E smoke tests with real APIs. Documentation complete.

| Work Item | Boundary | Effort |
|---|---|---|
| E2E smoke tests (3 pairs, real API, scores in expected ranges) | evaluation | 4h |
| E2E optimization test with Ollama | optimization | 4h |
| E2E pipeline run (preflight→report) | orchestration | 6h |
| Security header verification on all response types | server | 2h |
| Performance baseline (response times, memory, DB queries) | server | 4h |
| Gate checklist verification (all stages pass) | all | 4h |

**Total: ~24 hours.**

## Per-Boundary Path

### Server HTTP & Security → Production
1. Stage 0: IDOR tests + JWT transition test (6h)
2. Stage 1: API schema snapshots (8h)
3. Stage 2: Fix Playwright CI, integration tests (7h)
4. Stage 3: (no property-based work needed)
5. Stage 4: Fix IDOR source, CORS fix, rate limit multi-user (13h)
6. Stage 5: Security header verification, performance baseline (6h)
- **Total: ~40h. Critical path: IDOR fixes block multi-user deployment.**
- See: `server-http-security.md`

### Database & Persistence → Production
1. Stage 0: Fix schema parity test (2h)
2. Stage 1: Column-level parity C1-C4, model defaults U1-U6 (9h)
3. Stage 2: Alembic data preservation, init_db idempotency (5h)
4. Stage 4: Postgres CI, WAL concurrent write test (7h)
- **Total: ~23h. Critical path: C1 (schema parity) unblocks all other DB work.**
- See: `database-persistence.md`

### Background Orchestration → Production
1. Stage 2: A/B runner simulation, full pipeline integration (10h)
2. Stage 3: (no property-based work)
3. Stage 4: Concurrent claiming, batch commit, SSE overflow (9h)
4. Stage 5: E2E pipeline run (6h)
- **Total: ~25h. Critical path: A/B runner tests unblock A/B feature release.**
- See: `background-orchestration.md`

### Safety & Observability → Production
1. Stage 0: Fix false-confidence tests (4h)
2. Stage 2: Presidio integration (3h)
3. Stage 3: PII adversarial, Wilson CI property, calibration ROC (14h)
4. Stage 4: Prometheus metric recording (2h)
- **Total: ~23h. Critical path: PII evasion testing blocks safety sign-off.**
- See: `safety-observability-decision.md`

### Evaluation Strategies → Production
1. Stage 1: Evaluator key ↔ METRIC_WEIGHTS alignment (4h)
2. Stage 2: Fallback chain, VCR cassettes (10h)
3. Stage 3: Wilson CI property, fence stripping, batch BERTScore (9h)
4. Stage 5: E2E smoke test (4h)
- **Total: ~27h. Critical path: Fence stripping blocks non-OpenAI model support.**
- See: `evaluation-strategies.md`

### Ingest Adapters → Production
1. Stage 1: DataAdapter contract tests (4h)
2. Stage 2: Redis Docker integration, fix OTel nondeterminism (10h)
3. Stage 3: Splitter property-based tests (3h)
- **Total: ~17h. Minimal external dependencies.**
- See: `ingest-adapters.md`

### Optimization Engines → Production
1. Stage 1: (no contract work needed beyond shared fixtures)
2. Stage 2: VCR cassettes (6h)
3. Stage 3: Metric blended score (3h)
4. Stage 4: GEPA timeout, cost cap enforcement (6h)
5. Stage 5: Ollama integration (4h)
- **Total: ~19h. Critical path: Cost cap blocks production deployment.**
- See: `optimization-engines.md`

### Report Generation & CLI → Production
1. Stage 1: Report generator contract tests, shared factory (7h)
2. Stage 2: ci_output tests, CLI smoke tests, display tests (9h)
- **Total: ~16h. Lowest risk boundary.**
- See: `report-generation-cli.md`

## Dependency Graph

```
                    Stage 0 (Safety Net)
                         |
              Stage 1 (Contract Coverage)
                    /         \
        Stage 2 (Integration)  Stage 3 (Property-Based)
                    \         /
             Stage 4 (Resilience)
                         |
            Stage 5 (Production Sign-Off)
```

**Cross-boundary dependencies:**
- `database` C1 (schema parity) → blocks `server` integration tests (need correct schema)
- `server` IDOR fixes → blocks multi-user deployment (all boundaries affected)
- `evaluation` contract tests → blocks `optimization` metric tests (shared scoring functions)
- `shared` migration_result_factory → blocks efficient work in `report-cli`, `server`, `evaluation`
- `shared` VCR infrastructure → blocks `evaluation` and `optimization` cassette tests
- Postgres CI setup (database) → blocks `orchestration` concurrent claiming test

**Critical path:** Stage 0 → Stage 1 (schema parity + shared factory) → Stage 2 (IDOR fix + integration tests) → Stage 4 (Postgres CI) → Stage 5.

## Milestone Plan

### Milestone 1: Safety Net + Foundation (~3 person-weeks)
**Scope:** Stage 0 + Stage 1 + shared infrastructure (migration_result_factory, VCR setup)
**Entry:** Current state
**Exit:** All false-confidence tests fixed, IDOR tests written (and failing), schema parity test written (and failing), contract tests passing, shared fixture factory merged.
**Effort:** ~45 hours
**What this buys:** Confidence that existing tests actually test what they claim. Foundation for all future test work.

### Milestone 2: Integration + Security Fixes (~3 person-weeks)
**Scope:** Stage 2 + IDOR source fixes from Stage 4
**Entry:** Milestone 1 complete
**Exit:** IDOR bugs fixed, Redis/Postgres integration tests passing, A/B runner tested, Playwright running in CI, VCR cassettes recorded.
**Effort:** ~54 hours
**What this buys:** Multi-user mode is safe to deploy. All external service boundaries tested against real (local) instances.

### Milestone 3: Statistical Hardening (~2 person-weeks)
**Scope:** Stage 3
**Entry:** Milestone 2 complete
**Exit:** Wilson CI, PII regex, calibration, and scoring math have property-based tests. JSON fence stripping covers CoT models.
**Effort:** ~27 hours
**What this buys:** Migration recommendations are mathematically sound. Safety module resists basic evasion.

### Milestone 4: Resilience + Production (~2 person-weeks)
**Scope:** Stage 4 (remaining) + Stage 5
**Entry:** Milestone 3 complete
**Exit:** Postgres CI green, concurrent safety verified, cost caps enforced, E2E smoke tests pass, performance baselined.
**Effort:** ~50 hours
**What this buys:** Production-hardened. Can deploy with confidence under load.

## Risk Register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | IDOR bugs require source fixes, not just tests — scope creep from plan-only into implementation | HIGH | Stage 0 writes failing tests only. Source fixes are a separate tracked work item in Stage 4. |
| 2 | DSPy version upgrade breaks mocked test internals silently | HIGH | Add a DSPy version-pinning contract test. Run real DSPy compilation in nightly CI (Ollama). |
| 3 | Postgres CI setup takes longer than estimated (infra work) | MEDIUM | Start with GitHub Actions service container. Document fallback to local Docker. |
| 4 | VCR cassettes become stale when API response formats change | MEDIUM | Tag cassettes with API version. Re-record on model version bump. |
| 5 | Shared migration_result_factory becomes its own maintenance burden | MEDIUM | Keep it minimal (dict merge, not a builder pattern). Accept 80% coverage. |
| 6 | Property-based tests find bugs that require non-trivial source changes | MEDIUM | Budget 20% buffer time in Milestone 3. Track findings in tangential-findings.md. |
| 7 | Playwright tests remain flaky in CI | MEDIUM | Use retry: 2 in CI config. Prefer API tests for logic, Playwright for visual/flow only. |
| 8 | PII adversarial testing reveals fundamental regex limitations | LOW | Document known limitations. Recommend Presidio as production PII engine. |
| 9 | A/B runner source has bugs exposed by new tests | LOW | Track as separate bug fix work. Tests should fail, not be weakened. |
| 10 | Team bandwidth doesn't support 4-milestone plan | LOW | Milestones are independently shippable. Even Milestone 1 alone improves confidence. |

## What the First PR Should Be

**PR: "Fix false-confidence schema parity test and add column-level comparison"**

Write `test_schema_parity_columns` in `tests/test_alembic.py` that:
1. Creates a DB via `Base.metadata.create_all(engine)`
2. Creates a DB via `alembic upgrade head`
3. Inspects both with `sqlalchemy.inspect(engine).get_columns(table)` for every table
4. Asserts column names, types, and nullable match

This is the single highest-ROI test in the entire review:
- ~2 hours of work
- Zero external dependencies
- Catches the most dangerous class of regression (schema drift)
- Immediately actionable — no design decisions needed
- The existing test's false confidence makes this urgent, not just important

If the test fails (which it likely will, given `_migrate_add_columns` as a second source of truth), that failure is itself the deliverable — it proves the test was needed.
