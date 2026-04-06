# Server HTTP & Security -- Testing Review

**Scope:** FastAPI application, 20+ API routers, auth middleware (JWT + API key + CSRF), RBAC, rate limiting, CSP nonce middleware, security headers, CORS, Jinja2 template rendering, static file serving.

**Date:** 2026-04-06
**Author:** Testing Lead (Server HTTP & Security)

---

## 1. Boundary Map

### Inside (we test it)

```
FastAPI app (create_app)
  |
  +-- Middleware stack (execution order: RequestID -> SecurityHeaders -> CORS -> Auth -> CSRF -> route)
  |     +-- SecurityHeadersMiddleware (CSP nonce, X-Frame-Options, HSTS-like headers)
  |     +-- RequestIDMiddleware (UUID per request)
  |     +-- AuthMiddleware (JWT decode, API key verify, session cookie check)
  |     +-- CSRFMiddleware (double-submit cookie)
  |     +-- CORSMiddleware (Starlette built-in)
  |
  +-- API routers (JSON responses)
  |     +-- /api/v1/migrations (CRUD, test-cases, diagnostics, stream SSE, approve/reject, shadow)
  |     +-- /api/v1/comparisons (distributions, diff)
  |     +-- /api/v1/reports (markdown, HTML, PDF)
  |     +-- /api/v1/models (registered model CRUD)
  |     +-- /api/v1/costs (cost tracking, monthly spend)
  |     +-- /api/v1/alerts (alert CRUD, batch operations)
  |     +-- /api/v1/ab-tests (A/B testing CRUD + results + conclude)
  |     +-- /api/v1/versioning (migration version snapshots, restore, promote)
  |     +-- /api/v1/audit-log (read-only listing with filters)
  |     +-- /api/v1/pipelines (multi-module pipeline CRUD)
  |     +-- /api/v1/users (admin-only user CRUD)
  |     +-- /api/v1/teams (team + membership CRUD)
  |     +-- /api/v1/annotations (annotation queue)
  |     +-- /api/v1/auth (login, register, refresh, me)
  |     +-- /api/v1/tasks (background task status)
  |     +-- /api/v1/dataset-runs (dataset generation tracking)
  |     +-- /api/v1/health, /health/ready, /health/live
  |     +-- /metrics (prometheus)
  |     +-- approvals (approval workflow)
  |     +-- deprecation (model deprecation scanner)
  |
  +-- UI routers (HTML responses via Jinja2)
  |     +-- /ui/ (dashboard), /ui/migrations, /ui/migrations/{id}, /ui/migrations/new
  |     +-- /ui/alerts, /ui/audit-log, /ui/users, /ui/teams, /ui/annotations
  |     +-- /ui/login, /ui/logout, /ui/costs, /ui/pipelines
  |     +-- /ui/fragments/* (HTMX partials for diff, charts)
  |
  +-- auth_utils.py (password hashing, JWT create/decode -- pure functions)
  +-- rbac.py (require_role dependency, check_resource_owner, get_current_user_id)
  +-- rate_limit.py (sliding window, in-memory)
  +-- schemas.py (Pydantic response models)
  +-- database.py (engine factory, session generator, init_db, schema migration)
  +-- models.py (SQLModel table definitions)
  +-- progress.py, pipeline_runner.py, task_dispatch.py, task_worker.py
  +-- Error handlers (404/500 HTML vs JSON branching)
  +-- Static file serving (/static/)
  +-- Lifespan (startup: init_db, orphan recovery, JWT check, deprecation check)

### Outside (we mock/stub)

- LiteLLM (model info API calls in models.py)
- Redis (optional task queue backend, only when REDIS_URL set)
- Sentry SDK (optional error reporting)
- Real LLM API calls (OpenAI, Anthropic, etc.)
- passlib/bcrypt (tested via auth_utils but not the crypto itself)
- PyJWT internals (tested via decode/encode roundtrips)
- Filesystem for data file storage (mocked via tmp_path)
- prometheus_client (tested presence/absence only)
- weasyprint (PDF generation, tested as 501 fallback)

### On the fence (integration with real local instance)

- SQLite in-memory (used by all TestClient tests -- real SQL, real ORM)
- Jinja2 template rendering (real templates, real data binding)
- Playwright against live uvicorn process on port 8765
```

---

## 2. Current Coverage Audit

### File-by-file analysis

#### `tests/test_server/conftest.py`
- **Covers:** engine fixture (in-memory SQLite or DATABASE_URL), session, sample_migration, sample_test_cases, client fixture with DI override.
- **Misses:** No fixture for multi-user mode with JWT. No fixture for authenticated client with Bearer token. Each multi-user test file recreates its own engine/app setup independently.
- **Brittle:** None observed.
- **Dead tests:** None.

#### `tests/test_server/test_auth_csrf.py`
- **Covers:** Auth disabled (all open), auth enabled (API 401 without header, 200 with Bearer, 401 wrong key, UI redirects to login, login/logout flows, session cookie). CSRF: cookie set on GET, 403 without token, passes with valid token (form and header), mismatched token 403, skipped for API routes. `_csrf_enabled()` unit tests. Auth helper unit tests (_verify_key, _create_session_token).
- **Misses:** CSRF with multi-user JWT mode (only tested with legacy API key). No test for CSRF bypass via /api/ path prefix injection. No test for CSRF with PUT/DELETE methods (only POST tested).
- **False-confidence:** `test_csrf_skipped_for_api_routes` only checks status != 403, doesn't verify the request actually succeeded with the right data.

#### `tests/test_server/test_auth_jwt.py`
- **Covers:** Register first user (admin), second user (viewer), duplicate username 409, login returns JWT, wrong password 401, /me endpoint, /refresh endpoint, endpoints return 404 without multi-user.
- **Misses:** No test for expired JWT auth attempt against protected endpoints. No test for JWT with wrong secret against middleware. No test for auth/register self-assign admin role prevention from a second user. No test for deactivated user login attempt. No test for token refresh with expired token. Rate limiting on auth endpoints not tested here.
- **Brittle:** Uses temp file DB with `reset_engine()` -- if tests run in parallel, the global engine singleton could race.

#### `tests/test_server/test_jwt_validation.py`
- **Covers:** Default secret raises RuntimeError in multi-user. Short secret warns. Secure secret no warning. No warning in single-user. All truthy variants ("1", "true", "yes", "True", "YES") raise with default.
- **Misses:** Does not test that the server actually refuses to start (only tests the `_check_jwt_secret()` function in isolation, not via lifespan). This is acceptable since it's the same code path.

#### `tests/test_server/test_security.py`
- **Covers:** CSP frame-ancestors, object-src, form-action. Permissions-Policy header. CORS absent when not configured, present for allowed origin, absent for unlisted origin. CSRF cookie Secure flag (with/without HTTPS). Session cookie Secure flag. CSP nonce in script-src, no unsafe-inline in script-src, nonce differs per request, style-src retains unsafe-inline. CORS always registered.
- **Misses:** No test for X-Content-Type-Options value. No test for X-Frame-Options value. No test for X-XSS-Protection value. No test for Referrer-Policy value. No test that CSP nonce actually matches what templates receive (only checks header). No test for CSP connect-src directive. **style-src 'unsafe-inline' is noted present -- this is a known CSP weakness but documented as intentional for Tailwind.**
- **False-confidence:** `test_csp_nonce_differs_per_request` checks csp1 != csp2 which is sufficient, but doesn't verify the nonce format is cryptographically random.

#### `tests/test_server/test_rbac.py`
- **Covers:** No-op without multi-user. 401 with no user. 403 with wrong role. Passes with matching role. Passes with one-of-multiple roles. 403 for viewer on editor endpoint. Handles object-type user. 401 when no user attr on state.
- **Misses:** No test for `check_resource_owner()`. No test for `get_current_user_id()`. No test for `is_admin_user()`. These helper functions are tested implicitly via test_api_isolation.py but not unit-tested directly.

#### `tests/test_server/test_rate_limit.py`
- **Covers:** Allows within window. Blocks on limit exceeded. Positive retry_after. Independent limits per IP. Metric recording on rejection.
- **Misses:** No test for multi-user mode key derivation (user_id-based). No test for window expiry (time travel). No test for different endpoints having independent limits. No property-based tests for edge cases (limit=0, limit=1, concurrent access). No test for `reset_for_testing()` completeness.

#### `tests/test_server/test_api_migrations.py`
- **Covers:** Health endpoint. List empty/with data. Pagination. Get detail (scores, test cases, cluster summary, null cluster, 404). Create migration (success, missing fields, negative max_cost 422, valid max_cost, cluster_prompts). List test cases (filter by phase, output_type, pagination, 404). Get test case detail, 404. Config filtering (lm_extra_kwargs stripped). UI endpoints (dashboard, migrations list, detail pages with dummy data).
- **Misses:** No test for migration delete. No test for migration re-run/retry. No test for SSE stream beyond basic checks (that's in test_api_sse.py). No test for the diagnostics endpoint. No test for executive report endpoint. No test for export endpoints. **No IDOR test for migration detail in non-multi-user mode (test_api_isolation covers multi-user only).**

#### `tests/test_server/test_api_isolation.py`
- **Covers:** Multi-user migration isolation (can't see others, can see own, cross-user GET 403, admin sees all, single-user returns all). Pipeline isolation (same pattern).
- **Misses:** No isolation test for A/B tests. No isolation test for annotations. No isolation test for audit log (audit log has NO access control at all -- see Risk #2). No isolation test for costs endpoint.

#### `tests/test_server/test_api_comparisons.py`
- **Covers:** _word_diff_html unit tests (identical, multiline, changed/deleted/inserted words, HTML escaping). Distributions (with data, empty, 404, multiple types). Diff (no content, with content, 404 cases). UI fragments (fallback, real data, charts placeholder).
- **Misses:** No test for XSS in diff HTML output with adversarial input beyond basic `<b>` tag test. No test for very large diffs.

#### `tests/test_server/test_negative_stress.py`
- **Covers:** File upload abuse (50MB boundary, 50MB+1 rejected, non-JSONL, path traversal, empty file, binary file, no file, empty filename, wrong content-type, rapid uploads). Concurrent migration stress (10 rapid, pending detail page, startup recovery). Database edge cases (deleted record, orphaned test cases, deletion mid-task). API endpoint abuse (SQL injection, XSS, path traversal -- checked later sections).
- **Misses:** Based on file size (~500 lines), sections 6 and 7 (API abuse, state machine violations) are present but I could only partially read. The XSS tests likely test template rendering but may not cover all injection vectors.
- **Brittle:** 50MB upload test allocates 50MB in memory per run, making the test suite memory-heavy.

#### `tests/test_server/test_api_audit.py`
- **Covers:** log_audit utility (creates entry, with details, with user_id). List audit log (all, filter by resource_type, filter by action, pagination, empty).
- **Misses:** **No access control test. The audit log endpoint has zero auth/RBAC protection. Any authenticated user can read all audit entries including other users' actions. This is a P0.5 IDOR issue.** No test for date range filters (start_date, end_date). No test for user_id filter.

#### `tests/test_server/test_api_alerts.py`
- **Covers:** Alert generation, API endpoints, UI page (based on file beginning).
- **Misses:** Alerts have RBAC via require_role, but no ownership scoping -- all users with viewer+ role see all alerts. This may be intentional (alerts are global).

#### `tests/test_server/test_api_ab_testing.py`
- **Covers:** Create A/B test, missing migration 404.
- **Misses:** Based on what I read, basic CRUD is covered. Need to verify conclude, results, and status transitions are tested.

#### `tests/test_server/test_api_sse.py`
- **Covers:** SSE content type, 404 for unknown migration, catchup on connect (terminal status), nginx buffering header, no-cache header, progress field inclusion.
- **Misses:** No test for SSE with active (non-terminal) migration. No test for SSE auth (stream endpoint may be unprotected). No test for SSE connection timeout/cleanup.

#### `tests/test_server/test_api_users.py`, `test_api_teams.py`
- **Covers:** Basic CRUD operations with JWT auth.
- **Misses:** Based on partial read, likely has basic coverage. Need full read to confirm edge cases.

#### `tests/test_server/test_api_approvals.py`, `test_api_annotations.py`
- **Covers:** Basic CRUD with JWT auth.
- **Misses:** Based on partial read, basic flows covered.

#### `tests/test_server/test_app.py`
- **Covers:** Sentry init (with DSN, without DSN, graceful ImportError). create_app succeeds.
- **Misses:** No test for lifespan (startup recovery, task worker init). No test for error handlers (404 HTML vs JSON branching, 500 handlers).

#### `tests/test_server/test_health_probes.py`
- **Covers:** /health/live always 200. /health backward compat. /health/ready 200 or 503 based on DB. Response shape.
- **Misses:** No test for degraded state (task worker down but DB up). No test for Redis component status.

#### `tests/test_server/test_p5_polish.py`
- **Covers:** Error pages (UI 404 branded HTML, API 404 JSON).
- **Misses:** Based on partial read, likely covers security headers and mobile nav.

#### `tests/test_server/test_api_shadow.py`
- **Covers:** Shadow config YAML download (returns YAML, valid YAML, contains models, structure, content-disposition).
- **Misses:** No access control test on shadow config download.

#### `tests/test_e2e/test_playwright_ui.py`
- **Covers:** Models dashboard (loads, shows models, active count, deprecated card, alerts banner, add model button, explore table). Many more pages based on test class names visible in the file.
- **Misses:** All Playwright tests use dummy data (DUMMY_MODELS, DUMMY_MIGRATIONS), not real DB data. No Playwright tests for auth flows (login page, session cookie, protected routes). No tests for multi-user workflows.
- **CI Portability:** Uses `lsof` (macOS/Linux only), kills port 8765 processes, spawns uvicorn via `uv run`. Hardcoded `BASE_URL = "http://localhost:8765"`. Will not work on Windows. The `_kill_port` function uses `kill -9` which is aggressive. Server startup timeout is 30s with 0.5s polling -- could be flaky on slow CI.
- **Brittle:** Session-scoped server fixture means if one test breaks the server state, all subsequent tests in the session fail. The 60s default timeout per assertion is generous but could mask slow rendering.

#### `tests/test_server/test_auth_utils.py`
- **Covers:** Hash password format. Verify correct/wrong password. JWT create returns string. JWT roundtrip. Expired token. Invalid token. Wrong secret.
- **Misses:** No test for very long passwords. No test for empty password. No test for unicode passwords. No test for JWT with custom expiry.

---

## 3. Risk Ranking

### R1. CRITICAL -- JWT default secret usable in single-user mode (P0.3)

**Manifestation:** In single-user mode with `ROSETTASTONE_API_KEY` set, the JWT secret defaults to `"dev-secret-change-in-production"`. If multi-user mode is later enabled without changing the secret, all previously-issued JWTs remain valid. Even in single-user mode, the `_try_decode_jwt()` fallback in AuthMiddleware attempts JWT decode with the default secret, meaning anyone who knows the default can forge a JWT that passes auth if multi-user is enabled later. The `_check_jwt_secret()` only raises in multi-user mode -- it's a warning, not a gate.

**Existing tests catch it?** Partially. `test_jwt_validation.py` tests the startup check raises in multi-user mode, but does NOT test that a forged JWT with the default secret is rejected at the middleware level in single-user mode. The middleware doesn't even try JWT in pure single-user mode (it falls through to API key check), so the risk is specifically at the transition point to multi-user.

**Likelihood:** Medium (requires specific configuration transition). **Blast radius:** Total auth bypass.

### R2. HIGH -- IDOR on audit log (S3/P0.5)

**Manifestation:** The `/api/v1/audit-log` endpoint has NO access control -- no `require_role`, no `check_resource_owner`. In multi-user mode, any authenticated user (even a viewer) can read the full audit log including actions by other users, resource IDs, and details. The audit log contains migration IDs, user IDs, and action details that could be used for reconnaissance.

**Existing tests catch it?** No. `test_api_audit.py` only tests the utility function and basic list/filter functionality. There is no test asserting that non-admin users cannot see other users' audit entries.

**Likelihood:** High (endpoint is openly accessible). **Blast radius:** Information disclosure across user boundaries.

### R3. HIGH -- IDOR on migration detail / comparisons / reports / shadow config

**Manifestation:** The comparisons, reports, and shadow config endpoints use `_get_migration_or_404()` which does NOT check `owner_id`. In multi-user mode, a user who knows another user's migration ID can access:
- `/api/v1/migrations/{id}/distributions`
- `/api/v1/migrations/{id}/test-cases/{tc_id}/diff`
- `/api/v1/migrations/{id}/report/markdown`
- `/api/v1/migrations/{id}/report/html`
- `/api/v1/migrations/{id}/shadow/config.yaml`

The migration detail endpoint (`GET /api/v1/migrations/{id}`) DOES call `check_resource_owner`, but these subordinate endpoints do not.

**Existing tests catch it?** `test_api_isolation.py` tests the migration list and detail isolation, but NOT the subordinate endpoints (distributions, diff, report, shadow).

**Likelihood:** High. **Blast radius:** Data leakage of prompts, responses, scores, optimized prompts across users.

### R4. HIGH -- CORS misconfiguration risk

**Manifestation:** `ROSETTASTONE_CORS_ORIGINS` is a comma-separated string. If set to `*` or a broad wildcard, credentials-bearing cross-origin requests become possible. The middleware uses `allow_credentials=True`, which combined with a permissive origin list, allows a malicious site to make authenticated API calls using the victim's session cookie. There is no validation that origins are HTTPS or that `*` is rejected when credentials are enabled.

**Existing tests catch it?** `test_security.py` tests that unlisted origins are blocked, but does NOT test the `*` wildcard case or verify that `allow_credentials=True` + broad origins is rejected.

**Likelihood:** Medium (requires admin misconfiguration). **Blast radius:** Full account takeover via CSRF-like attack.

### R5. MEDIUM -- CSP style-src 'unsafe-inline'

**Manifestation:** The CSP header includes `style-src 'self' 'unsafe-inline' fonts.googleapis.com`. This allows inline style injection, which can be used for CSS-based data exfiltration. Documented as intentional for Tailwind CDN, but it's a defense-in-depth gap.

**Existing tests catch it?** Yes, `test_csp_style_src_retains_unsafe_inline` explicitly asserts this. The test documents the intentional trade-off.

**Likelihood:** Low (requires XSS foothold first). **Blast radius:** Limited data exfiltration.

### R6. MEDIUM -- Rate limiting bypass via multi-user key switching

**Manifestation:** The rate limiter keys by `user:{user_id}` in multi-user mode, `ip:{host}` in single-user. An attacker behind a proxy (same IP) could create multiple user accounts and use each one's JWT to bypass rate limits, effectively getting N * limit requests per window.

**Existing tests catch it?** No. Rate limit tests only cover single-user IP-based limiting.

**Likelihood:** Medium. **Blast radius:** Resource exhaustion, cost overrun on LLM API calls.

### R7. MEDIUM -- Auth middleware doesn't check multi-user for UI routes

**Manifestation:** In multi-user mode, the AuthMiddleware only checks JWT for `/api/` and `/metrics` routes. UI routes (`/ui/*`) still use the legacy session cookie mechanism if `ROSETTASTONE_API_KEY` is set. If multi-user is enabled WITHOUT an API key, UI routes beyond login/static are not authenticated at all -- the middleware falls through to `return await call_next(request)` at line 131.

**Existing tests catch it?** No. The multi-user + no-API-key + UI routes combination is not tested.

**Likelihood:** Medium (specific config combo). **Blast radius:** Unauthenticated UI access.

### R8. MEDIUM -- SSE stream endpoint lacks ownership check

**Manifestation:** `GET /api/v1/migrations/{id}/stream` appears to not call `check_resource_owner`. Any authenticated user could subscribe to progress events for any migration.

**Existing tests catch it?** No. `test_api_sse.py` does not test access control.

**Likelihood:** Medium. **Blast radius:** Information disclosure (migration progress, status, stage details).

### R9. LOW -- In-memory rate limiter state lost on restart

**Manifestation:** The rate limiter uses module-level `defaultdict` and `threading.Lock`. Server restart clears all rate limit state, allowing burst of requests immediately after restart. In multi-worker deployments (gunicorn with multiple workers), each worker has independent rate limit state.

**Existing tests catch it?** The `reset_for_testing()` function implicitly acknowledges this, but no test verifies the restart-clears-state behavior.

**Likelihood:** Medium (production deployments typically use multiple workers). **Blast radius:** Low (temporary rate limit bypass).

### R10. LOW -- Playwright CI portability

**Manifestation:** Tests use macOS-specific `lsof` command, `kill -9`, and hardcoded port. Will fail on Windows CI. Uses `uv run uvicorn` which requires `uv` in PATH. Session-scoped server means test isolation issues.

**Existing tests catch it?** N/A -- this is the tests themselves being brittle.

**Likelihood:** High (any non-macOS CI). **Blast radius:** Low (test suite failure, not production issue).

---

## 4. Test Plan by Tier

### Tier 1: Unit (pure logic, no I/O)

| # | Test | Assertions | Status | Write-time |
|---|------|-----------|--------|------------|
| U1 | `_word_diff_html` with adversarial input (null bytes, very long strings, Unicode RTL) | No crash, HTML properly escaped | PARTIAL (basic escaping tested) | S |
| U2 | `check_resource_owner` with all role/owner combos | 403 for non-owner non-admin, pass for owner, pass for admin, no-op in single-user | MISSING | S |
| U3 | `get_current_user_id` with dict user, object user, None, single-user mode | Correct ID extraction or None | MISSING | S |
| U4 | `is_admin_user` with all user types | True for admin, False for others, False for None | MISSING | S |
| U5 | `_get_key` (rate_limit) multi-user mode with user_id, single-user with IP | Correct key string format | MISSING | S |
| U6 | `_csrf_enabled` boundary: empty string env var, whitespace, case variations | Correct boolean return | EXISTS | S |
| U7 | `_verify_key` with empty strings, unicode, very long strings | Correct constant-time comparison | PARTIAL (basic cases only) | S |
| U8 | `_create_session_token` idempotency and format | SHA-256 hex output | EXISTS | S |
| U9 | Pydantic schema validation (MigrationSummary, TestCaseDetail, etc.) with invalid data | ValidationError for bad types/missing fields | MISSING | M |
| U10 | `_check_jwt_secret` with edge cases (32-byte exactly, 31-byte, UTF-8 multibyte) | Correct raise/warn/pass behavior | PARTIAL | S |

### Tier 2: Contract (API schema stability, response shapes)

| # | Test | Assertions | Status | Write-time |
|---|------|-----------|--------|------------|
| C1 | GET /api/v1/migrations response matches PaginatedResponse[MigrationSummary] | All fields present with correct types | PARTIAL (some fields checked) | M |
| C2 | GET /api/v1/migrations/{id} response matches MigrationDetail schema | All 20+ fields present | PARTIAL | M |
| C3 | GET /api/v1/audit-log response matches PaginatedResponse[AuditLogEntry] | Pagination metadata, item shapes | EXISTS | S |
| C4 | POST /api/v1/auth/login response matches TokenResponse | access_token, token_type, user_id, role | EXISTS | S |
| C5 | Error responses consistently use {"detail": ...} format for all endpoints | Consistent error shape | MISSING | M |
| C6 | GET /api/v1/health/ready response includes components.database.status | Schema contract for health probes | EXISTS | S |
| C7 | All paginated endpoints respect per_page <= 100 constraint | 422 for per_page > 100 | MISSING | S |
| C8 | SSE event format: "data: {json}\n\n" with correct payload shape | type, migration_id, status fields | EXISTS | S |
| C9 | All datetime fields in API responses use ISO 8601 format | Consistent date serialization | MISSING | S |
| C10 | OpenAPI spec generated by FastAPI matches actual response shapes | Spec vs reality | MISSING | L |

### Tier 3: Integration (TestClient full-stack)

| # | Test | Assertions | Status | Write-time |
|---|------|-----------|--------|------------|
| I1 | **Audit log IDOR: viewer user reads all entries in multi-user mode** | Currently 200 (bug); should be 403 or filtered | MISSING (P0.5) | M |
| I2 | **Comparisons IDOR: user accesses other user's distributions/diff** | Currently no ownership check (bug) | MISSING (P0.5) | M |
| I3 | **Reports IDOR: user downloads other user's report** | Currently no ownership check (bug) | MISSING (P0.5) | M |
| I4 | **Shadow config IDOR: user downloads other user's shadow config** | Currently no ownership check (bug) | MISSING (P0.5) | S |
| I5 | **SSE stream IDOR: user subscribes to other user's migration stream** | Needs ownership check | MISSING | S |
| I6 | Multi-user + no API key + UI route access | Should require JWT cookie or redirect to login | MISSING (R7) | M |
| I7 | CORS with `*` origin + allow_credentials | Should reject or warn | MISSING | S |
| I8 | Auth middleware: expired JWT returns 401 (not 500) | Graceful JWT error handling | MISSING | S |
| I9 | Auth middleware: malformed Bearer token (not JWT, not API key) | 401 not 500 | MISSING | S |
| I10 | CSRF with PUT/DELETE methods on UI routes | 403 without token | MISSING | S |
| I11 | Rate limit integration: submission endpoint returns 429 with Retry-After header | Header present, correct value | MISSING | M |
| I12 | Error handler: unhandled exception in /api/ route returns JSON 500 | JSON format, no stack trace | MISSING | S |
| I13 | Error handler: unhandled exception in /ui/ route returns HTML 500 | Branded 500 page | MISSING | S |
| I14 | Migration create sets owner_id from JWT in multi-user mode | owner_id matches token sub | MISSING | M |
| I15 | Approval workflow: approver can approve, viewer gets 403 | RBAC enforced on approve/reject | PARTIAL | M |
| I16 | Approval workflow: duplicate approval from same user returns 409 | Dedup logic works | PARTIAL | S |
| I17 | User management: non-admin can update own email/password but not role | Privilege boundary | MISSING | M |
| I18 | User management: admin cannot delete self | 400 response | MISSING | S |
| I19 | Team membership: add non-existent user returns 404 | FK validation | PARTIAL | S |
| I20 | Lifespan: orphaned migration recovery on startup | running -> failed | EXISTS | S |
| I21 | Lifespan: JWT secret check blocks startup with default secret in multi-user | RuntimeError in lifespan | EXISTS (via _check_jwt_secret) | S |
| I22 | Request ID middleware: X-Request-ID header present and unique per request | UUID4 format, differs between requests | MISSING | S |
| I23 | Security headers present on ALL response types (API, UI, error, static) | Headers on 200, 404, 500 | PARTIAL (only /api/v1/health tested) | M |
| I24 | Template rendering with XSS payloads in model names, migration data | Jinja2 auto-escaping prevents injection | PARTIAL (in negative_stress) | M |
| I25 | Annotation create with invalid annotation_type returns 422 | Enum validation | MISSING | S |
| I26 | Cost endpoint user scoping in multi-user mode | User sees only own costs | MISSING | M |

### Tier 4: Property-based (Hypothesis/hypothesis-style)

| # | Test | Assertions | Status | Write-time |
|---|------|-----------|--------|------------|
| P1 | Rate limiter: arbitrary interleaving of check_rate_limit calls never exceeds limit | Invariant: window count <= limit after any sequence | MISSING | M |
| P2 | Auth header parsing: arbitrary strings in Authorization header never crash middleware | No 500, only 401 or pass | MISSING | M |
| P3 | JWT decode: arbitrary byte strings never crash decode_jwt (raises InvalidTokenError) | No unhandled exceptions | MISSING | S |
| P4 | Pagination: arbitrary page/per_page values produce valid responses or 422 | No 500, no negative offsets | MISSING | M |
| P5 | CSRF token: arbitrary cookie/header combinations either pass or return 403 | No bypass possible | MISSING | M |

### Tier 5: End-to-end (Playwright)

| # | Test | Assertions | Status | Write-time |
|---|------|-----------|--------|------------|
| E1 | Login flow: enter API key, get redirected, access protected page | Cookie set, page loads | MISSING | M |
| E2 | Login flow: wrong key shows error message | Error visible, no redirect | MISSING | M |
| E3 | Multi-user login: register, login with username/password, access /ui/ | JWT flow via browser | MISSING | L |
| E4 | CSRF in browser: form submission includes _csrf_token | Form submits without 403 | MISSING | L |
| E5 | CSP nonce: inline scripts have nonce attribute matching CSP header | Script execution works | MISSING | L |
| E6 | Dark mode toggle: cookie/localStorage persists across page navigation | Theme state maintained | EXISTS | S |
| E7 | Migration detail with real DB data (not dummy) | Scores, test cases render | MISSING | L |
| E8 | Error page rendering: 404 page shows branded content | Custom 404 visible | MISSING | M |
| E9 | SSE progress: create migration, watch progress bar update | EventSource connects, DOM updates | MISSING | XL |
| E10 | Mobile responsive: navigation works at 375px width | Hamburger menu, no overflow | EXISTS (partial) | M |

---

## 5. Synthetic Data Generation Strategy

### What "realistic" means

1. **Multi-user scenarios:** 3-5 users (admin, editor, viewer, approver) with distinct owner_ids. Migrations owned by different users. Cross-user access attempts.
2. **Concurrent requests:** Parallel migration submissions from different users. Simultaneous SSE subscriptions. Rapid-fire auth attempts.
3. **Malicious inputs:** SQL injection payloads in query parameters (`'; DROP TABLE--`). XSS payloads in model names (`<script>alert(1)</script>`). Path traversal in file uploads (`../../etc/passwd`). JWT tokens with tampered payloads (changed role, changed sub). CSRF tokens with partial matches.
4. **Realistic migration data:** Migrations with 100+ test cases spanning 4 output types. Realistic score distributions (0.6-0.99 range). Mix of GO/NO_GO/CONDITIONAL recommendations. Warnings with HIGH/MEDIUM/LOW severity.

### How we generate it

- **Factory functions:** Extend conftest.py `sample_migration` to accept parameters (status, owner_id, recommendation, test_case_count). Add `sample_user(role)` factory.
- **Faker/hypothesis:** For property-based tests, use `hypothesis.strategies` to generate random auth headers, pagination params, and score values.
- **Multi-user scenario builder:** A fixture that creates admin + editor + viewer users, inserts migrations owned by each, and returns tokens + migration IDs. Reusable across all IDOR tests.

### Stability, rot prevention, cost profile

- All synthetic data created in-memory SQLite per test (no external state).
- Factory functions should be deterministic given a seed -- avoid `datetime.now()` in test data (use fixed timestamps).
- Cost: Zero (no LLM calls, no external services). The 50MB upload test is the most expensive at ~50MB memory per run. Consider marking it as `@pytest.mark.slow`.
- Rot prevention: Schema changes in `models.py` will break factories. Use a single `_make_migration()` helper that mirrors all current model fields with defaults, so adding a new column requires updating one place.

---

## 6. Fixtures, Fakes, and Mocks

### New fixtures needed

| Fixture | Scope | Description | Shared? |
|---------|-------|-------------|---------|
| `multi_user_env` | function | Sets ROSETTASTONE_MULTI_USER=true, JWT_SECRET to a test value, clears API_KEY. Yields, then cleans up. | Cross-subagent: yes, needed by DB tests too |
| `multi_user_scenario` | function | Creates admin (user_id=1), editor (user_id=2), viewer (user_id=3) in DB. Returns dict of {role: (user_id, jwt_token)}. | Server-only |
| `authenticated_client` | function | TestClient factory that takes a role and returns a client with Bearer token pre-configured. | Server-only |
| `sample_migration_factory` | function | Callable that creates a MigrationRecord with configurable owner_id, status, recommendation. Returns the record. | Cross-subagent: shares with DB-persistence boundary |
| `rate_limit_reset` | function/autouse | Calls `reset_for_testing()` before and after each test. Already exists in test_rate_limit.py but should be in conftest for all server tests. | Server-wide |

### Cross-subagent shared fixtures

- **DB session/engine:** The current `conftest.py` in test_server/ provides `engine`, `session`, `client`. The DB-persistence boundary likely needs the same setup. These should be factored into a shared conftest at `tests/conftest.py` level or a `tests/fixtures/` module.
- **TestClient:** Multiple test files (test_auth_jwt, test_api_users, test_api_teams, test_api_approvals, test_api_annotations, test_api_isolation) each create their own engine + app setup. This should be consolidated into a parameterizable fixture.

### Mocks

| Mock | What it replaces | Used where |
|------|-----------------|-----------|
| `MagicMock(task_worker)` | TaskDispatcher | test_negative_stress.py (prevents real background tasks) |
| `monkeypatch.setenv` | Environment variables | All auth/CORS/CSRF tests |
| `patch("rosettastone.server.app.get_engine")` | Database engine | Startup recovery tests |
| `patch("rosettastone.core.migrator.Migrator")` | Real LLM migration | Background task tests |

### Fakes that do NOT exist but should

- **FakeJWT:** A helper that creates JWTs with arbitrary claims (expired, wrong role, missing sub). Currently each test file has its own `_make_token()` helper -- consolidate.
- **FakeRequest:** A standardized mock Request with configurable `.state.user`, `.client.host`, `.headers`, `.cookies`. Currently using `SimpleNamespace` and `MagicMock` inconsistently.

---

## 7. Gaps You Can't Close

1. **Real HTTPS behavior:** The `Secure` cookie flag tests check the flag is set/unset based on env var, but cannot verify that browsers actually reject the cookie over HTTP. This requires a real browser + HTTPS termination, not TestClient.

2. **Timing attacks on `_verify_key`:** The function uses `hmac.compare_digest` which is constant-time, but there's no way to verify timing properties in a unit test. Would require statistical analysis of response times.

3. **Multi-worker rate limiting:** The in-memory rate limiter is per-process. Testing multi-worker behavior requires spawning multiple uvicorn workers, which is outside unit/integration test scope. This is a known architectural limitation.

4. **Real JWT cryptographic strength:** Tests verify HS256 roundtrip, but cannot verify the entropy of `secrets.token_urlsafe(16)` for CSP nonces or `secrets.token_hex(32)` for CSRF tokens. Cryptographic quality is assumed correct from Python's `secrets` module.

5. **Browser-specific CSP enforcement:** Tests verify the CSP header content, but cannot verify that Chrome/Firefox actually block inline scripts that lack the nonce. Requires real browser testing (Playwright could partially close this).

6. **Sentry error reporting fidelity:** Tests verify `sentry_sdk.init` is called with correct params, but cannot verify that exceptions actually reach Sentry. Requires a real Sentry instance or mock server. NEEDS_HUMAN_REVIEW: Is Sentry error reporting critical enough to warrant an integration test with a mock Sentry server?

7. **PostgreSQL-specific schema migration:** Tests run against SQLite. The `_migrate_add_columns` function has separate Postgres and SQLite code paths. The Postgres path (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`) is not tested in CI unless `DATABASE_URL` is set. NEEDS_HUMAN_REVIEW: Does CI include a PostgreSQL service?

---

## 8. Cost and Time Estimate

### By tier

| Tier | New tests | Estimated time | Notes |
|------|-----------|---------------|-------|
| Unit (U1-U10) | ~10 tests | 3-4 hours | Mostly simple, pure function tests |
| Contract (C1-C10) | ~10 tests | 4-6 hours | Schema validation, response shape assertions |
| Integration - IDOR fixes (I1-I5) | ~5 tests | 4-6 hours | Requires multi-user fixture setup; tests will FAIL until source is patched |
| Integration - Auth/Security (I6-I13) | ~8 tests | 6-8 hours | CORS, CSRF, error handlers, middleware edge cases |
| Integration - RBAC/Ownership (I14-I26) | ~13 tests | 8-12 hours | Multi-user scenarios, approval workflows, annotations |
| Property-based (P1-P5) | ~5 tests | 6-8 hours | Requires hypothesis setup, careful invariant definition |
| End-to-end (E1-E10) | ~10 tests | 12-16 hours | Playwright setup, server lifecycle management, CI integration |
| Fixture consolidation | N/A | 4-6 hours | Factor out shared fixtures, eliminate per-file engine creation |

### Total

- **Test writing:** 47-66 hours (~6-8 developer-days)
- **Source fixes for failing tests (IDOR, auth gaps):** 8-12 hours (~1-1.5 developer-days)
- **CI pipeline integration (Playwright, PostgreSQL):** 4-8 hours

### Grand total: ~60-86 hours (8-11 developer-days)

---

## 9. Path to Production

### Current readiness level

**Not production-ready.** Three blocking issues:

1. **P0.3 JWT default secret:** Single-user mode uses a hardcoded default. The startup check only enforces in multi-user mode. If a deployment transitions from single-user to multi-user without changing the secret, all auth is compromised. The source code already has the fix (`_check_jwt_secret` raises), but there's no enforcement that the secret was EVER set if multi-user was enabled after initial deployment.

2. **P0.5 IDOR on audit log, comparisons, reports, shadow config, SSE:** Five endpoint groups have no ownership checks. Any authenticated user can access any migration's data in multi-user mode.

3. **CORS allow_credentials=True without origin validation:** If an admin sets `ROSETTASTONE_CORS_ORIGINS=*`, full credential-bearing cross-origin access is enabled. No guard against this.

### Gap to production-hardened

- Fix IDOR issues (add `check_resource_owner` to comparisons, reports, shadow, SSE endpoints; add role-based access to audit log)
- Add CORS origin validation (reject `*` when credentials are enabled, or set `allow_credentials=False` for wildcard)
- Add auth check for UI routes in multi-user mode without API key
- Move rate limiter to Redis for multi-worker support (or document single-worker requirement)
- Add Retry-After header to rate-limited responses

### Gates

1. **All IDOR tests pass** (I1-I5) -- currently these tests don't exist, and the source has the bugs
2. **All auth edge case tests pass** (I6-I9)
3. **Security headers verified on ALL response types** (I23)
4. **Playwright tests pass in CI** (E1-E5 minimum)

### Ordered sequence

1. **Week 1:** Write IDOR tests (I1-I5). They will fail. Fix source (add ownership checks). Write multi-user fixture consolidation. Write auth edge case tests (I6-I13).
2. **Week 2:** Write contract tests (C1-C10). Write remaining integration tests (I14-I26). Write unit tests for untested RBAC helpers (U2-U5).
3. **Week 3:** Write property-based tests (P1-P5). Set up Playwright in CI. Write auth flow E2E tests (E1-E4).
4. **Week 4:** Write remaining E2E tests. Performance baseline. Documentation of security invariants.

### Smallest next slice

**Write and fix I1-I5 (IDOR tests).** This is the highest-impact work: it exposes real security bugs, the tests are straightforward (copy the pattern from `test_api_isolation.py`), and the fixes are mechanical (add `check_resource_owner` calls). Estimated: 1 developer-day for tests + fixes.

### Dependencies on other boundaries

- **Database-persistence boundary:** Shares engine/session fixtures. The `conftest.py` in `tests/test_server/` owns the engine fixture; DB tests likely have their own. These should be aligned so both can use the `multi_user_scenario` fixture.
- **Pipeline/task-worker boundary:** The SSE stream and background task tests mock the task worker. If the task worker boundary changes how it reports progress, the SSE tests will need updating.
- **Template rendering:** UI tests depend on Jinja2 templates in `src/rosettastone/server/templates/`. Template changes (e.g., renaming a CSS class) can break Playwright tests. The Playwright tests should use semantic selectors (roles, test-ids) rather than CSS class names.
