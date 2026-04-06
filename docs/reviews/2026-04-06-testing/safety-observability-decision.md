# Safety, Observability & Decision Intelligence -- Testing Review

**Scope:** Safety (pii_scanner, presidio_engine, prompt_auditor), Observability (server/metrics.py, server/logging_config.py, Sentry SDK), Decision (recommendation, statistics, ab_stats), Calibration (calibrator, collector, types), Shadow (config, evaluator, log_format), Clustering (cluster/embedder)

**Date:** 2026-04-06
**Author:** Testing Lead (Safety/Observability/Decision)

---

## 1. Boundary Map

### Inside (we test it -- our code, our logic)

| Module | Key Functions/Classes |
|---|---|
| `safety/pii_scanner.py` | `scan_text()`, `scan_pairs()`, `PIIWarning`, `_PII_PATTERNS` regex dict |
| `safety/presidio_engine.py` | `scan_text_presidio()`, `scan_pairs_presidio()`, `anonymize_text()`, `anonymize_pairs()`, `_severity_for()`, `_extract_prompt_text()` |
| `safety/prompt_auditor.py` | `audit_prompt()`, `AuditFinding`, boilerplate filtering, 30-char minimum, 500-char truncation |
| `decision/recommendation.py` | `make_recommendation()`, `RecommendationResult`, `Recommendation` enum, `_has_high_severity()`, threshold merge, Wilson CI gate |
| `decision/statistics.py` | `wilson_interval()`, `compute_type_stats()`, percentile calc |
| `decision/ab_stats.py` | `chi_squared_test()`, `bootstrap_ci()`, `compute_ab_significance()`, `_chi2_survival_approx()` |
| `calibration/calibrator.py` | `ThresholdCalibrator.fit()`, `.compute_alpha()`, `.report()` |
| `calibration/collector.py` | `generate_synthetic_pairs()`, `stratified_sample()` |
| `calibration/types.py` | `LabeledPair`, `CalibrationDataset`, `DimensionalScores`, `HumanLabel`, `ProductionSafety`, majority vote logic |
| `shadow/config.py` | `ShadowConfig`, `RollbackConfig`, `EndpointsConfig` (Pydantic models) |
| `shadow/evaluator.py` | `score_shadow_logs()` |
| `shadow/log_format.py` | `ShadowLogEntry`, `write_log_entry()`, `read_log_entries()` |
| `cluster/embedder.py` | `PromptClusterer.cluster()`, `._embed_prompts()`, `._fit_clusters()`, `._build_result()`, `._auto_label()`, TF-IDF fallback |
| `cluster/types.py` | `PromptCluster`, `ClusterResult` |
| `server/metrics.py` | All Counter/Gauge/Histogram definitions, `metrics_response()`, `is_available()`, helper functions (`record_stage_duration`, `record_evaluator_duration`, `record_task_queue_wait`, `record_rate_limit_hit`) |
| `server/logging_config.py` | `JsonFormatter.format()`, `configure_logging()`, `set_request_id()`, `get_request_id()` |
| `test_pii_invariant.py` | AST-based PII logging invariant check (meta-test) |

### Outside (we mock/stub)

| Dependency | Why Outside | Mock Strategy |
|---|---|---|
| **Presidio (presidio_analyzer, presidio_anonymizer)** | External NLP engine; optional dep; heavyweight init | Mock `_get_analyzer()` and `_get_anonymizer()` returning `MagicMock` with `.analyze()` / `.anonymize()` configured per test |
| **scikit-learn (sklearn.metrics.roc_curve, sklearn.cluster.KMeans/HDBSCAN, sklearn.metrics.silhouette_score, sklearn.feature_extraction.text.TfidfVectorizer)** | ML library; optional dep | `pytest.importorskip("sklearn")` gate; mock KMeans/HDBSCAN in cluster tests |
| **krippendorff** | Inter-rater reliability; optional dep | `pytest.importorskip("krippendorff")` gate |
| **sentence-transformers** | Embedding model; optional dep; GPU-heavy | Mock `_get_sentence_transformer()` returning MagicMock with `.encode()` |
| **prometheus_client** | Metrics library; optional dep | `monkeypatch` `_PROMETHEUS_AVAILABLE` flag; real prometheus_client used when available |
| **Sentry SDK (sentry_sdk)** | Error tracking; optional dep; initialized in `app.py` | Env var (`SENTRY_DSN`) driven; no test exists yet |
| **scipy.stats.chi2_contingency** | Chi-squared test; optional dep | Code has pure-Python fallback `_chi2_survival_approx()`; both paths should be tested |
| **litellm** | LLM API calls in shadow proxy | Mocked in `test_proxy.py` |
| **CompositeEvaluator** | Full evaluation pipeline in shadow evaluator | Mocked via `patch("...CompositeEvaluator.evaluate_multi_run")` |

### On the Fence

| Item | Status |
|---|---|
| **prompt_auditor "injection detection"** | NEEDS_HUMAN_REVIEW -- The brief says prompt_auditor does "injection detection" but the actual code does *verbatim training data leakage detection* (substring matching). There is NO prompt injection detection module anywhere in the codebase. This is either a naming mismatch or a missing feature. |
| **Sentry integration** | `_init_sentry()` in `app.py` initializes Sentry if `SENTRY_DSN` is set. No tests exist for this. It's a 5-line function; testing is low value but we should at least verify it doesn't crash. |
| **shadow_proxy.py** | Lives in `scripts/`, not `src/`. Tests exist in `tests/test_shadow/test_proxy.py`. On the fence because it's a deployment script, not library code. |

### Dependency Diagram

```
                      +------------------+
                      |  External World  |
                      +------------------+
                              |
        +---------------------+---------------------+
        |                     |                     |
  [Presidio]            [scikit-learn]        [sentence-
   analyzer              roc_curve             transformers]
   anonymizer             KMeans                SentenceTransformer
        |                 HDBSCAN                    |
        |                 silhouette_score            |
        |                 TfidfVectorizer             |
        v                     v                      v
  +-------------+     +--------------+      +---------------+
  | presidio_   |     | calibrator   |      | cluster/      |
  | engine.py   |     |   .fit()     |      | embedder.py   |
  +-------------+     +--------------+      +---------------+
        |                     |
        v                     v
  +-------------+     +--------------+     +--------------+
  | pii_scanner |     | statistics   |---->| recommend-   |
  | .py (regex) |     | .py          |     | ation.py     |
  +-------------+     +--------------+     +--------------+
                              |
                      +--------------+
                      | ab_stats.py  |---> [scipy.stats] (optional)
                      +--------------+

  +-------------------+      +-------------------+
  | prompt_auditor.py |      | shadow/           |
  | (leakage detect)  |      | evaluator, config,|
  +-------------------+      | log_format        |---> [CompositeEvaluator]
                             +-------------------+

  +-------------------+      +-------------------+
  | server/metrics.py |----->| [prometheus_client]|
  +-------------------+      +-------------------+
  | server/logging_   |
  | config.py         |      [sentry_sdk] <--- app.py
  +-------------------+      [krippendorff] <--- calibrator.compute_alpha()
```

---

## 2. Current Coverage Audit

### `tests/test_safety/test_pii_scanner.py` -- 33 tests

**Covers well:**
- All 5 PII pattern types (email, us_phone, ssn, credit_card, ipv4) with positive and negative cases
- Severity level correctness for each type
- `scan_text()` edge cases (empty string, no PII, multiple types, duplicate counting)
- `scan_pairs()` with string prompts, list-of-dicts prompts (both `content` and `text` keys)
- Multi-pair index tracking
- Occurrence count semantics (3 emails = count 3)
- Metadata and output_type don't interfere
- Known false positive for credit card documented as test

**Misses:**
- No tests for PII spanning prompt AND response in the same pair (count accumulation across both)
- No adversarial/evasion patterns (e.g., `j o h n @ e x a m p l e . c o m`, unicode homoglyphs, zero-width characters)
- No international phone formats (only US)
- No IPv6
- No boundary testing for SSN (e.g., `000-00-0000`, `999-99-9999`)
- Credit card regex lacks Luhn validation -- documented but no Luhn-passing vs Luhn-failing test pair
- `test_no_false_positives_for_invalid_email` is a non-assertion (just checks `isinstance(findings, list)`) -- **false confidence test**

**Verdict:** Solid for happy paths. Weak on adversarial/evasion inputs. One false-confidence test.

### `tests/test_safety/test_presidio_engine.py` -- 25 tests

**Covers well:**
- All main entity types via mocked analyzer
- scan_text_presidio, scan_pairs_presidio, anonymize_text, anonymize_pairs
- Severity map spot checks (3 tests)
- ImportError handling (3 tests)
- Preserves list-of-dicts prompt structure during anonymization
- Pair index correctness
- Both prompt and response anonymized

**Misses:**
- ZERO integration tests with real Presidio (everything is mocked)
- No test for anonymize_text with multiple overlapping entities
- No test for OperatorConfig fallback path (line 214-215: `except ImportError` for `presidio_anonymizer.entities`)
- No test for `_extract_prompt_text()` directly
- No test for thread-safety of `_get_analyzer()` double-checked locking
- No test for the `_get_anonymizer()` ImportError path
- ImportError tests (lines 480-536) manipulate `sys.modules` directly and are **brittle** -- they may interfere with other tests that import the module. The `finally` cleanup in `test_install_suggestion_in_error_text` only restores original modules, not the cached engine singletons (`_analyzer_instance`, `_anonymizer_instance`).

**Verdict:** Good contract coverage via mocks. No real Presidio integration test. ImportError tests are fragile.

### `tests/test_safety/test_prompt_auditor.py` -- 19 tests

**Covers well:**
- Empty training pairs
- No-match case
- AuditFinding dataclass fields
- Boilerplate filtering (>10% AND <50 chars)
- Long substring NOT marked as boilerplate (>= 50 chars)
- Source count tracking
- Edge cases: single pair, subset matching, longer optimized prompt, special chars, unicode, whitespace

**Misses:**
- `test_exact_response_match_detected` asserts `len(result) >= 0` which is always true -- **dead assertion / false confidence**
- `test_30_char_minimum_substring_required` only asserts `isinstance(result, list)` -- **false confidence**
- `test_substring_matching_case_sensitive` only asserts `isinstance(result, list)` -- **false confidence**
- No test for the 500-char `MAX_TEXT_LENGTH` truncation behavior
- No test for prompts being scanned (the code scans both `pair.response` AND `_flatten_prompt(pair.prompt)` but tests only put content in response)
- No performance test for the O(N*M) substring generation (could be slow with many large training pairs)
- `_flatten_prompt()` function has no direct tests

**Verdict:** Structurally complete but 3 tests give false confidence through trivial assertions.

### `tests/test_decision/test_recommendation.py` -- 19 tests

**Covers well:**
- All three recommendation outcomes: GO, NO_GO, CONDITIONAL
- High-severity safety warning blocking (single, multiple, case-insensitive)
- Medium/Low severity NOT blocking
- Plain string warnings NOT triggering NO_GO
- Below-threshold CONDITIONAL
- Custom threshold override
- Insufficient samples (< MIN_SAMPLES_FOR_CONDITIONAL and < MIN_RELIABLE_SAMPLES)
- Empty results CONDITIONAL
- Priority ordering (NO_GO beats threshold failure)
- Wilson CI lower bound gate (the critical test at line 220)
- per_type_details populated correctly

**Misses:**
- No test for multiple output types where one passes and one fails
- No test for the `"unknown"` output type path (when no output_type annotations exist)
- No test for the default threshold fallback (line 94: `effective_thresholds.get(ot, 0.80)`)
- No test for warning dict with `"msg"` key vs `"message"` key (line 120-121)

**Verdict:** Strong. Covers the critical Wilson CI gate. Minor gaps in edge paths.

### `tests/test_decision/test_statistics.py` -- 12 tests

**Covers well:**
- `wilson_interval()` edge cases: zero total, all wins, no wins, valid probabilities, symmetry at 50%, z-score widening, single trial
- `compute_type_stats()`: empty, filtering by output_type, no matching type, win rate computation, all wins, single-sample percentiles, percentile ordering, mean/median, CI validity, return type

**Misses:**
- No property-based testing (Wilson interval with random inputs)
- No test for very large n (e.g. n=10000) to verify numerical stability
- No negative score handling (scores below 0 or above 1)

**Verdict:** Good mathematical coverage. Would benefit from property-based testing.

### `tests/test_decision/test_ab_stats.py` -- 8 tests

**Covers well:**
- Chi-squared: significant difference, identical rates, empty groups, all-wins, all-losses, one-sided zero
- Bootstrap CI: clearly different, similar scores, empty lists, deterministic with seed
- `compute_ab_significance()`: full analysis, empty results

**Misses:**
- No test for the pure-Python `_chi2_survival_approx()` fallback (when scipy is absent)
- No test comparing scipy path vs pure-Python path for numerical consistency
- No test for `bootstrap_ci` with one list empty and the other non-empty
- No test for custom n_bootstrap or confidence parameters
- No test for extreme distributions (all same scores)

**Verdict:** Adequate. The scipy vs fallback path gap is notable.

### `tests/test_calibration/test_calibrator.py` -- 7 tests

**Covers well:**
- `fit()` returns all 4 output types
- Sparse types fall back to defaults
- `compute_alpha()` perfect agreement
- `report()` output format
- ImportError for missing sklearn
- FPR budget constraint verification (Bug C1 and C2 regression tests)
- Highest threshold within budget selection

**Misses:**
- No test for `compute_alpha()` with low agreement (alpha < 0.80 warning path)
- No test for `compute_alpha()` with zero or one reviewer
- No test for `compute_alpha()` with empty dataset
- No test for `compute_alpha()` with borderline labels
- No test for `fit()` with custom `fpr_targets` override
- No test for the krippendorff ImportError path

**Verdict:** Strong on the critical ROC calibration logic. Weak on alpha edge cases.

### `tests/test_calibration/test_types.py` -- 5 tests

**Covers well:**
- `majority_label` with mixed votes
- `majority_label` None when no labels
- `is_safe_majority` True
- JSON round-trip serialization
- `CalibrationDataset.by_output_type()` and `.labeled_pairs()` filtering

**Misses:**
- No test for `is_safe_majority` False (unsafe majority)
- No test for `is_safe_majority` None (borderline majority)
- No test for tie-breaking in `majority_label` (equal vote counts)
- No test for `DimensionalScores` defaults

**Verdict:** Basic coverage. Important edge cases missing (tie-breaking, unsafe/borderline majorities).

### `tests/test_shadow/test_evaluator.py` -- 2 tests

**Covers well:**
- Empty log directory returns zero summary
- 5 entries produces valid summary (mocked evaluator)

**Misses:**
- No test for malformed log entries (entries with parse errors)
- No test for mixed success/failure entries
- No test for non-deterministic flag
- No test for per_type_scores aggregation with multiple output types
- No test for cost accumulation
- Only 2 tests for a 104-line module -- low coverage density

**Verdict:** Minimal. Barely tests the happy path.

### `tests/test_shadow/test_log_format.py` -- 4 tests

**Covers well:**
- JSONL round-trip serialization
- `to_prompt_pair_dict()` structure
- Write and read multiple entries
- PromptPair compatibility

**Misses:**
- No test for `read_log_entries()` with malformed lines (the code silently skips them on line 68)
- No test for `write_log_entry()` creating directories (`mkdir(parents=True)`)
- No test for the `error` field on ShadowLogEntry
- No test for `from_jsonl_line()` with invalid JSON

**Verdict:** Adequate for happy path. Silent error swallowing in `read_log_entries()` is untested.

### `tests/test_shadow/test_proxy.py` -- 3 tests

**Covers well:**
- Primary response returned on success
- Shadow failure does not block primary
- Log entry written after request

**Misses:**
- Tests import from `scripts.shadow_proxy` -- will break if the script moves
- No test for sample_rate < 1.0 (should skip some shadow calls)
- No test for rollback configuration behavior

**Verdict:** Good for core invariant (shadow never blocks primary). Tied to script location.

### `tests/test_cluster/test_embedder.py` -- 28 tests

**Covers well:**
- ClusterResult and PromptCluster types (4 tests)
- Main clustering flow with mocked embeddings and labels (10 tests)
- Edge cases: single pair, two pairs, identical prompts, empty list, n_clusters capping, different topics, HDBSCAN noise, short prompts (8 tests)
- Embedding: ndarray return, string prompts, list-of-dict prompts, mixed formats, TF-IDF fallback, consistent dimensions (6 tests)
- Auto-label: common words, string type, max length truncation, single pair (4 tests)

**Misses:**
- ALL clustering tests mock `_embed_prompts` and/or `_fit_clusters` -- zero integration with real sklearn or sentence-transformers
- No test for `_tfidf_embed()` with real TfidfVectorizer
- No test for `_embed_prompts()` when `_get_sentence_transformer` is None (the module-level fallback path)
- No test for silhouette_score computation with real data
- No test for HDBSCAN `_fit_clusters` path with real HDBSCAN
- No test for KMeans `_fit_clusters` path with real KMeans

**Verdict:** Excellent structural coverage through mocks. Zero real ML integration. The "zero test coverage" claim in the brief is incorrect -- there ARE tests, but they don't exercise the actual ML dependencies.

### `tests/test_server/test_metrics.py` -- 7 tests

**Covers well:**
- Auth behavior (4 tests: no key, key required, valid key, wrong key)
- 404 when prometheus unavailable
- 200 when prometheus available
- Counter label verification
- New instrument existence checks
- Helper function callability

**Misses:**
- No test that helper functions actually *record* values to the registry
- No test for `metrics_response()` content format
- No test for `record_rate_limit_hit` hashing behavior
- No test for Histogram bucket configurations

**Verdict:** Good existence/shape checks. No functional verification that metrics are actually recorded.

### `tests/test_server/test_logging.py` -- 5 tests

**Covers well:**
- X-Request-ID header presence and UUID format
- JsonFormatter valid JSON output with required fields
- Request ID consistency and uniqueness
- Extra fields (request_id, migration_id, duration_ms)
- Migration complete log event fields

**Misses:**
- No test for `configure_logging()` function
- No test for `get_request_id()` / `set_request_id()` context var propagation
- No test for handler deduplication (lines 150-152)
- No test for exception formatting (exc_info path)
- No test for env var `ROSETTASTONE_LOG_LEVEL` override

**Verdict:** Good for JSON format. Weak on configure_logging and context propagation.

### `tests/test_pii_invariant.py` -- 1 test

**Covers:**
- AST scan of all source files for logging calls that pass PII variable names

**Assessment:** Clever meta-test. Correctly uses exact name matching to avoid false positives. Only catches `ast.Name` nodes (local variables), not attribute access like `pair.prompt`. This is a reasonable tradeoff -- `pair.prompt` would be hard to detect reliably via AST. Covers the invariant well.

### Collector Module -- 0 tests

`calibration/collector.py` has `generate_synthetic_pairs()` and `stratified_sample()`. No tests exist for either function. The synthetic pair generator IS used as test infrastructure (indirectly by calibrator tests building their own pairs), but `collector.py` itself has zero test coverage.

---

## 3. Risk Ranking

| # | Failure Mode | Manifestation | Existing Tests Catch It? | Severity |
|---|---|---|---|---|
| 1 | **PII leaks through regex evasion** | Attacker uses Unicode homoglyphs, zero-width chars, or space-separated digits to bypass pii_scanner regex. Real PII passes undetected into logs/reports. | No. No adversarial input tests. | **CRITICAL** |
| 2 | **Wrong GO/NO_GO decision from Wilson CI** | Numerical error in `wilson_interval()` or threshold comparison produces incorrect GO when data is uncertain, leading to premature production migration. | Partially. One specific CI test exists (n=12). No property-based fuzzing. | **CRITICAL** |
| 3 | **Uncalibrated thresholds from ROC bugs** | `ThresholdCalibrator.fit()` picks wrong threshold due to fpr/threshold array misalignment (Bug C1/C2). Migrations approved with unacceptable false positive rates. | Yes. Two regression tests verify FPR budget and direction. | **HIGH** (mitigated) |
| 4 | **Silent Prometheus metric loss** | `_PROMETHEUS_AVAILABLE` is False in production; all `record_*` helpers silently no-op. Monitoring dashboards show no data, but no alert fires because the absence looks like zero activity. | Partially. Test verifies 404 response but not the no-op behavior's operational impact. | **HIGH** |
| 5 | **PII logged to stdout/Sentry** | A developer adds `logger.info("prompt: %s", prompt)`. The `test_pii_invariant.py` meta-test catches local variable names but misses attribute access patterns like `entry.prompt`. | Partially. AST scan catches `Name` nodes only. | **HIGH** |
| 6 | **Missed training data leakage** | `prompt_auditor.py` truncates responses at 500 chars (`MAX_TEXT_LENGTH`). Verbatim content after char 500 leaks into optimized prompt undetected. | No. No test for truncation behavior. | **MEDIUM** |
| 7 | **Shadow evaluator masks failures** | `read_log_entries()` silently swallows malformed lines (bare `except`). Corrupted logs produce partial results that look like valid low-win-rate data. | No. No malformed-line test. | **MEDIUM** |
| 8 | **majority_label tie-break nondeterminism** | `LabeledPair.majority_label` uses `max()` on a dict. With equal vote counts, Python's `max()` returns an arbitrary winner depending on iteration order. Calibration results become nondeterministic. | No. No tie-break test. | **MEDIUM** |
| 9 | **Chi-squared fallback numerical drift** | `_chi2_survival_approx()` is used when scipy is absent. The Wilson-Hilferty approximation is "accurate to ~0.01" but borderline p-values near 0.05 could flip significance. | No. No scipy vs fallback comparison test. | **MEDIUM** |
| 10 | **Cluster embedder ImportError cascade** | If numpy is installed but sentence-transformers AND sklearn are both missing, `_embed_prompts()` raises ImportError. If only sklearn is missing, `_tfidf_embed()` raises. No graceful degradation. | Partially. TF-IDF fallback test exists but only with mocked `_tfidf_embed`. | **LOW** |

---

## 4. Test Plan by Tier

### Tier 1: Unit Tests

| Test | Module | Assertions | Size | Status |
|---|---|---|---|---|
| PII regex: Unicode homoglyph evasion (Cyrillic `a` in email) | pii_scanner | No match for homoglyph-substituted email | S | MISSING |
| PII regex: zero-width character insertion | pii_scanner | No false negative when zwj inserted in SSN | S | MISSING |
| PII regex: space-padded credit card | pii_scanner | `4 5 3 2 1 2 3 4 5 6 7 8 9 0 1 0` is NOT detected (documented false negative) | S | MISSING |
| PII regex: SSN boundary values (000-00-0000, 999-99-9999) | pii_scanner | Detected as SSN | S | MISSING |
| PII scan_pairs: count accumulation across prompt+response | pii_scanner | Single warning with count=2 when email in both prompt and response | S | MISSING |
| Wilson interval: large n numerical stability | statistics | `wilson_interval(9999, 10000)` produces valid bounds | S | MISSING |
| Wilson interval: negative wins (invalid input) | statistics | Graceful handling (return 0,0 or raise) | S | MISSING |
| make_recommendation: unknown output_type path | recommendation | Empty output_type annotations use "unknown" | S | MISSING |
| make_recommendation: mixed types (one GO, one fail) | recommendation | CONDITIONAL with correct reasoning | S | MISSING |
| make_recommendation: warning with "msg" key (not "message") | recommendation | msg value appears in reasoning | S | MISSING |
| _has_high_severity: non-dict, non-string input | recommendation | Returns False | S | MISSING |
| _chi2_survival_approx: known p-values | ab_stats | Approx matches scipy for chi2=3.84 (p~0.05) and chi2=6.63 (p~0.01) | S | MISSING |
| bootstrap_ci: one empty, one non-empty | ab_stats | Returns (0.0, 0.0, 0.0) | S | MISSING |
| majority_label: tie-break behavior | types | Document/define behavior for 1 SAFE + 1 UNSAFE | S | MISSING |
| is_safe_majority: False for unsafe majority | types | Returns False | S | MISSING |
| is_safe_majority: None for borderline majority | types | Returns None | S | MISSING |
| generate_synthetic_pairs: correct count and score distribution | collector | n_pairs=100 returns 100 pairs, 10 per bucket | S | MISSING |
| stratified_sample: respects n_per_bucket | collector | Never more than n_per_bucket per score bucket | S | MISSING |
| prompt_auditor: MAX_TEXT_LENGTH truncation | prompt_auditor | Content after char 500 NOT scanned | S | MISSING |
| prompt_auditor: prompt text scanned (not just response) | prompt_auditor | Substring from prompt (not response) detected | S | MISSING |
| ShadowConfig: Pydantic validation | config | sample_rate > 1.0 rejected, duration_hours < 1 rejected | S | EXISTS (implicit via Pydantic) |
| ShadowLogEntry: error field populated | log_format | Entry with error=string serializes correctly | S | MISSING |
| record_rate_limit_hit: hash determinism | metrics | Same user_key always produces same hash prefix | S | MISSING |
| configure_logging: env var override | logging_config | ROSETTASTONE_LOG_LEVEL=DEBUG sets root level to DEBUG | S | MISSING |
| configure_logging: handler dedup | logging_config | Calling configure_logging twice doesn't create 2 handlers | S | MISSING |
| JsonFormatter: exception formatting | logging_config | exc_info key present when exception logged | S | MISSING |

### Tier 2: Contract Tests

| Test | Module | Assertions | Size | Status |
|---|---|---|---|---|
| Presidio engine: scan_text_presidio return type matches scan_text | presidio_engine | Both return list of tuples with same semantics | M | MISSING |
| Presidio engine: scan_pairs_presidio returns PIIWarning with same fields | presidio_engine | Same PIIWarning dataclass, same field semantics | S | EXISTS |
| Prometheus metric shapes: Counter has .labels(), .inc() | metrics | MIGRATIONS_TOTAL.labels(status="x", source_model="y", target_model="z").inc() doesn't raise | S | PARTIAL |
| Prometheus metric shapes: Histogram has .observe() with bucket config | metrics | MIGRATION_COST_USD has configured buckets [0.01, ..., 50.0] | S | MISSING |
| CalibrationDataset contract: labeled_pairs filters unlabeled | types | Only pairs with labels returned | S | EXISTS |
| ShadowLogEntry <-> PromptPair: to_prompt_pair_dict produces valid PromptPair | log_format | PromptPair(**dict) succeeds | S | EXISTS |

### Tier 3: Integration Tests

| Test | Module | Assertions | Size | Status |
|---|---|---|---|---|
| Presidio real analyzer: detect email in plain text | presidio_engine | Real AnalyzerEngine detects "john@example.com" | L | MISSING |
| Presidio real anonymizer: replace email with placeholder | presidio_engine | Real AnonymizerEngine produces `<EMAIL_ADDRESS>` | L | MISSING |
| Prometheus scrape endpoint: /metrics returns parseable text | metrics + app | Response body contains `migrations_total` metric family | M | PARTIAL (test_metrics_endpoint_returns_200) |
| sklearn roc_curve + calibrator.fit: end-to-end threshold calibration | calibrator | fit() on synthetic data produces threshold that satisfies FPR constraint | L | EXISTS |
| krippendorff alpha with real library | calibrator | compute_alpha with known labels matches expected value | M | EXISTS |
| KMeans clustering with real sklearn | embedder | _fit_clusters with real KMeans produces valid labels | M | MISSING |
| TF-IDF embedding with real sklearn | embedder | _tfidf_embed with real TfidfVectorizer produces valid ndarray | M | MISSING |
| Shadow evaluator + real log files: full pipeline | evaluator | Write entries, score them, verify summary dict | L | PARTIAL (mocked evaluator) |

### Tier 4: Property-Based Tests (Hypothesis)

| Test | Module | Strategy | Size | Status |
|---|---|---|---|---|
| PII regex: no false negatives for structurally valid emails | pii_scanner | `hypothesis.strategies.emails()` | M | MISSING |
| PII regex: no crash on arbitrary unicode | pii_scanner | `hypothesis.strategies.text()` | M | MISSING |
| Wilson CI: bounds always in [0,1], lower <= upper | statistics | `st.integers(0,N), st.integers(0,N)` with wins<=total | M | MISSING |
| Wilson CI: larger n narrows interval | statistics | Fixed p, increasing n | M | MISSING |
| Wilson CI: monotonicity -- more wins = higher lower bound | statistics | Fixed n, increasing wins | M | MISSING |
| Chi-squared pure-Python vs scipy agreement | ab_stats | Random contingency tables, compare p-values within tolerance | L | MISSING |
| Bootstrap CI: contains true mean difference for normal distributions | ab_stats | Generate from known distributions, verify coverage | L | MISSING |
| Calibration ROC: fit threshold is monotonic with FPR target | calibrator | Tighter FPR target -> higher threshold | L | MISSING |
| Prompt auditor: never returns substrings shorter than 30 chars | prompt_auditor | Random training data + optimized prompts | M | MISSING |
| PII scanner: scan_pairs occurrence_count = sum of scan_text counts | pii_scanner | Random pairs | M | MISSING |

### Tier 5: End-to-End Tests

| Test | Module | Assertions | Size | Status |
|---|---|---|---|---|
| Full safety scan pipeline: PII scan + audit + recommendation | pii_scanner + prompt_auditor + recommendation | PII warnings trigger NO_GO when HIGH severity | XL | MISSING |
| Calibration pipeline: generate_synthetic -> fit -> report | collector + calibrator | Report contains calibrated thresholds for all types | L | MISSING |
| Shadow pipeline: write logs -> read logs -> score | log_format + evaluator | End-to-end without mocks (except LLM calls) | XL | MISSING |

---

## 5. Synthetic Data Generation Strategy

### PII Test Data

**Approach:** Hand-crafted patterns that look realistic but are provably fake.

| PII Type | Generation Method | Examples |
|---|---|---|
| Email | `{first}.{last}@example.{tld}` using RFC 2606 reserved domains | `john.doe@example.com`, `jane_smith123@example.org` |
| US Phone | `555-01XX-XXXX` range (reserved for fictional use by NANPA) | `555-0100-1234`, `(555) 012-3456` |
| SSN | Use IRS-documented never-issued ranges: `987-65-XXXX` (advertising), `000-XX-XXXX`, `XXX-00-XXXX` | `987-65-4321`, `000-12-3456` |
| Credit Card | Luhn-invalid 16-digit sequences from test ranges (Stripe test: `4242424242424242`) | `4242-4242-4242-4242`, `0000-0000-0000-0000` |
| IPv4 | RFC 5737 documentation ranges: `192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24` | `192.0.2.1`, `198.51.100.42` |

**Stability:** Fixtures are deterministic strings, not randomly generated. Stored as constants in a shared test module.

**Rot prevention:** PII patterns in source code change rarely. If they do, the AST-based `test_pii_invariant.py` and regex unit tests will catch regressions.

### Calibration Datasets

**Approach:** Use `calibration/collector.py`'s `generate_synthetic_pairs()` with fixed seeds.

```
generate_synthetic_pairs("json", n_pairs=100, seed=42)
generate_synthetic_pairs("classification", n_pairs=100, seed=42)
```

The function already distributes scores across 10 buckets (0.0-1.0). For multi-reviewer tests, add synthetic `HumanLabel` entries with controlled agreement rates.

**Stability:** Seeded RNG. Deterministic outputs.

### Evaluation Score Distributions

For property-based tests of Wilson CI and recommendation logic:

| Distribution | Use Case |
|---|---|
| All 1.0 (perfect scores) | GO path verification |
| All 0.0 (zero scores) | NO_GO path verification |
| Beta(2,5) bimodal | Realistic mixed-quality scores |
| Uniform(0,1) | Stress test for percentile computation |
| Single value repeated N times | Edge case for variance=0 |

**Cost profile:** All synthetic. Zero API calls. Sub-second generation.

---

## 6. Fixtures, Fakes, and Mocks

### New Fixtures Needed

| Fixture | Scope | Shared? | Description |
|---|---|---|---|
| `fake_pii_texts` | session | Yes (with any PII-consuming test) | Dict mapping PII type -> list of provably-fake PII strings |
| `pii_pairs_with_known_findings` | function | No | Pre-built `PromptPair` list with known PII counts per type |
| `mock_presidio_analyzer` | function | No | Preconfigured MagicMock with `.analyze()` returning entity results |
| `calibration_dataset_100` | session | Cross-area (evaluation-strategies) | `CalibrationDataset` with 100 labeled pairs across 4 output types, seed=42 |
| `calibration_dataset_skewed` | function | No | Dataset with all labels SAFE (for alpha=1.0 test) |
| `shadow_log_dir` | function | No | `tmp_path` with 10 pre-written `ShadowLogEntry` JSONL files |
| `eval_results_factory` | session | Cross-area (evaluation-strategies) | Factory function: `make_eval_results(output_type, n, win_rate)` -> `list[EvalResult]` |

### Cross-Subagent Shared Fixtures

| Fixture | This Area Uses | Other Area Uses |
|---|---|---|
| `eval_results_factory` | recommendation.py, statistics.py tests | evaluation-strategies tests for CompositeEvaluator output |
| `calibration_dataset_100` | calibrator tests | evaluation-strategies for score distribution analysis |
| `sample_pairs` (existing in conftest.py) | pii_scanner, prompt_auditor | All areas |

### Mocks to Retain / Add

| Mock | Currently Exists | Notes |
|---|---|---|
| `_get_analyzer` -> MagicMock | Yes | Keep as-is. Add integration tests alongside. |
| `_get_anonymizer` -> MagicMock | Yes | Keep as-is. |
| `_get_sentence_transformer` -> MagicMock | Yes | Keep as-is. |
| `CompositeEvaluator.evaluate_multi_run` -> mock results | Yes | Keep as-is. Shadow evaluator can't run real evals without LLM. |
| `sklearn.metrics.roc_curve` | No (uses real sklearn) | Keep using real sklearn -- calibration tests are meaningless without it. |
| `scipy.stats.chi2_contingency` | No (tested implicitly) | Need explicit mock to test fallback path. |

---

## 7. Gaps You Can't Close

| Gap | Why | Mitigation |
|---|---|---|
| **Real Presidio accuracy** | Presidio's NER model quality is upstream. We can verify our integration but not that Presidio correctly identifies "John Smith" as a person in all contexts. | Contract tests verify our wiring. Presidio's own test suite covers accuracy. |
| **Sentence-transformers embedding quality** | Whether `all-MiniLM-L6-v2` produces good clusters for our domain is an ML evaluation question, not a software test. | Could add a golden-set test with known-good clusters, but this is more ML eval than testing. |
| **Sentry error capture in production** | We can verify `_init_sentry()` doesn't crash, but testing that errors actually reach Sentry requires a real Sentry DSN and network access. | Smoke test the init. Rely on Sentry's own reliability guarantees. |
| **Prometheus scrape reliability** | Whether a production Prometheus server successfully scrapes the `/metrics` endpoint depends on network config, not our code. | Verify the response format is valid Prometheus exposition format. |
| **Krippendorff alpha edge cases** | The `krippendorff` library's behavior with degenerate inputs (all-same labels, single column) is upstream. | Document expected behavior. Guard with minimum reviewer/pair counts. |
| **Real LLM responses in shadow evaluation** | `score_shadow_logs()` ultimately needs LLM responses to evaluate. We mock `CompositeEvaluator` because real LLM calls are expensive and nondeterministic. | Mock evaluator for unit tests. Real LLM integration in a separate expensive test suite (out of scope). |

---

## 8. Cost and Time Estimate

| Tier | Test Count | Write Time | Run Time (per CI) | Dependencies |
|---|---|---|---|---|
| **Unit (Tier 1)** | ~26 new tests | 2-3 days | < 5 seconds | None |
| **Contract (Tier 2)** | ~4 new tests | 0.5 day | < 2 seconds | None |
| **Integration (Tier 3)** | ~5 new tests | 1-2 days | 10-30 seconds (Presidio init, sklearn fit) | presidio-analyzer, presidio-anonymizer, sklearn |
| **Property-Based (Tier 4)** | ~10 new tests | 2-3 days | 5-15 seconds (Hypothesis shrinking) | hypothesis |
| **End-to-End (Tier 5)** | ~3 new tests | 1-2 days | 5-10 seconds | All optional deps |

**Total: ~48 new tests, 7-11 days of focused work.**

**CI cost impact:** Minimal. The expensive operations (Presidio engine init, sklearn fits) take < 30 seconds total. Hypothesis tests are configurable via `@settings(max_examples=...)`. No LLM API calls.

**Priority order for maximum risk reduction per hour:**

1. Unit tests for Wilson CI property-based (Risk #2) -- 1 day, CRITICAL
2. Unit tests for PII regex evasion (Risk #1) -- 0.5 day, CRITICAL
3. Unit tests for majority_label tie-break (Risk #8) -- 0.5 day, MEDIUM but prevents nondeterminism
4. Contract test: Presidio return type compatibility (Risk #1) -- 0.5 day
5. Fix false-confidence tests in prompt_auditor -- 0.5 day
6. Property-based chi-squared fallback vs scipy (Risk #9) -- 1 day
7. Everything else

---

## 9. Path to Production

### Current Readiness Level

| Area | Level | Assessment |
|---|---|---|
| **PII Scanner (regex)** | **Yellow** | Good happy-path tests. No adversarial/evasion tests. CRITICAL gap for a safety module. |
| **Presidio Engine** | **Yellow** | Good mock coverage. Zero real integration tests. |
| **Prompt Auditor** | **Yellow** | 3 false-confidence tests. No truncation test. |
| **Recommendation Engine** | **Green** | Strong coverage including Wilson CI gate. Minor edge gaps. |
| **Statistics** | **Green** | Good mathematical coverage. Needs property-based hardening. |
| **A/B Stats** | **Yellow** | No fallback path test. No scipy vs Python comparison. |
| **Calibrator** | **Green** | Good ROC regression tests. Weak alpha edge cases. |
| **Collector** | **Red** | Zero tests. |
| **Calibration Types** | **Yellow** | Missing tie-break, unsafe/borderline majority tests. |
| **Shadow Evaluator** | **Red** | Only 2 tests. Minimal coverage. |
| **Shadow Log Format** | **Yellow** | No malformed line test despite silent error swallowing. |
| **Shadow Proxy** | **Yellow** | Works but tied to script location. |
| **Cluster Embedder** | **Yellow** | 28 tests but all mocked. Zero real ML. |
| **Server Metrics** | **Yellow** | Existence checks only. No functional verification. |
| **Server Logging** | **Yellow** | Good format tests. No configure_logging or context var tests. |
| **PII Invariant** | **Green** | Clever meta-test. Catches the obvious cases. |

### Gap to Production-Hardened

| Production Gate | Current State | Action Required | Blocking? |
|---|---|---|---|
| **X2: PII enforcement** | Regex scanner works for standard patterns. No adversarial resilience. No Presidio integration test. | Add evasion tests (Tier 1), property-based fuzzing (Tier 4), at least one real Presidio integration test (Tier 3). Fix `test_no_false_positives_for_invalid_email`. | **Yes** |
| **T2.1: Calibration** | Calibrator ROC logic tested. Collector untested. Types partially tested. Alpha edge cases missing. | Add collector tests (Tier 1), alpha edge cases (Tier 1), tie-break test (Tier 1). | **Yes** |
| **T2.5: Shadow** | Evaluator barely tested. Log format adequate. Proxy tested but fragile. | Add evaluator tests for malformed logs, mixed results, per-type aggregation (Tier 1). | **Yes** |
| **Embedder coverage** | 28 tests, all mocked. | Add at least one real KMeans and one real TF-IDF integration test (Tier 3). | **No** (not on critical path) |
| **Observability confidence** | Metrics exist-checks only. No recording verification. | Add functional metric recording tests (Tier 1). Test configure_logging (Tier 1). | **No** (operational, not safety) |

### Ordered Sequence

1. **Smallest next slice:** Fix the 3 false-confidence tests in `test_prompt_auditor.py` and the 1 in `test_pii_scanner.py`. Zero new tests needed -- just strengthen assertions. (0.5 day)

2. **PII hardening:** Add adversarial regex tests + property-based email/unicode fuzzing. (1.5 days)

3. **Collector tests:** Add unit tests for `generate_synthetic_pairs()` and `stratified_sample()`. Unblocks calibration gate. (0.5 day)

4. **Wilson CI property tests:** Add Hypothesis-based tests for bounds, monotonicity, numerical stability. (1 day)

5. **Shadow evaluator tests:** Add malformed log, mixed results, multi-type tests. (1 day)

6. **Calibration types edge cases:** Tie-break, unsafe/borderline majority, compute_alpha edge cases. (0.5 day)

7. **Integration tests:** Presidio real analyzer, real KMeans, real TF-IDF. (1.5 days)

8. **A/B stats fallback:** scipy vs pure-Python comparison test. (0.5 day)

9. **Observability:** Metrics recording, configure_logging, context vars. (1 day)

10. **End-to-end:** Full safety pipeline, calibration pipeline. (1.5 days)

### Dependencies

- Steps 1-6 have no external dependencies (all use stdlib + existing test deps)
- Step 7 requires `presidio-analyzer`, `scikit-learn` in test environment
- Step 4 requires `hypothesis` package
- Steps can be parallelized: steps 2+3+5+6 are independent. Steps 4+8 are independent.

### NEEDS_HUMAN_REVIEW Items

1. **prompt_auditor.py "injection detection"** -- The brief describes this module as handling "injection detection" but the code only detects verbatim training data leakage. Is there a separate injection detection module planned? If so, it doesn't exist yet and needs to be built before this area can be considered production-ready.

2. **majority_label tie-break** -- `LabeledPair.majority_label` has undefined behavior on ties. The code uses `max(counts, key=...)` which picks the first max key in iteration order. This should be explicitly defined (e.g., "BORDERLINE on tie" or "raise on tie") and tested.

3. **Credit card false positive rate** -- The regex has a documented high false-positive rate with no Luhn validation. Is this acceptable for production, or should Luhn be added? The Presidio engine has better credit card detection but requires the optional dependency.
