# Evaluation Strategies — Testing Review

**Date:** 2026-04-06
**Scope:** `src/rosettastone/evaluate/`, `src/rosettastone/decision/`, and their tests.
**Boundary owner:** Evaluation Strategies testing lead

---

## 1. Boundary Map

### Inside (we test it)

```
evaluate/
  base.py             Evaluator ABC (score interface)
  types.py            detect_output_type, _strip_code_fence
  exact_match.py      ExactMatchEvaluator, string_similarity
  json_validator.py   JSONEvaluator, _strip_fences
  json_structural.py  JSONStructuralEvaluator, _coerce_match, _lcs_length,
                       _extract_keypaths, _array_sim, _dict_compare, _compare
  bertscore.py        BERTScoreEvaluator, compute_bertscore, batch_compute_bertscore
  embedding.py        EmbeddingEvaluator, compute_embedding_sim, _get_sentence_transformer
  llm_judge.py        LLMJudgeEvaluator, _build_messages, _parse_score, _normalize
  composite.py        CompositeEvaluator (evaluate, evaluate_multi_run, _aggregate_runs,
                       _score, _score_semantic, _composite_score, _get_threshold)
                      EVALUATOR_ROUTING, METRIC_WEIGHTS, DEFAULT_WIN_THRESHOLDS

decision/
  statistics.py       wilson_interval, compute_type_stats (incl. percentile helper)
  recommendation.py   make_recommendation, Recommendation enum, RecommendationResult,
                       DEFAULT_THRESHOLDS, MIN_RELIABLE_SAMPLES, MIN_SAMPLES_FOR_CONDITIONAL
  ab_stats.py         chi_squared_test, _chi2_survival_approx, bootstrap_ci,
                       compute_ab_significance, ABSignificanceResult
```

### Outside (we mock/stub)

| Dependency | Mock strategy |
|---|---|
| `litellm.completion` | `unittest.mock.patch` — all calls in CompositeEvaluator.evaluate and LLMJudgeEvaluator.score |
| `bert_score.score` | `patch.dict("sys.modules")` with MagicMock returning torch tensors |
| `sentence_transformers.SentenceTransformer` | `patch.dict("sys.modules")` with MagicMock returning numpy arrays |
| `scipy.stats.chi2_contingency` | Tested via ImportError fallback path; no explicit mock needed |
| `rosettastone.server.metrics.record_evaluator_duration` | Swallowed by try/except in production code; not mocked in tests |
| `rosettastone.core.context.PipelineContext` | Passed as `ctx=None` or as a mock when testing cost accumulation |

### On the fence (integration candidates)

| Item | Notes |
|---|---|
| `bert_score` with real distilbert model | Could run locally if CI has GPU/CPU budget; ~2s per call |
| `sentence_transformers` with real all-MiniLM-L6-v2 | ~200MB download; could run in a dedicated integration job |
| LLM judge against a real cheap model | Real-money test; VCR cassettes preferred |

### Dependency tree (simplified)

```
CompositeEvaluator
  |-- litellm.completion          [MOCK]
  |-- detect_output_type          [PURE]
  |-- JSONEvaluator               [PURE]
  |-- JSONStructuralEvaluator     [PURE]
  |-- ExactMatchEvaluator         [PURE]
  |-- BERTScoreEvaluator          [OPTIONAL_DEP: bert_score]
  |   '-- batch_compute_bertscore [OPTIONAL_DEP: bert_score]
  |-- EmbeddingEvaluator          [OPTIONAL_DEP: sentence_transformers, numpy]
  |-- LLMJudgeEvaluator           [MOCK: litellm]
  |-- _composite_score            [PURE]
  |-- _aggregate_runs             [PURE]
  '-- _get_threshold              [PURE]

make_recommendation
  |-- compute_type_stats          [PURE]
  |   '-- wilson_interval         [PURE]
  '-- _has_high_severity          [PURE]

compute_ab_significance
  |-- chi_squared_test            [PURE, optional scipy]
  '-- bootstrap_ci                [PURE, uses random.Random]
```

---

## 2. Current Coverage Audit

### `tests/test_evaluate/test_bertscore.py` (7 tests)

**Covers:** BERTScoreEvaluator.score return structure, value range, correct argument order to bert_score.score, ImportError propagation, single-key output.

**Misses:**
- `batch_compute_bertscore` — zero tests. This is the hot path used by CompositeEvaluator phase 2.
- Empty string inputs.
- Very long input strings (token limits of distilbert).
- Tests depend on `import torch` being available at test time to build mock tensors. If torch is missing, the test file itself errors on import, not on the bert_score mock.

**Brittleness:** Every test clears `sys.modules["rosettastone.evaluate.bertscore"]` manually. If the module has been imported earlier in the test session through a different path (e.g., composite imports it), the delete may not fully reset state. This is fragile but has not caused failures yet.

### `tests/test_evaluate/test_embedding.py` (10 tests)

**Covers:** EmbeddingEvaluator.score return structure, value range, identical/orthogonal vectors, correct model name, zero-norm guard, ImportError propagation.

**Misses:**
- `_get_sentence_transformer` LRU cache behavior — tests never verify that repeated calls reuse the cached model. The LRU cache is `maxsize=4`, and cache pollution across tests is a real risk.
- Negative cosine similarity values (anti-correlated embeddings) — the evaluator can return negative values but no test checks behavior below 0.

**Brittleness:** Same `sys.modules` clearing pattern as bertscore tests. Same risk.

### `tests/test_evaluate/test_exact_match.py` (14 tests)

**Covers:** Exact match (case-insensitive), string_similarity (case-sensitive), whitespace stripping, return keys, value types, value ranges, partial matches.

**Misses:** Nothing significant. This evaluator is simple and well-tested. Could add Unicode normalization tests (e.g., accented characters, NFC vs NFD) but that's low priority.

**Status:** Good.

### `tests/test_evaluate/test_json_validator.py` (16 tests)

**Covers:** Identical dicts, partial key overlap, disjoint keys, invalid JSON (expected, actual, both), arrays (equal, different), primitives (string, number), empty dicts, nested dicts, range checks.

**Misses:**
- The `_strip_fences` function in json_validator.py is more sophisticated than the one in types.py (handles thinking prefix, bare JSON at end of text). The fence-stripping tests are in a separate file but don't cover the thinking-prefix path or the bare-JSON-at-end fallback.
- Deeply nested JSON with mixed arrays and dicts.
- Very large JSON objects (performance boundary).
- JSON with special float values (NaN, Infinity) — `json.loads` rejects these by default, but the behavior should be verified.

### `tests/test_evaluate/test_json_fence_stripping.py` (11 tests)

**Covers:** Fenced JSON with `json` tag, plain fences, both fenced, unfenced regression, mixed fencing, fenced non-JSON, different values, extra whitespace, complex JSON.

**Misses:**
- Thinking prefix before fence block (the code explicitly handles this in `_strip_fences` with a `re.search` fallback).
- Bare JSON after non-JSON thinking text (third branch in `_strip_fences`).
- Multiple fence blocks in one response (code takes the first `re.search` match).
- Fence with language tags other than `json` (e.g., `` ```python ``).

### `tests/test_evaluate/test_json_structural.py` (27 tests)

**Covers:** `_coerce_match` (equal, strings, int/string coercion, None, bool/int), `_lcs_length` (empty, identical, disjoint, partial, single element), evaluator interface (return keys, types, ranges, non-JSON inputs, empty strings, malformed JSON), identical structures (flat dict, nested dict, array, empty dict/array, scalar, number), schema match (same keys different values, partial overlap, disjoint, superset), structural similarity (same/different values, type coercion, nested full/partial match), arrays (identical, different, partial, empty, one empty, different lengths, LCS schema match), edge cases (whitespace, scalar vs dict, dict vs array, no config, deep nesting, boolean, null).

**Misses:**
- `_extract_keypaths` — zero direct tests. This function is tested indirectly through the evaluator, but edge cases (root-level scalar, array of dicts, deeply nested arrays) are not exercised.
- `_array_sim` — zero direct tests. The unordered value matching logic (greedy with `used_b` set) has subtle behavior with duplicate elements that is never tested.
- Arrays of objects (common in real API responses).
- Mixed-type arrays (e.g., `[1, "two", null, {"k": "v"}]`).

**False confidence risk:** The structural similarity tests are thorough for simple cases but don't exercise the recursive `_compare` → `_dict_compare` → `_compare` cycle deeply enough. A bug in recursion depth > 3 would go unnoticed.

### `tests/test_evaluate/test_llm_judge.py` (16 tests)

**Covers:** `_parse_score` (bare digit, whitespace, in-sentence, first valid, None for no digit, out of range, zero, all valid), `_normalize` (endpoints and midpoints), `_build_messages` (default/flipped order, prompt inclusion/absence, system message, rubric), LLMJudgeEvaluator.score (perfect/worst/middle, config model, default model, API error, parse failure, empty inputs, range, prompt forwarding, bidirectional calls, single key, second-call failure).

**Misses:**
- Cost tracking via `self._ctx.add_cost` — never tested. A mock PipelineContext would verify costs accumulate.
- `_parse_score` with multi-digit numbers containing 1-5 (e.g., "15 points" — the `\b([1-5])\b` regex should match "1" in "15", which is arguably wrong).
- Temperature and max_tokens params passed to litellm — not verified.

### `tests/test_evaluate/test_output_types.py` (22 tests)

**Covers:** JSON detection (dict, numeric, array, boolean), empty/whitespace, classification (single word, 5 words), short text (6 words, 50 words), long text (51 words), newline prevention, whitespace stripping, invalid JSON, multiline, code fence stripping (json tag, plain, array, non-JSON, empty dict, case-insensitive, unfenced regression), classify/label override (numeric+classify, numeric+label, no classify, unrelated prompt, float, categorize, tag, dict stays JSON, None prompt, case-insensitive keyword).

**Misses:**
- The `categorise` (British spelling) keyword is in `_CLASSIFY_KEYWORDS` but never tested.
- Edge case: response is `"null"` (valid JSON) — classified as JSON, but could confuse downstream evaluators.
- Response is a JSON string that looks like classification (e.g., `'"positive"'`).

**Status:** Good overall.

### `tests/test_evaluate/test_composite.py` (19 tests)

**Covers:** `_composite_score` (empty, single, all zeros/ones, JSON gating, json_valid excluded, weighted average), win thresholds (defaults exist, JSON strictest, config override), `_score` routing (JSON, classification, short_text fallback, long_text fallback), `evaluate()` integration (JSON/classification scoring, is_win true/false, result structure, optimized prompt prepended, multiple pairs, empty test set, null content, perfect JSON match, output_type in details, progress callback, skipped pairs warning, failure reasons: None/api_error/timeout/json_gate/rate_limit/no_response).

**Misses:**
- **`evaluate_multi_run`** integration via `evaluate()` — never tested here; deferred to test_multi_run.py.
- **Semantic fallback chain** — the `_score_semantic` method's 3-tier fallback (BERTScore → embedding → exact_match) is tested only via `_score` routing tests that block both optional deps. The intermediate case (BERTScore unavailable, embedding available) is never tested.
- **Batch BERTScore integration** — phase 2 of evaluate() calls `batch_compute_bertscore` for free-text pairs. This is never tested in the composite tests.
- **Thread pool behavior** — `num_threads` config param controls worker count. Never tested with different values. Thread-safety of the progress callback is never verified.
- **Cost accumulation** — `self._ctx.add_cost("evaluation", total_eval_cost)` is never tested.
- **Token tracking** — `self._ctx.add_tokens(...)` is never tested.
- **`>20% skip rate` error log** — tested indirectly by the skipped pairs test (3/5 = 60%), but the test doesn't assert the specific error-level log.

**Brittleness:** The `test_logs_warning_on_skipped_pairs` test creates side-effect mocking with a counter, which is order-dependent when threads are involved. It currently works because `ThreadPoolExecutor.map` processes items in order, but this is an implementation detail.

### `tests/test_evaluate/test_multi_run.py` (8 tests)

**Covers:** Single run passthrough, 3-run with evaluate called 3 times, aggregation strategies (median, mean, p25), variance flagging (above/below threshold), mismatched skips alignment by pair identity, `build_result` integration (non_deterministic_count, eval_runs).

**Misses:**
- `_aggregate_runs` with all identical scores (stdev=0).
- `_aggregate_runs` with a single run_eval (len=1, which can't happen via evaluate_multi_run but tests the edge case).
- `_aggregate_runs` preserving `base.scores` from first run — never asserted.
- The `is_win` recalculation in `_aggregate_runs` uses `_get_threshold` with output type detection — never tested with a pair whose `output_type` is None (forces `detect_output_type` call).

### `tests/test_decision/test_statistics.py` (11 tests)

**Covers:** `wilson_interval` (zero total, all wins, no wins, valid probabilities, 50% midpoint, larger z widens, single trial), `compute_type_stats` (empty, filtering by type, no matching type, win rate, all wins, single sample percentiles, percentile ordering, mean/median, CI validity, TypeStats type).

**Misses:**
- `wilson_interval` with wins > total (invalid input) — the function doesn't guard against this.
- `wilson_interval` with negative values.
- `compute_type_stats` with non-uniform output types mixed in — only basic filtering tested.
- Percentile interpolation accuracy against numpy reference values.

### `tests/test_decision/test_recommendation.py` (16 tests)

**Covers:** RecommendationResult dataclass, NO_GO (high severity single/multiple, case-insensitive, medium doesn't trigger, plain string doesn't trigger), CONDITIONAL (below threshold, contains threshold info, custom threshold, insufficient samples, single sample), GO (all types pass, reasoning mentions types, low warnings don't block), per_type_details (populated, correct count), edge cases (no results, NO_GO priority over threshold), Wilson CI lower bound CONDITIONAL.

**Misses:**
- Multiple output types where one passes and one fails — only tested with single types.
- The "unknown" output type path (when no `output_type` annotations exist).
- `_has_high_severity` with non-dict, non-string inputs.
- `model_copy` call for the "unknown" injection path — never tested.

### `tests/test_decision/test_ab_stats.py` (7 tests)

**Covers:** `chi_squared_test` (significant, identical, empty, all wins, all losses, one-sided zero), `bootstrap_ci` (clearly different, similar, empty, deterministic seed), `compute_ab_significance` (full analysis, empty results).

**Misses:**
- `_chi2_survival_approx` — zero direct tests. Tested only through `chi_squared_test` when scipy is unavailable, but whether scipy is available in CI is environment-dependent, so the fallback may never run.
- `bootstrap_ci` with `confidence` != 0.95.
- `bootstrap_ci` with `n_bootstrap` edge values (1, very large).
- `bootstrap_ci` index boundary math for edge cases where `alpha/2 * n_bootstrap` is not an integer.
- `compute_ab_significance` with ties (`winner: "tie"` or `winner: None`).
- `compute_ab_significance` with missing keys in result dicts.

---

## 3. Risk Ranking

| # | Failure mode | Likelihood | Blast radius | Risk | Existing tests catch it? |
|---|---|---|---|---|---|
| 1 | **Fallback chain silent degradation**: BERTScore unavailable, embedding unavailable, falls back to exact_match for free text — composite scores drop dramatically, win rates collapse, recommendation flips to NO_GO. No user visibility into which evaluator actually ran. | HIGH | CRITICAL | **CRITICAL** | PARTIAL — fallback is tested but the intermediate case (BERTScore down, embedding up) is not. No test verifies `details["evaluators_used"]` reflects the actual fallback path. |
| 2 | **Wilson CI math error**: An incorrect Wilson interval shifts the CI lower bound, causing GO recommendations for unsafe migrations or blocking safe ones. | LOW | CRITICAL | **HIGH** | YES for basic cases. No property-based tests validating the formula against known reference implementations or mathematical invariants (e.g., interval always contains p_hat, interval shrinks with n). |
| 3 | **JSON fence stripping misparse**: Thinking-prefix models (Qwen, DeepSeek) emit `<think>...</think>` blocks before JSON. The `_strip_fences` function in json_validator.py handles this, but the code path is untested. A failure here means json_valid=0 and composite_score=0 for every JSON pair from these models. | HIGH | HIGH | **HIGH** | NO — thinking prefix path has zero test coverage. |
| 4 | **Batch BERTScore index misalignment**: `batch_compute_bertscore` returns scores in order, and `bertscore_map` maps them back to original indices via `free_text_indices`. An off-by-one or index corruption silently assigns wrong scores to wrong pairs. | MEDIUM | HIGH | **HIGH** | NO — the batch path in CompositeEvaluator.evaluate phase 2 has zero test coverage. |
| 5 | **Multi-run aggregation with skips**: If different runs skip different pairs (API errors), the intersection logic drops pairs. With high skip rates, aggregated results may be based on a tiny subset, making win rates unreliable. | MEDIUM | MEDIUM | **MEDIUM** | PARTIAL — one test covers mismatched skips, but doesn't test the case where intersection is empty. |
| 6 | **Thread-safety of progress callback and cost accumulation**: `on_progress` and `self._ctx.add_cost` are called from worker threads inside `ThreadPoolExecutor`. If `PipelineContext` or the callback is not thread-safe, race conditions corrupt data. | MEDIUM | MEDIUM | **MEDIUM** | NO — zero thread-safety tests. |
| 7 | **`_parse_score` false positive on multi-digit numbers**: Input `"15"` matches `\b([1-5])\b` → returns 1.0, not None. This could inflate LLM judge scores if the judge model returns verbose text with numbers outside 1-5 that happen to contain a 1-5 digit. | MEDIUM | LOW | **MEDIUM** | NO — no test for this edge case. |
| 8 | **`_chi2_survival_approx` inaccuracy**: The Wilson-Hilferty approximation diverges from scipy for small chi2 values (< 1.0). If scipy is not installed in production, the pure-Python fallback could produce wrong significance decisions. | LOW | MEDIUM | **MEDIUM** | NO — the approximation function has zero direct tests or accuracy benchmarks. |
| 9 | **`_extract_keypaths` infinite recursion on circular references**: Not possible with JSON (no circular refs in json.loads output), but if anyone passes a manually-constructed dict with cycles, this will stack overflow. | VERY LOW | LOW | **LOW** | NO — but risk is near zero for JSON-parsed inputs. |
| 10 | **Embedding LRU cache cross-test pollution**: `_get_sentence_transformer` uses `functools.lru_cache`. In test suites, the cached mock from one test can leak into another. | MEDIUM | LOW | **LOW** | NO — tests clear `sys.modules` but don't clear the LRU cache. The mock pattern happens to prevent this because the module is re-imported, but it's fragile. |

---

## 4. Test Plan by Tier

### Tier 1: Unit (pure logic, no I/O)

| Test | Assertions | Write time | Status |
|---|---|---|---|
| `_strip_fences` with thinking prefix + fenced JSON | Extracts JSON from `"<think>reasoning</think>\n```json\n{...}\n```"` | S | MISSING |
| `_strip_fences` with bare JSON after thinking text | Extracts `{"key": "val"}` from `"Some thinking\n{"key": "val"}"` | S | MISSING |
| `_strip_fences` with multiple fence blocks | Returns content from first fence block | S | MISSING |
| `_strip_fences` with non-JSON language fences | Returns raw content for `` ```python `` blocks | S | MISSING |
| `_extract_keypaths` direct tests | Root scalar → `{"__root__": val}`, nested arrays of dicts, empty containers | S | MISSING |
| `_array_sim` direct tests | Duplicate elements, empty vs empty, single element, mixed types | S | MISSING |
| `_array_sim` greedy matching with duplicates | `[1,1,2]` vs `[1,2,2]` — verify matched_values correct | S | MISSING |
| `_composite_score` with unknown metric name | Falls back to weight 1.0 | S | MISSING |
| `_composite_score` with only `json_valid` in scores | Returns 0.0 (no non-gating metrics) | S | MISSING |
| `_score_semantic` intermediate fallback | BERTScore ImportError + EmbeddingEvaluator available → returns `embedding_sim` | M | MISSING |
| `_score_semantic` with pre-computed bertscore_f1 | Returns `{"bertscore_f1": value}` without calling any evaluator | S | MISSING |
| `_parse_score` with "15" | Should return None or 1.0 — document intended behavior | S | MISSING |
| `_parse_score` with "Score: 3/5" | Returns 3.0 | S | MISSING |
| `_normalize` edge cases | Values outside 1-5 (0, 6, negative) — document intended behavior | S | MISSING |
| `string_similarity` with Unicode | NFC vs NFD forms of accented characters | S | MISSING |
| `detect_output_type` with `"null"` response | Returns JSON (verify this is intended) | S | MISSING |
| `detect_output_type` with `"categorise"` keyword | British spelling triggers override | S | MISSING |
| `batch_compute_bertscore` with empty list | Returns `[]` | S | MISSING |
| `_aggregate_runs` with identical scores | stdev=0, is_non_deterministic=False | S | MISSING |
| `_aggregate_runs` with output_type=None on pair | Forces detect_output_type for threshold lookup | S | MISSING |
| `wilson_interval` with wins > total | Document crash or return behavior | S | MISSING |
| `wilson_interval` with negative inputs | Document crash or return behavior | S | MISSING |

### Tier 2: Contract (both sides of an interface)

| Test | Assertions | Write time | Status |
|---|---|---|---|
| Every Evaluator subclass returns `dict[str, float]` with documented keys | Score keys match METRIC_WEIGHTS keys + any gating keys | M | PARTIAL — exists per evaluator but not systematized |
| `EVALUATOR_ROUTING` maps match actual evaluator output keys | For each OutputType, the routed evaluators produce keys that exist in METRIC_WEIGHTS | M | MISSING |
| `_composite_score` handles every metric key produced by every evaluator | No unknown metric gets silent weight=1.0 | M | MISSING |
| `EvalResult` schema contract | All fields populated, `failure_reason` taxonomy values match F6 spec | S | PARTIAL |
| `RecommendationResult.per_type_details` matches TypeStats schema | Verify dataclass fields are present and typed correctly | S | EXISTS |
| `compute_type_stats` output feeds correctly into `make_recommendation` | Type-level integration: stats → threshold comparison → recommendation | M | EXISTS (implicitly via recommendation tests) |

### Tier 3: Integration (real local service, deterministic)

| Test | Assertions | Write time | Status |
|---|---|---|---|
| `CompositeEvaluator.evaluate` full pipeline with mocked litellm | JSON pair → json_valid + json_field_match + json_structural_sim + json_schema_match all present | M | PARTIAL — json_structural not verified in composite tests |
| `CompositeEvaluator.evaluate` with batch BERTScore mock | Short text pairs → bertscore_map populated → bertscore_f1 in scores | L | MISSING |
| `CompositeEvaluator.evaluate` cost tracking via PipelineContext | Mock ctx, verify add_cost called with correct stage | M | MISSING |
| `CompositeEvaluator.evaluate` token tracking via PipelineContext | Mock ctx, verify add_tokens called with prompt/completion counts | M | MISSING |
| `evaluate_multi_run` → `evaluate` → scoring → aggregation full chain | 3 runs, verify final composite_score, run_scores, score_std, is_non_deterministic | L | PARTIAL — tested with patched evaluate, not full chain |
| `make_recommendation` with mixed output types | Some pass, some fail → CONDITIONAL with correct reasoning | M | MISSING |
| `make_recommendation` with "unknown" output type path | No output_type annotations → model_copy injection → correct stats | M | MISSING |
| `chi_squared_test` scipy vs pure-Python comparison | Both paths produce same significance decision for 10 test cases | M | MISSING |

### Tier 4: Property-based (Hypothesis)

| Test | Assertions | Write time | Status |
|---|---|---|---|
| `wilson_interval` invariants | For all (wins, total) where 0 <= wins <= total: (1) 0 <= lo <= hi <= 1, (2) lo <= p_hat <= hi, (3) interval shrinks as total increases | M | MISSING |
| `_composite_score` range | For all scores dicts with values in [0,1]: result in [0,1] (unless JSON gating) | M | MISSING |
| `_coerce_match` symmetry | `_coerce_match(a, b) == _coerce_match(b, a)` for all a, b | S | MISSING |
| `_lcs_length` bounds | For all (a, b): 0 <= LCS <= min(len(a), len(b)) | S | MISSING |
| `_lcs_length` symmetry | `_lcs_length(a, b) == _lcs_length(b, a)` | S | MISSING |
| `_dict_compare` scores in [0,1] | For all pairs of flat dicts: structural_sim and schema_match in [0,1] | M | MISSING |
| `detect_output_type` always returns valid OutputType | For all strings: result is a member of OutputType enum | S | MISSING |
| `string_similarity` range | For all (a, b): 0.0 <= result <= 1.0 | S | MISSING |
| `bootstrap_ci` CI contains observed diff at high confidence | For large n_bootstrap, ci_lower <= mean_diff <= ci_upper (may fail ~5% of time; use higher confidence) | M | MISSING |
| `_chi2_survival_approx` vs scipy accuracy | For chi2 in [0.01, 20], abs(approx - scipy) < 0.05 | M | MISSING |
| JSON evaluator scores in [0,1] | For all valid JSON pairs: all score values in [0,1] | M | MISSING |
| `_array_sim` scores in [0,1] | For all pairs of lists: both return values in [0,1] | S | MISSING |

### Tier 5: End-to-end (real APIs, real money)

| Test | Assertions | Write time | Status |
|---|---|---|---|
| LLMJudgeEvaluator against real model | Score 5 known pairs, verify scores are sensible (not NaN, in range) | L | MISSING |
| CompositeEvaluator full pipeline against real target model | 3 prompt pairs, verify EvalResults have all expected fields, non-zero scores | XL | MISSING |
| Full migration pipeline → recommendation | End-to-end: ingest → optimize → evaluate → recommendation | XL | MISSING — outside this boundary, coordination with optimization-engines |

---

## 5. Synthetic Data Generation Strategy

### What "realistic" means

1. **Labeled prompt/response pairs** with known quality grades:
   - Grade A (near-identical): same content, minor rephrasing
   - Grade B (semantically equivalent): same meaning, different wording
   - Grade C (partial match): correct topic, wrong details
   - Grade D (wrong): completely different response
   - Grade F (malformed): invalid JSON, empty strings, truncated output

2. **Known edge cases for JSON fencing:**
   - Bare JSON: `{"key": "value"}`
   - `` ```json\n{...}\n``` `` fence
   - `` ```\n{...}\n``` `` plain fence
   - Thinking prefix: `<think>Some reasoning</think>\n```json\n{...}\n```"`
   - Thinking prefix + bare JSON: `"Reasoning here.\n{"key": "val"}"`
   - Multiple fences: `"```json\n{a:1}\n```\nmore text\n```json\n{b:2}\n```"`
   - Non-JSON fence: `` ```python\nprint("hi")\n``` ``

3. **Mixed output types** per test set:
   - 30% JSON (flat dicts, nested dicts, arrays, arrays of objects)
   - 20% CLASSIFICATION (single word, multi-word labels, numeric labels)
   - 25% SHORT_TEXT (6-50 words)
   - 25% LONG_TEXT (51-200 words)

### How we generate it

**Static fixture file** (`tests/fixtures/eval_synthetic_pairs.json`):
- 50 manually-crafted pairs covering all grades and output types
- Each entry has: `prompt`, `expected_response`, `graded_responses` (dict of grade → response), `output_type`, `expected_scores` (approximate ranges for validation)
- Version-controlled; changes require review

**Parametrize-based generation** for property tests:
- Use Hypothesis `st.text()`, `st.from_regex()`, `st.recursive()` for JSON structure generation
- Use `st.sampled_from(OutputType)` for output type coverage

**VCR cassettes** for LLM judge tests:
- Record real LLM responses using `vcrpy` or `pytest-recording`
- Store in `tests/cassettes/llm_judge/`
- Re-record quarterly or on model version changes
- Each cassette contains: request messages, response content, cost, tokens

### Stability and rot prevention

- Static fixtures pinned to known scores; CI fails if evaluator changes break expected ranges (regression tests)
- Cassettes are replayed deterministically; no network calls in CI
- Hypothesis seeds stored in `.hypothesis/` database for reproducibility
- Quarterly review cycle: re-record cassettes, update expected score ranges if evaluator logic changes intentionally

### Cost profile

- Static fixtures: $0
- Property-based: $0
- VCR cassettes: ~$0.50 per recording session (10 LLM judge calls × 2 directions × ~$0.01/call)
- Real E2E: ~$2-5 per run (10 completions × $0.01-0.50 depending on model)

---

## 6. Fixtures, Fakes, and Mocks

### New fixtures needed

| Fixture | Location | Shared? |
|---|---|---|
| `eval_config` — MigrationConfig with test defaults | `tests/test_evaluate/conftest.py` | No |
| `eval_config_local_only` — config with `local_only=True` | `tests/test_evaluate/conftest.py` | No |
| `eval_config_multi_run` — config with `eval_runs=3` | `tests/test_evaluate/conftest.py` | No |
| `mock_litellm_completion` — reusable fixture that returns configurable responses | `tests/test_evaluate/conftest.py` | No |
| `graded_pairs` — dict mapping grade → list of (expected, actual) tuples | `tests/test_evaluate/conftest.py` | No |
| `json_fenced_pairs` — list of (raw_fenced_string, expected_parsed_json) | `tests/test_evaluate/conftest.py` | No |
| `mock_pipeline_context` — PipelineContext with assertions on add_cost/add_tokens | `tests/conftest.py` | **YES — coordination point with safety-observability-decision** |

### New fakes needed

| Fake | Purpose |
|---|---|
| `FakeBERTScoreModule` | Deterministic fake returning pre-computed scores for known inputs. Replaces the fragile `sys.modules` patching. Should be a pytest plugin or fixture. |
| `FakeEmbeddingModel` | Returns fixed embedding vectors for known strings. Avoids numpy dependency in unit tests. |
| `FakeLiteLLMCompletion` | Configurable: takes a dict of `{prompt_hash: response}` and returns matching responses. Supports error injection by prompt pattern. |

### Coordination points with other testing leads

1. **`mock_pipeline_context`** — shared with safety-observability-decision lead. The `PipelineContext.add_cost` and `add_tokens` interfaces are used by both evaluation and optimization stages. Any changes to this interface affect both test suites.

2. **`EvalResult` schema** — shared with optimization-engines lead. The optimizer consumes EvalResults to compute improvement metrics. If `run_scores`, `score_std`, or `is_non_deterministic` fields change, both boundaries need test updates.

3. **`make_recommendation` input** — shared with safety-observability-decision lead. The recommendation engine takes `validation_results: list[EvalResult]` and `safety_warnings: list[Any]`. Both leads need to agree on the EvalResult fixture format.

4. **`sample_pairs` in `tests/conftest.py`** — already shared globally. Any changes affect all test suites.

---

## 7. Gaps You Can't Close

1. **LLM non-determinism in E2E tests.** Even with temperature=0, LLM outputs vary across API versions, load conditions, and model updates. Cassettes help but don't eliminate the problem for live E2E tests. Acceptable tolerance bands must be set manually and updated over time. **NEEDS_HUMAN_REVIEW: what tolerance is acceptable for E2E score stability?**

2. **BERTScore model accuracy validation.** We can test that `compute_bertscore` returns a float, but we can't unit-test whether distilbert-base-uncased produces *correct* similarity scores. That requires human judgment on reference pairs. The best we can do is regression tests against known pairs with known expected ranges.

3. **Real model output distribution.** Synthetic test data can approximate but not replicate the distribution of outputs from production models. Edge cases we don't think of won't be in the test set. Production monitoring (outside this boundary) is the only mitigation.

4. **scipy vs pure-Python chi-squared accuracy.** The Wilson-Hilferty approximation is documented as "accurate to ~0.01." We can benchmark against scipy, but in CI environments without scipy, we're trusting the approximation. We can't close this without making scipy a hard dependency. **NEEDS_HUMAN_REVIEW: should scipy be a required dependency for production use?**

5. **Thread-safety of third-party libraries.** We can test our code's thread-safety, but if `litellm.completion`, `bert_score.score`, or `sentence_transformers` have thread-safety bugs, we can't catch them in unit tests. Integration tests with concurrent calls are the only option.

---

## 8. Cost and Time Estimate

| Tier | Test count (new) | Estimated write time | Dependency |
|---|---|---|---|
| Unit — pure logic | 22 tests | 3-4 hours (mostly S/M) | None |
| Contract | 5 tests | 2 hours | None |
| Integration (mocked) | 8 tests | 4-5 hours (M/L) | mock_pipeline_context fixture |
| Property-based (Hypothesis) | 12 tests | 4-5 hours | hypothesis package |
| VCR cassette recording | 5 cassettes | 2 hours + ~$0.50 | vcrpy or pytest-recording |
| E2E (real APIs) | 2-3 tests | 3 hours + ~$5/run | API keys, budget approval |
| Fixture/fake creation | 7 fixtures + 3 fakes | 3-4 hours | Coordination with other leads |
| **Total** | **~55 new tests** | **~22-25 hours** | **~$6 per E2E run** |

Priority order for maximum risk reduction per hour:
1. `_strip_fences` thinking-prefix tests (2h, closes Risk #3)
2. Batch BERTScore integration test (2h, closes Risk #4)
3. Fallback chain intermediate case test (1h, closes Risk #1)
4. Wilson CI property-based tests (2h, hardens Risk #2)
5. All remaining unit tests (4h)
6. Contract tests (2h)
7. Integration tests (5h)
8. Property-based remainder (3h)
9. VCR + E2E (5h)

---

## 9. Path to Production

### Current readiness level

**3/5 — Core logic tested, but significant gaps in integration paths and edge cases.**

The individual evaluators (exact_match, json_validator, json_structural, llm_judge) have solid unit test coverage. The composite orchestration layer has partial coverage with known blind spots (batch BERTScore, fallback chain, thread safety). The decision layer (statistics, recommendation) has good coverage with property-based testing as the main missing piece.

### Gap to production-hardened

1. **Fallback chain visibility** — No test verifies that the fallback from BERTScore → embedding → exact_match is visible in EvalResult.details. In production, operators need to know which evaluator actually ran. The `details["evaluators_used"]` field exists but is populated from `scores.keys()`, which doesn't distinguish "intended evaluator" from "fallback evaluator."

2. **Fence stripping for non-OpenAI models** — The thinking-prefix path in `_strip_fences` is untested. Production will see Qwen, DeepSeek, and other CoT models. This is a **release blocker**.

3. **Wilson CI correctness** — Used to make GO/NO_GO decisions. A math error here has direct business impact. Property-based tests are needed before production sign-off.

4. **Cost tracking** — Never tested. Production needs accurate cost reporting for billing and budget monitoring.

### Gates

| Gate | Criteria | Status |
|---|---|---|
| G1: All evaluators unit-tested | Each evaluator has >=5 tests covering happy path, edge cases, error handling | PASS (existing) |
| G2: Fallback chain tested | All 3 fallback tiers verified with correct output keys | FAIL |
| G3: JSON fence stripping complete | Thinking-prefix, bare-JSON-after-text, multiple fences all tested | FAIL |
| G4: Wilson CI property-tested | Invariants validated for 1000+ random inputs | FAIL |
| G5: Composite integration tested | Full evaluate() pipeline with batch BERTScore mock, cost tracking, token tracking | FAIL |
| G6: Multi-run aggregation hardened | Edge cases (empty intersection, all identical scores, single run) | PARTIAL |
| G7: E2E smoke test passes | 3 pairs through real API, scores in expected ranges | FAIL |

### Ordered sequence

1. **Week 1 (highest risk reduction):**
   - Write `_strip_fences` thinking-prefix tests (G3)
   - Write fallback chain tests (G2)
   - Write batch BERTScore integration test (G5 partial)
   - Create shared fixtures (`conftest.py`)

2. **Week 2 (statistical correctness):**
   - Add Hypothesis property tests for wilson_interval (G4)
   - Add Hypothesis tests for _composite_score, _coerce_match, _lcs_length
   - Write contract tests for METRIC_WEIGHTS ↔ evaluator key alignment
   - Write cost/token tracking integration tests (G5 complete)

3. **Week 3 (hardening):**
   - Write chi-squared approximation accuracy benchmarks
   - Write multi-run edge case tests (G6)
   - Record VCR cassettes for LLM judge
   - Write recommendation tests with mixed output types

4. **Week 4 (E2E and sign-off):**
   - Run E2E smoke tests (G7)
   - Fix any issues found
   - Final review and gate check

### Smallest next slice

Write 4 tests for `_strip_fences` covering the thinking-prefix path. This closes Risk #3 (rated HIGH), takes ~1 hour, has zero dependencies on other boundaries, and can be done immediately.

### Dependencies on other boundaries

| Dependency | What we need | From whom |
|---|---|---|
| `PipelineContext` mock fixture | Agreed-upon interface for `add_cost` and `add_tokens` | safety-observability-decision lead |
| `EvalResult` field stability | Guarantee that `run_scores`, `score_std`, `is_non_deterministic` fields won't change names | optimization-engines lead |
| `MigrationConfig` field stability | Guarantee that `eval_runs`, `eval_aggregation`, `variance_flag_threshold` won't change types | project-wide |
| VCR infrastructure setup | `pytest-recording` or `vcrpy` added to dev dependencies | project-wide |
