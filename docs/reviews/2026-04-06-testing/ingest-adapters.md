# Ingest Adapters & Data Pipeline — Testing Review

**Boundary owner:** Ingest subsystem
**Date:** 2026-04-06
**Reviewer:** Testing lead (plan-only)

---

## 1. Boundary Map

### What's inside (we test it)

```
src/rosettastone/ingest/
  base.py           DataAdapter ABC (load() -> list[PromptPair])
  schema.py         PromptPairInput Pydantic model + normalize_response validator
  splitter.py       deduplicate() via SHA-256, split_data() with train/val/test ratios
  jsonl.py          JSONLAdapter — read .jsonl files, validate per line via schema.py
  csv_adapter.py    CSVAdapter + CSVColumnMapping — read .csv/.tsv, column mapping, delimiter detection
  redis_adapter.py  RedisAdapter — SCAN-based iteration, auto-format-detection, lazy import
  redis_formats.py  4 parsers: parse_litellm_entry, parse_langchain_entry, parse_redisvl_entry, parse_gptcache_entry
  langsmith_adapter.py  LangSmithAdapter — REST client wrapper, run parsing, lazy import
  braintrust_adapter.py BraintrustAdapter — REST client wrapper, log entry parsing, lazy import
  otel_adapter.py   OTelAdapter — OTLP JSON file/directory reader, gen_ai span extraction
```

### What's outside (we mock/stub)

```
  redis (PyPI)            External package — mock client.scan(), client.get()
  langsmith (PyPI)        External package — mock langsmith.Client, client.list_runs()
  braintrust (PyPI)       External package — mock braintrust.Braintrust, projects.retrieve()
  litellm (PyPI)          Used by SyntheticDataGenerator — outside ingest boundary
  Filesystem              Real for JSONL/CSV/OTel tests (tmp_path), no mocking needed
  PromptPair (core/types) Consumed output type — treated as stable contract, not mocked
  MigrationConfig         Upstream config — tested in pipeline.py's _build_adapter(), outside our scope
```

### On the fence (integration candidates)

```
  Redis (Docker)          RedisAdapter against a real Redis instance — currently all-mocked
  src/rosettastone/testing/redis_populator.py — writes LiteLLM-format entries to real Redis
  _build_adapter() in pipeline.py — adapter factory, wires config to ingest; tested elsewhere
```

### Dependency tree (consumer side)

```
DataAdapter.load()
  --> list[PromptPair]
        --> splitter.deduplicate() + split_data()
              --> (train, val, test) lists
                    --> pipeline.load_and_split_data()
                          --> optimize, evaluate, report subsystems
```

---

## 2. Current Coverage Audit

### `tests/test_ingest/test_schema.py` (11 tests)

**Covers:** PromptPairInput validation for all 3 required fields, list-prompt format, dict-response normalization (both `content` key and fallback `str(dict)`), optional field defaults, optional field acceptance, multi-turn messages, None prompt/response rejection, empty dict rejection.

**Misses:**
- Empty string prompt/response — Pydantic will accept `""` as valid; is that intended? The CSV adapter skips empty values, but the schema doesn't enforce non-empty. **NEEDS_HUMAN_REVIEW: Should PromptPairInput reject empty-string prompt or response?**
- `int`, `float`, or `bool` as prompt — schema type is `str | list[dict[str, Any]]`, so Pydantic will coerce `123` to `"123"` via str coercion. No test exercises this.
- Deeply nested dict response without `content` key — only one test for this, and it only checks `"positive" in result.response` which is fragile (relies on Python dict repr ordering, which is stable in CPython 3.7+ but the test assertion is weak).

**Brittle:** `test_dict_response_without_content_key_normalized_to_str_repr` — asserts `"positive" in result.response` which would pass even if the normalization logic changed to JSON-serialize instead of `str()`. Not false-confidence exactly, but the assertion is too loose to detect the difference between `str(dict)` and `json.dumps(dict)`.

### `tests/test_ingest/test_jsonl.py` (14 tests)

**Covers:** Single/multiple valid pairs, blank line skipping, list-format messages, dict response normalization via schema, metadata+feedback, empty/blank-only files, string path acceptance, malformed JSON (with line number), missing required fields (prompt, response, source_model), partial JSON, error-stops-on-first-bad-line behavior.

**Misses:**
- Unicode/encoding edge cases (BOM, non-UTF-8, emoji, CJK characters in prompts).
- Very large files (performance/memory — currently loads all into memory).
- File not found (no `FileNotFoundError` test).
- Permission denied.
- Lines with only whitespace + valid JSON (e.g., `  {"prompt":...}  `).

**No dead or false-confidence tests.**

### `tests/test_ingest/test_csv_adapter.py` (19 tests)

**Covers:** Standard CSV, custom column mapping (prompt/response/source_model/metadata/feedback), TSV auto-detection, .tab extension, explicit delimiter override, BOM-encoded files, quoted fields with commas, blank row skipping, whitespace stripping, missing prompt/response column errors, headers-only empty CSV, source_model fallback to constructor, CSVColumnMapping defaults, column validation.

**Misses:**
- Multiline quoted fields (CSV RFC 4180 allows `\n` inside quoted cells).
- Very large CSVs (memory).
- Null bytes or binary content in CSV.
- File not found / permission errors.
- Windows-style line endings (`\r\n`) — csv module handles this, but no test confirms it.
- Row where only prompt is empty (skipped) vs row where only response is empty (skipped) — logic exists but test `test_blank_rows_skipped` only tests both-empty.
- Empty feedback column value (code returns `None` — no test for this specific path).

**Subtle issue:** `test_row_number_in_error_messages` doesn't actually trigger a row-level error; it triggers a column-not-found error at the header level. The assertion `"nonexistent_col" in str(exc_info.value) or "row" in str(exc_info.value).lower()` passes because the first disjunct is true, but this isn't testing what the name says. Not quite false-confidence, but misleading.

### `tests/test_ingest/test_redis_adapter.py` (9 tests)

**Covers:** LiteLLM entries parsed to PromptPairs, source_model propagation, SCAN usage (not KEYS), multi-cursor pagination, unparseable entries skipped, warning log for skipped entries, zero-match ValueError, empty Redis ValueError, import error messaging, expired-key (None GET) handling.

**Misses:**
- Auto-format-detection across different formats (e.g., mix of LiteLLM + LangChain entries — which wins?).
- Format detection with `_SAMPLE_SIZE` boundary (exactly 100 entries, >100 entries).
- Non-LiteLLM format parsing through the full `load()` flow (all tests use `_litellm_value`).
- Connection errors / timeouts from Redis client.
- Large key volume (>10k keys).
- `decode_responses=True` vs `False` — the adapter uses raw bytes but doesn't assert encoding.

**Brittle:** `test_scan_multiple_cursor_iterations` — uses `side_effect` list that will raise `StopIteration` if scan is called a third time. The source code has a `try/except StopIteration` guard (line 133-135 of redis_adapter.py) specifically for this, which masks test infrastructure issues.

### `tests/test_ingest/test_redis_formats.py` (18 tests)

**Covers:** LiteLLM: valid entry, last-user-message selection, system-message exclusion, string response, invalid JSON, missing messages, no user message, missing response, empty choices, empty bytes, non-dict value. LangChain: direct input/output, LLMResult with message content, invalid JSON, no-prompt-found. RedisVL: prompt/response, input_text/response, output key variant, invalid JSON, missing prompt, missing response. GPTCache: query/answer, question/answer, prompt/response, invalid JSON, unrecognized format, LiteLLM fallback.

**Misses:**
- LiteLLM: user message with empty string `content` (should return None but untested).
- LiteLLM: response.choices[0].message.content is `None` (OpenAI returns this for tool calls).
- LangChain: LLMResult with `prompt` or `input` top-level fallback keys (code exists at lines 128-134 but no test exercises this path specifically in isolation).
- RedisVL: empty-string values for prompt/response (code checks `if candidate:` which rejects `""` — untested).
- GPTCache: empty-string answer field.
- All parsers: non-UTF-8 bytes (e.g., Latin-1 encoded values).
- All parsers: very large values (multi-MB JSON).

**No dead or false-confidence tests.**

### `tests/test_ingest/test_langsmith_adapter.py` (20 tests)

**Covers:** Valid runs to PromptPairs, prompt extraction from messages/input/prompt keys, response from output/generations keys, model from run metadata, model fallback, metadata preservation, date range forwarding, date filter omission, project name, execution_order=1, skip runs with no outputs, empty project, missing inputs, empty inputs, mixed valid/invalid, warning log, pagination (50 runs), env var API key, source_model on all pairs, import error (3 tests), prompt format tests (list, string, multi-role, dict response coercion).

**Misses:**
- Response from `outputs['response']` key (code path exists at line 152 but untested).
- `run.extra` being `None` (code handles via `run.extra or {}` but the `_make_run` helper always provides a dict).
- Network errors / client exceptions during `list_runs`.
- Rate limiting from LangSmith API.
- Very large run payload (inputs/outputs with deeply nested structures).

**No dead or false-confidence tests. Well-structured test suite.**

### `tests/test_ingest/test_braintrust_adapter.py` (18 tests)

**Covers:** Valid entries, input->prompt, output->response, model from metadata, metadata preservation (entry_id, tags, timestamp, scores), model fallback, project name used, source_model on all pairs, empty project, missing input, missing output, mixed entries, warning log, dict output to JSON string, list-messages input, string input, import error (3 tests), expected->feedback, chat format.

**Misses:**
- `expected` field as a dict (code JSON-serializes it — line 148 — but no test for dict expected).
- Entry with `metadata: None` (code does `entry.get("metadata") or {}` — the `or {}` handles None, but untested).
- Entry with no `id` field (code checks `entry.get("id")` — untested path where entry_id is absent).
- API key from env var (`BRAINTRUST_API_KEY`).
- Network errors.

**No dead or false-confidence tests.**

### `tests/test_ingest/test_otel_adapter.py` (22 tests)

**Covers:** Single file, multiple spans, directory of files, prompt/response/model from attributes, fallback model, span metadata, full OTLP structure traversal, events fallback, non-gen_ai span filtering, missing completion/prompt skipping, attribute precedence over events, JSON-string messages parsing, string prompt preservation, multiple resource spans, empty JSON, malformed JSON skip, no gen_ai spans, mixed valid/invalid, structural logging, large file (200 spans), single file path, directory with multiple files, non-json ignored, empty directory, nonexistent path, no recursion.

**Misses:**
- `_extract_from_events` returning the *first* matching event attribute value regardless of key (line 179: `for value in event_attrs.values(): return value`) — this is a potential bug if the event has multiple attributes, but no test verifies which attribute value wins.
- Span with no `attributes` key at all (vs empty list).
- Non-`stringValue` attribute values (e.g., `intValue`, `boolValue`) — silently ignored by `_attrs_to_dict`.
- Deeply nested `resourceSpans` with multiple `scopeSpans` per resource.
- File encoding issues (non-UTF-8 JSON files).

**No dead or false-confidence tests. This is the most thorough test file.**

### `tests/test_ingest/test_splitter.py` (17 tests)

**Covers:** Dedup exact duplicates, preserve uniques, mixed dups+uniques, same-prompt-diff-response, same-response-diff-prompt, empty list, single pair, insertion order, list-prompt fingerprinting. Split: total count, non-empty train/val/test, no overlap, 3/2/1 pair edge cases, train ratio with large dataset, val ratio, dedup-before-split, returns 3-tuple, all elements are PromptPair, reproducibility with seed, no-seed doesn't crash.

**Misses:**
- `split_data` with `train_ratio=0.0` or `train_ratio=1.0` — the `max(1, ...)` prevents zero-size train, but behavior at extremes is untested.
- `val_ratio=0.0` or `val_ratio=1.0` edge cases.
- Negative ratios (should probably raise, but code doesn't validate).
- Very large datasets (1M+ pairs — memory).
- Fingerprint collision resistance (SHA-256 collisions are astronomically unlikely, but the test doesn't verify the fingerprint logic itself).
- `_fingerprint` with list-prompt containing dicts in different key orders — `sort_keys=True` in `json.dumps` handles this, but no test exercises key-order-independent dedup.
- `split_data` mutates the input list via `shuffle` — no test verifies whether the original list is modified or not. The code shuffles a copy after dedup (the `deduplicate` function creates a new list), so the original is safe, but this contract is untested.

**Subtle issue:** `test_split_data_train_ratio_respected_with_large_dataset` calls `random.seed(42)` at the module level but then passes no `seed` to `split_data`. The `split_data` function calls `random.shuffle(pairs)` using the global RNG, so this test is effectively deterministic. But it's fragile because any test running before it that touches `random` could change the outcome. This isn't technically wrong (the assertion allows a wide range), but the test should use `seed=42` in `split_data` instead of `random.seed(42)`.

### `tests/conftest.py` (2 fixtures)

`sample_pairs` and `sample_jsonl_file` — shared fixtures. `sample_jsonl_file` uses `tempfile.NamedTemporaryFile(delete=False)` and never cleans up. Not a bug but a minor resource leak in test runs.

---

## 3. Risk Ranking

| # | Failure Mode | Manifestation | Likelihood | Blast Radius | Current Tests Catch It? | Rating |
|---|---|---|---|---|---|---|
| 1 | **Redis format mis-detection with mixed-format data** | Winning parser chosen based on first 100 keys; remaining keys parsed with wrong parser; silent data loss (entries return None, get skipped) | MEDIUM | HIGH — entire migration based on wrong subset | NO — all Redis tests use single-format data | **CRITICAL** |
| 2 | **Splitter input mutation** | `split_data` calls `random.shuffle(pairs)` on the deduped list; if caller retains a reference and expects original order, downstream code sees shuffled data | LOW | HIGH — corrupts pipeline state silently | NO — no test verifies original list immutability | **HIGH** |
| 3 | **OTel event extraction returns wrong attribute value** | `_extract_from_events` returns the first `values()` entry from event attributes dict; if an event has multiple attributes (e.g., `gen_ai.prompt` + a metadata attr), the wrong value could be selected | MEDIUM | MEDIUM — corrupted prompts in pipeline | NO — all test events have exactly one attribute | **HIGH** |
| 4 | **CSV/JSONL with empty-string prompt or response accepted** | Schema accepts `""` as valid; downstream eval runs on empty data; scores are meaningless | MEDIUM | MEDIUM — garbage-in/garbage-out migration results | NO — no test for empty-string fields | **HIGH** |
| 5 | **LangSmith adapter swallows all exceptions in _parse_run** | Broad `except Exception` at line 170 of langsmith_adapter.py; any bug in parsing logic (e.g., wrong attribute name) silently returns None and skips the run | MEDIUM | MEDIUM — silent data loss, reduced dataset | PARTIAL — tests check that bad runs are skipped, but can't distinguish "legitimately unparseable" from "bug in parser" | **HIGH** |
| 6 | **Large file memory exhaustion** | All adapters load entire dataset into memory (`list[PromptPair]`); no streaming/chunking | LOW (most datasets <100k pairs) | HIGH — OOM crash | NO | **MEDIUM** |
| 7 | **JSONL/CSV file encoding mismatch** | Non-UTF-8 files (Latin-1, Windows-1252) cause UnicodeDecodeError on read | MEDIUM | LOW — loud crash, user can fix encoding | NO — no encoding-mismatch tests | **MEDIUM** |
| 8 | **Dedup fingerprint ignores source_model** | Two pairs with same prompt+response but different source_model are deduped to one | LOW (unusual in practice) | MEDIUM — data loss if legitimately different pairs exist across models | NO — no test for cross-model dedup behavior | **MEDIUM** |
| 9 | **Redis connection failure not handled** | `_make_client()` creates client; if Redis is unreachable, `client.scan()` throws at runtime; no retry, no clear error message | MEDIUM | LOW — loud crash | NO — all tests mock the client | **MEDIUM** |
| 10 | **Braintrust/LangSmith API pagination not tested** | Code relies on client returning all results; if API paginates and client doesn't auto-paginate, data is silently truncated | LOW | HIGH — incomplete dataset | PARTIAL — LangSmith test has 50-run pagination test but through mock iterator, not real pagination | **LOW** |

---

## 4. Test Plan by Tier

### Unit Tests

| Test | Asserts | Status | Write-time |
|---|---|---|---|
| `test_fingerprint_ignores_source_model` | Two pairs with same prompt+response but different source_model produce same fingerprint | MISSING | S |
| `test_fingerprint_list_prompt_key_order_independent` | `[{"b": 2, "a": 1}]` and `[{"a": 1, "b": 2}]` produce identical fingerprints | MISSING | S |
| `test_fingerprint_whitespace_sensitivity` | `"hello "` and `"hello"` produce different fingerprints (they should — no stripping) | MISSING | S |
| `test_schema_empty_string_prompt_accepted_or_rejected` | `PromptPairInput(prompt="", response="x", source_model="m")` — clarify behavior | MISSING | S |
| `test_schema_numeric_prompt_coerced_to_string` | `prompt=123` becomes `"123"` via Pydantic coercion | MISSING | S |
| `test_schema_dict_response_without_content_key_uses_str` | Verify exact normalization output (not just substring) | MISSING (strengthens existing) | S |
| `test_csv_row_with_only_empty_prompt_skipped` | Row with `prompt=""`, `response="x"` is skipped | MISSING | S |
| `test_csv_row_with_only_empty_response_skipped` | Row with `prompt="x"`, `response=""` is skipped | MISSING | S |
| `test_csv_empty_feedback_becomes_none` | Empty feedback column value maps to `feedback=None` | MISSING | S |
| `test_split_data_extreme_ratios` | `train_ratio=0.0` and `train_ratio=1.0` — verify behavior | MISSING | S |
| `test_split_data_negative_ratio_raises` | Negative train_ratio should raise ValueError (or document that it doesn't) | MISSING | S |
| `test_split_data_does_not_mutate_input` | Original list is unchanged after `split_data` | MISSING | S |
| `test_otel_attrs_to_dict_ignores_non_string_values` | Attributes with `intValue` or `boolValue` are not included in result dict | MISSING | S |
| `test_otel_extract_from_events_multiple_attributes` | Event with 2+ attributes: verify which value is returned | MISSING | S |
| `test_litellm_parser_empty_content_user_message` | `{"role": "user", "content": ""}` returns None | MISSING | S |
| `test_litellm_parser_null_content_in_response` | `choices[0].message.content` is None (tool call scenario) | MISSING | S |
| `test_langchain_parser_top_level_prompt_fallback` | LLMResult with no `message.content` but `"prompt"` top-level key | MISSING | S |
| `test_braintrust_dict_expected_to_feedback` | `expected={"key": "val"}` becomes JSON string feedback | MISSING | S |
| `test_braintrust_metadata_none_handled` | Entry with `metadata: None` doesn't crash | MISSING | S |
| `test_braintrust_no_entry_id` | Entry without `id` field — metadata should not have `entry_id` key | MISSING | S |

### Contract Tests

| Test | Asserts | Status | Write-time |
|---|---|---|---|
| `test_all_adapters_return_list_of_prompt_pairs` | Every concrete adapter's `load()` returns `list[PromptPair]` — type check, not empty-list | EXISTS (spread across adapter tests) | - |
| `test_prompt_pair_has_required_fields` | Every PromptPair from every adapter has non-None `prompt`, `response`, `source_model` | PARTIAL — checked in some tests, not as a contract | M |
| `test_adapter_load_returns_empty_for_empty_source` | Every adapter returns `[]` (or raises ValueError for Redis) when source has no data | PARTIAL — tested per-adapter, not systematized | S |
| `test_splitter_output_types_match_input` | `split_data` output elements are the exact same PromptPair objects from input (identity, not just equality) | MISSING | S |
| `test_pipeline_build_adapter_produces_correct_type` | `_build_adapter(config)` returns a `DataAdapter` subclass for each `AdapterChoice` value — belongs to pipeline tests, flag as coordination point | MISSING (outside scope) | M |

### Integration Tests

| Test | Asserts | Status | Write-time |
|---|---|---|---|
| `test_redis_adapter_with_real_redis` | Start Redis in Docker, populate with `RedisPopulator`, load via `RedisAdapter`, verify round-trip | MISSING | L |
| `test_redis_format_detection_with_mixed_formats` | Populate Redis with entries in 2+ formats, verify winning format and parse results | MISSING | L |
| `test_redis_large_scan_pagination` | 10k+ keys, verify SCAN pagination exhausts all keys | MISSING | L |
| `test_otel_directory_with_real_export` | Use a checked-in golden OTLP export directory (or generate one) | MISSING | M |
| `test_csv_windows_line_endings` | Create a CSV with `\r\n` endings, verify correct parsing | MISSING | S |
| `test_jsonl_non_utf8_file_raises` | Latin-1 encoded JSONL file raises clear error | MISSING | S |

### Property-Based Tests (Hypothesis)

| Test | Strategy | Asserts | Status | Write-time |
|---|---|---|---|---|
| `test_dedup_idempotent` | `st.lists(prompt_pair_strategy)` | `deduplicate(deduplicate(pairs)) == deduplicate(pairs)` | MISSING | M |
| `test_dedup_subset_of_input` | `st.lists(prompt_pair_strategy)` | Every element in `deduplicate(pairs)` is `in` original | MISSING | M |
| `test_split_preserves_total_count` | `st.lists(prompt_pair_strategy, min_size=1)` + `st.floats(0.01, 0.99)` for ratios | `len(train) + len(val) + len(test) == len(deduplicate(input))` | MISSING | M |
| `test_split_no_overlap` | Same strategy | Set intersection of all 3 splits is empty | MISSING | M |
| `test_fingerprint_deterministic` | `st.text()` for prompt and response | `_fingerprint(pair)` called twice yields same hash | MISSING | S |
| `test_schema_roundtrip` | `st.text(min_size=1)` for prompt, response, source_model | `PromptPairInput.model_validate(data)` succeeds for any non-empty strings | MISSING | M |
| `test_csv_roundtrip_with_arbitrary_content` | `st.text(alphabet=st.characters(blacklist_categories=('Cs',)))` | Write CSV, load via CSVAdapter, verify prompt/response match | MISSING | L |
| `test_jsonl_roundtrip_with_arbitrary_content` | Same text strategy | Write JSONL, load via JSONLAdapter, verify content matches | MISSING | L |

### End-to-End Tests

| Test | What it exercises | Cost | Status | Write-time |
|---|---|---|---|---|
| `test_e2e_redis_ingest_to_split` | Real Redis + RedisPopulator + RedisAdapter + splitter | $0 (local Docker) | MISSING | XL |
| `test_e2e_langsmith_ingest` | Real LangSmith API, fetch runs, parse | $0 (free tier) but requires API key | MISSING | XL |
| `test_e2e_braintrust_ingest` | Real Braintrust API, fetch logs, parse | $0 (free tier) but requires API key | MISSING | XL |

These E2E tests should be opt-in (marked `@pytest.mark.e2e` or similar) and only run in CI with credentials present.

---

## 5. Synthetic Data Generation Strategy

### What "realistic" means for this boundary

Realistic test data for ingest means:

1. **JSONL**: Lines with multi-turn conversations (system + user + assistant), JSON-structured responses, Unicode content (CJK, emoji, RTL text), edge-case whitespace, metadata with nested dicts, prompts > 4096 tokens.
2. **CSV/TSV**: BOM-encoded files, quoted fields with embedded delimiters and newlines, columns in non-default order, extra columns beyond the mapping, rows with mixed presence of optional fields.
3. **Redis**: Entries in all 4 supported cache formats (LiteLLM, LangChain, RedisVL, GPTCache), mixed formats in the same keyspace, entries with expired TTLs, entries with binary/non-JSON values, entries with large payloads.
4. **OTel**: OTLP exports with multiple resourceSpans, mixed gen_ai and non-gen_ai spans, events-based vs attributes-based prompt/completion, JSON-encoded message lists in attributes, spans missing partial data.
5. **LangSmith/Braintrust**: Run objects with various input/output structures (messages, prompt, input keys), missing fields, metadata with model info, large batches.

### Generation approach

**Handwritten golden fixtures** (checked into `tests/fixtures/ingest/`):
- `valid_simple.jsonl` — 5 basic prompt/response pairs
- `valid_multi_turn.jsonl` — 3 pairs with multi-turn message prompts
- `valid_unicode.jsonl` — pairs with CJK, emoji, RTL text
- `edge_cases.jsonl` — empty metadata, null optional fields, very long prompts
- `standard.csv`, `standard.tsv` — basic CSV/TSV with all column types
- `bom_encoded.csv` — UTF-8 BOM file
- `quoted_multiline.csv` — RFC 4180 edge cases
- `otlp_mixed.json` — OTLP export with gen_ai + non-gen_ai spans
- `otlp_events_only.json` — OTLP export using only events (no inline attributes)

**Seeded generators** (in `tests/fixtures/generators/`):
- `generate_redis_fixtures.py` — produces JSON files with entries in all 4 Redis cache formats, seeded for determinism. Output: `tests/fixtures/ingest/redis_litellm.json`, `redis_langchain.json`, `redis_redisvl.json`, `redis_gptcache.json`, `redis_mixed.json`.
- These are run once (or on demand) and the output is checked in. They don't call any APIs.

**Property-based generators** (Hypothesis strategies, defined in `tests/conftest_ingest.py` or a shared `strategies.py`):
- `prompt_pair_strategy()` — generates `PromptPair` instances with text prompts or list-of-messages prompts
- `jsonl_line_strategy()` — generates valid JSONL line dicts
- `redis_entry_strategy(format)` — generates `(key, value)` tuples for each Redis cache format

### Stability

- All golden fixtures are checked into version control with version comments (e.g., `# v1 — 2026-04-06`).
- Hypothesis tests use `@settings(database=None)` in CI to avoid cross-run database sharing, and `@seed(...)` for reproducible failure cases.
- Seeded generators use `random.seed(42)` and output is diffed in CI. If generator code changes, output must be regenerated and diff reviewed.

### Preventing rot

- Golden fixtures are imported by at least one test each; if the schema changes and the fixture becomes invalid, the test fails.
- Generator scripts have a `--check` flag that verifies existing output matches what would be generated. CI runs this on every PR touching `src/rosettastone/ingest/`.
- Property-based strategies are defined in terms of the `PromptPair` model, so Pydantic schema changes break strategy definitions immediately.

### Cost profile

- Zero API cost. All fixtures are handwritten or seeded-RNG-generated.
- SyntheticDataGenerator (in `src/rosettastone/testing/synth_data.py`) calls real LLMs and is outside ingest scope. It's relevant only for E2E tests in the pipeline boundary.

### VCR-style cassette strategy

Not needed for ingest. The adapters that call external APIs (LangSmith, Braintrust) are tested via mock objects, and the mock objects are simple enough (they return lists of dicts) that record-and-replay adds complexity without proportional value. If integration tests are added later against real APIs, use `vcrpy` or `responses` library to record cassettes, stored in `tests/fixtures/cassettes/ingest/`, with a CI job that refreshes them quarterly.

---

## 6. Fixtures, Fakes, and Mocks

### New fixtures required

| Name | Scope | Location | Type | Notes |
|---|---|---|---|---|
| `prompt_pair_factory` | function | `tests/conftest.py` or `tests/test_ingest/conftest.py` | Factory function | `_make_pair()` exists in test_splitter.py; promote to shared fixture. Returns `PromptPair` with configurable fields. |
| `valid_jsonl_file` | function | `tests/test_ingest/conftest.py` | `tmp_path`-based fixture | Writes a standard 5-line JSONL file. Multiple tests recreate this inline. |
| `valid_csv_file` | function | `tests/test_ingest/conftest.py` | `tmp_path`-based fixture | Same for CSV. |
| `otlp_file` | function | `tests/test_ingest/conftest.py` | `tmp_path`-based fixture | Writes a standard OTLP JSON with 3 gen_ai spans. |
| `redis_entries_litellm` | function | `tests/test_ingest/conftest.py` | `dict[bytes, bytes]` | Pre-built LiteLLM Redis entries for reuse across tests. |
| `redis_entries_mixed` | function | `tests/test_ingest/conftest.py` | `dict[bytes, bytes]` | Mixed-format entries (LiteLLM + LangChain + RedisVL). |
| `mock_redis_client` | function | `tests/test_ingest/conftest.py` | MagicMock factory | Shared helper, currently duplicated as `_make_mock_redis()` in test_redis_adapter.py. |

### New fakes required

| Name | Location | Type | Notes |
|---|---|---|---|
| `FakeLangSmithClient` | `tests/fakes/langsmith.py` | Plain fake | Replaces MagicMock with a simple class that has `list_runs(project_name, execution_order, **kwargs)` returning a list of `FakeRun` objects. Contract-verified: method signature matches real `langsmith.Client.list_runs`. |
| `FakeBraintrustClient` | `tests/fakes/braintrust.py` | Plain fake | `projects.retrieve(name)` returns a `FakeProject` with `.logs.list()`. |
| `FakeRedisClient` | `tests/fakes/redis.py` | Plain fake | `scan(cursor)` and `get(key)` methods. Simulates cursor pagination over an in-memory dict. More realistic than MagicMock side_effect chains. |

### Coordination points (cross-subagent shared fixtures)

| Fixture | Shared with | Issue |
|---|---|---|
| `sample_pairs` in `tests/conftest.py` | All subagent test suites | Currently returns 3 pairs. If ingest tests depend on it, changes from other teams could break ingest tests. Consider making ingest tests self-contained. |
| `sample_jsonl_file` in `tests/conftest.py` | Pipeline tests, CLI tests | Same concern. Also: uses `delete=False` temp file with no cleanup. |
| `PromptPair` model in `core/types.py` | All subsystems | Any field addition/removal to PromptPair breaks all test factories. This is desired (contract enforcement) but requires coordination. |

---

## 7. Gaps You Can't Close

1. **Real LangSmith API behavior**: The `langsmith.Client.list_runs()` return type and pagination behavior can change with SDK updates. Our fakes assume a stable interface. Only real-API E2E tests catch drift, and those cost nothing monetarily but require API keys and are flaky (network, rate limits). **Risk: adapter breaks silently when langsmith SDK updates.**

2. **Real Braintrust API behavior**: Same as LangSmith. The `braintrust.Braintrust().projects.retrieve().logs.list()` call chain is deep and undocumented. Mock structure may diverge from reality.

3. **Redis format evolution**: LiteLLM, LangChain, RedisVL, and GPTCache all control their own cache entry formats. If any of them changes their serialization (e.g., LiteLLM v2 changes `response.choices` structure), our parsers break. We can't test against every version of every upstream cache library. **Mitigation: pin upstream versions in docs and add a `test_upstream_format_smoke` that imports each library and checks a known format.**

4. **Memory behavior under load**: All adapters load the full dataset into a `list[PromptPair]` in memory. Testing memory exhaustion for 10M+ pairs requires a controlled environment and is expensive to run. We can document the limitation but can't regression-test it.

5. **Non-deterministic Redis SCAN ordering**: Real Redis SCAN does not guarantee order and may return duplicates across cursor iterations. Our mock always returns exact, non-overlapping results. Testing with real Redis in Docker would catch issues here, but SCAN non-determinism means the test itself becomes flaky.

6. **File system edge cases**: Symlinks, FIFO pipes, `/dev/null` as input, network-mounted filesystems with latency — these are platform-specific and not worth testing in CI.

---

## 8. Cost and Time Estimate

### Write-time per item

| Category | Count | Per-item | Total |
|---|---|---|---|
| Unit tests (S) | 20 | 15 min | 5 hours |
| Contract tests (S/M) | 5 | 30 min | 2.5 hours |
| Integration tests (S/M/L) | 6 | 45 min avg | 4.5 hours |
| Property-based tests (M/L) | 8 | 45 min avg | 6 hours |
| Fixtures + fakes | 10 | 20 min avg | 3.3 hours |
| Golden fixture files | 9 | 15 min avg | 2.25 hours |
| Generator scripts | 1 | 1 hour | 1 hour |
| **Total** | | | **~24.5 hours** |

### CI time per run

| Tier | Time | Run frequency |
|---|---|---|
| Unit + contract | < 5 sec | Every push |
| Property-based | ~15 sec (with `max_examples=100`) | Every push |
| Integration (Docker Redis) | ~10 sec (with pre-pulled image) | Every push (or nightly) |
| E2E (real APIs) | ~30 sec per adapter | Nightly / manual trigger only |

### Real API cost

- LangSmith: $0 (free tier, read-only)
- Braintrust: $0 (free tier, read-only)
- Redis: $0 (Docker)
- SyntheticDataGenerator (out of scope): ~$0.50 per full generation run via LiteLLM

---

## 9. Path to Production

### Current readiness: **Beta**

The ingest subsystem has 6 adapters, all with substantial test coverage (120+ tests across the boundary). The core path (JSONL, CSV, splitter) is well-tested. The API-calling adapters (LangSmith, Braintrust) and Redis are tested only through mocks. No integration tests against real services exist.

### Gap to production-hardened

| Gap | Severity | Effort |
|---|---|---|
| Redis format auto-detection untested with mixed data | CRITICAL | M |
| No integration test against real Redis | HIGH | L |
| OTel event extraction may return wrong attribute | HIGH | S |
| Empty-string prompt/response accepted silently | HIGH | S (decide + implement) |
| LangSmith/Braintrust mock fidelity unknown | MEDIUM | M |
| No property-based tests for splitter/dedup | MEDIUM | M |
| No encoding-mismatch tests | MEDIUM | S |
| No golden fixtures checked in | LOW | M |

### Gates (conditions for production)

1. All CRITICAL and HIGH items from risk ranking (section 3) are covered by tests.
2. Redis adapter passes integration test against real Redis (Docker) with mixed-format data.
3. OTel `_extract_from_events` either has a deterministic attribute selection rule (tested) or is refactored.
4. Property-based tests for `deduplicate` and `split_data` pass with `max_examples=1000`.
5. Empty-string prompt/response handling has a deliberate design decision (reject or accept) with tests enforcing it.
6. All golden fixture files are checked in and loaded by at least one test.
7. CI runs all tiers except E2E on every push, with < 30 sec total runtime.

### Ordered sequence of work

1. **S — Fix OTel event extraction** (section 3, risk #3): Add test exposing multi-attribute event behavior. If it's a bug, fix the source. If it's intentional, document it.
2. **S — Decide on empty-string handling** (section 3, risk #4): Add `test_schema_empty_string_prompt` and either add a validator or document acceptance. NEEDS_HUMAN_REVIEW: Should empty strings be rejected?
3. **S — Add 20 unit tests** (section 4): These are all small, self-contained, and can be written quickly. Covers fingerprint, schema, CSV edge cases, Redis parsers, splitter edge cases.
4. **M — Write property-based tests for splitter** (section 4): 4 Hypothesis tests for dedup and split invariants.
5. **M — Write Redis integration test with Docker** (section 4): Use `RedisPopulator` to write mixed-format data, `RedisAdapter` to read it back.
6. **M — Create golden fixtures** (section 5): 9 fixture files checked into `tests/fixtures/ingest/`.
7. **M — Build fake clients** (section 6): `FakeRedisClient`, `FakeLangSmithClient`, `FakeBraintrustClient` — plain fakes replacing MagicMock chains.
8. **L — Add contract tests** (section 4): Cross-adapter invariants.
9. **L — Add remaining property-based tests** (section 4): CSV and JSONL roundtrip with arbitrary content.

### Smallest slice to move from beta to production-capable

Items 1-3 above. Fix the OTel bug-or-document-it, decide on empty strings, and add the 20 unit tests. This closes the CRITICAL risk (Redis mix-format detection gets a unit test as part of item 3) and all HIGH-severity gaps except the Redis integration test. Estimated time: ~8 hours.

### Dependencies on other boundaries

- **Pipeline boundary** (`_build_adapter` in `pipeline.py`): Owns the adapter factory. Changes to `MigrationConfig` adapter fields require coordination.
- **Core types** (`PromptPair`): Any field changes require updating all adapter tests and fixture factories.
- **Testing utilities** (`src/rosettastone/testing/`): `RedisPopulator` and `SyntheticDataGenerator` are shared infrastructure. Changes there could affect ingest integration tests.
- **Evaluate boundary**: Consumes `list[PromptPair]` from ingest. If evaluation starts requiring fields that adapters don't populate (e.g., `output_type`), ingest needs to change. Currently `output_type` is set to `None` by all adapters — the evaluate boundary auto-detects it.
