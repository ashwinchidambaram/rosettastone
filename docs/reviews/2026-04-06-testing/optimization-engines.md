# Optimization & DSPy Integration -- Testing Review

**Boundary**: `src/rosettastone/optimize/` (all modules)
**Tests**: `tests/test_optimize/` (all files)
**Date**: 2026-04-06

---

## 1. Boundary Map

```
                         INSIDE (we test)
    +------------------------------------------------------------+
    |  base.py          Optimizer ABC                             |
    |  gepa.py          GEPAOptimizer, GEPATimeoutWithResult      |
    |  mipro.py         MIPROv2Optimizer                          |
    |  metric.py        build_migration_metric, IterationTracker  |
    |  dspy_program.py  MigrationSignature, MigrationProgram      |
    |  pipeline_optimizer.py  PipelineProgram, optimize_pipeline  |
    |  pipeline_config.py     PipelineConfig, validate_dag, YAML  |
    |  teacher_student.py     TeacherStudentOptimizer             |
    |  feedback.py      build_feedback_map, prepend_feedback      |
    |  improvement.py   ImprovementScorer, blended scoring        |
    |  utils.py         extract_optimized_instructions            |
    +------------------------------------------------------------+
                              |
    ----- BOUNDARY (mock/stub these) ----------------------------
                              |
    +-- EXTERNAL DEPENDENCIES ---------------------+
    |  dspy.GEPA           (DSPy optimizer)         |  <-- mock
    |  dspy.MIPROv2        (DSPy optimizer)         |  <-- mock
    |  dspy.LM             (LiteLLM wrapper)        |  <-- mock
    |  dspy.context        (LM context manager)     |  <-- mock
    |  dspy.ChainOfThought (DSPy module)            |  <-- mock (pipeline)
    |  dspy.Prediction     (DSPy data class)        |  <-- real (thin Pydantic)
    |  dspy.Example        (DSPy data class)        |  <-- real (thin Pydantic)
    |  litellm.completion  (improvement scorer LLM) |  <-- mock
    |  bert_score          (optional dep)           |  <-- mock/ImportError
    |  sentence_transformers (optional dep)         |  <-- mock/ImportError
    |  PyYAML              (pipeline config YAML)   |  <-- real (pure parsing)
    +-----------------------------------------------+
                              |
    +-- CROSS-BOUNDARY (other subsystems) ---------+
    |  evaluate/bertscore.py   compute_bertscore    |  <-- mock at source
    |  evaluate/embedding.py   compute_embedding_sim|  <-- mock at source
    |  evaluate/exact_match.py string_similarity    |  <-- mock at source
    |  config.py               MigrationConfig      |  <-- real (Pydantic model)
    |  core/types.py           PromptPair, etc.     |  <-- real (data classes)
    |  core/pipeline.py        optimize_prompt()    |  <-- consumer of our code
    |  core/migrator.py        Migrator.run()       |  <-- consumer of our code
    +-----------------------------------------------+
                              |
    +-- ON THE FENCE (integration candidates) -----+
    |  Local Ollama/vLLM       (real LLM endpoint)  |  <-- integration tier
    |  DSPy GEPA compilation   (real DSPy run)      |  <-- e2e tier
    +-----------------------------------------------+
```

**External dependencies by module:**
- `gepa.py`: dspy.GEPA, dspy.LM, dspy.context, concurrent.futures
- `mipro.py`: dspy.MIPROv2, dspy.LM, dspy.context
- `metric.py`: evaluate/bertscore, evaluate/embedding, evaluate/exact_match (lazy import chain)
- `improvement.py`: litellm.completion (via `_call_litellm_completion`)
- `pipeline_optimizer.py`: dspy.GEPA, dspy.LM, dspy.context, dspy.ChainOfThought
- `teacher_student.py`: dspy.LM (for teacher generation)
- `pipeline_config.py`: PyYAML (for YAML loading)
- `dspy_program.py`: dspy.Signature, dspy.ChainOfThought, dspy.Module (structural only)

---

## 2. Current Coverage Audit

### `test_gepa.py` -- 12 tests

**Covers:**
- `extract_optimized_instructions`: primary path (predict.signature.instructions), fallback via named_predictors, error when no instructions found, coercion of non-string instructions
- `GEPAOptimizer.optimize()`: return type, target/reflection LM construction, GEPA kwargs wiring (auto, num_threads), trainset DSPy Example field correctness (expected_response, prompt-as-input), extracted instructions forwarding, ABC conformance, lm_extra_kwargs conflict filtering for reflection_lm

**Misses:**
- No test for `gepa_max_metric_calls` branch (the `if gepa_max_metric_calls is not None` path that skips `auto`)
- No test for `on_iteration` callback wiring inside GEPAOptimizer
- No test for `iteration_history_out` parameter population
- No test for the `_iteration_capturing_metric` wrapper behavior (snapshot extraction, failure swallowing)
- No test for empty trainset edge case
- No test that `best_intermediate` list is thread-safe under concurrent metric calls during GEPA compilation

**Brittle tests:** None identified. The mock patching pattern is consistent and targets the right import paths.

**False-confidence tests:** `test_coerces_non_string_instructions_to_string` -- the MagicMock `__str__` return "Coerced instructions" but the assertion only checks `isinstance(result, str)`, not that the coerced value is correct. A bug in `str()` coercion logic could pass this test.

### `test_gepa_timeout.py` -- 14 tests

**Covers:**
- TimeoutError propagation when no intermediate exists, error message content
- GEPATimeoutWithResult with single/multiple intermediates, most-recent selection, exception attributes
- Normal completion (no timeout), timeout=None direct execution path, future.result timeout value
- Immediate callback with intermediate capture before timeout
- Config validation: gepa_timeout_seconds minimum=30, validation errors
- Migrator-level timeout handling: GEPATimeoutWithResult warning propagation, intermediate-as-optimized-prompt, bare TimeoutError re-raise

**Misses:**
- No test for `executor.shutdown(wait=False)` being called (the comment in source specifically warns about this)
- No test for race condition where extract fails on every call but best_intermediate has stale entries
- No test for the metric being called 0 times before timeout (future.result raises immediately)

**Brittle tests:** The `_run_with_intermediates` helper is complex -- it patches 6 things simultaneously and uses `real_submit` to run GEPA synchronously then simulate timeout. If the internal wiring of GEPAOptimizer changes (e.g., metric wrapping order), this entire helper breaks.

**False-confidence tests:** None identified. The tests correctly exercise the timeout paths.

### `test_metric.py` -- 25 tests

**Covers:**
- Return type (dspy.Prediction with score/feedback), score range [0, 1], score capping
- Feedback threshold bands: <0.7 ("diverges"), 0.7-0.85 ("partially"), >=0.85 ("Good match")
- Boundary values at 0.7 and 0.85
- trace parameter acceptance (DSPy passes this)
- Import fallback chain: BERTScore -> embedding -> string_similarity
- Known-issue weight: score division, clamping at 0, no effect on non-issue pairs, config validation
- IterationTracker: callback firing after full sweep, per-sweep counts, zero trainset no-op, cumulative mean scores, plain float fallback, thread safety (single sweep + multi-sweep), history capture, thread-safe history, get_history returns copy

**Misses:**
- No test for `improvement_objectives` integration in the metric (the blended score path with `improvement_scorer is not None`)
- No test for non-string prompt (list[dict]) with known-issue feedback map lookup (the JSON encoding path inside the metric closure)
- No test for `pred_name` / `pred_trace` parameters (DSPy may pass these during GEPA)
- Fallback tests use `patch.dict(sys.modules)` which may not accurately simulate a missing package (the test patches the module to `None` but the import inside the closure does `from rosettastone.evaluate.bertscore import compute_bertscore`, which imports the module file, not the external bert_score package directly)

**Brittle tests:** The fallback tests (`test_falls_back_to_embedding_when_bertscore_missing`) use `patch.dict(sys.modules, {"bert_score": None})` -- this patches the top-level `bert_score` package, but the import inside the metric is `from rosettastone.evaluate.bertscore import compute_bertscore`. The test works because `rosettastone.evaluate.bertscore` itself imports `bert_score` at the top. If that import structure changes (e.g., lazy import inside `compute_bertscore`), the test would silently stop testing the fallback path.

**False-confidence tests:**
- `test_score_extraction_fallback`: Creates a new tracker+wrapped metric per call rather than reusing the same wrapped function across calls. In production the same wrapped function is called N times. This test wouldn't catch a bug where `wrap()` creates a broken closure on second call.

### `test_dspy_program.py` -- 9 tests

**Covers:**
- MigrationSignature: prompt/response field existence, input/output field type designations, exactly 2 fields
- MigrationProgram: predict attribute existence, ChainOfThought type, inner signature field binding, inner field types, dspy.Module subclass

**Misses:**
- No test for `MigrationProgram.forward()` -- the actual execution path. Tests only verify structural wiring.
- No test for `__init__` setting `self.predict = dspy.ChainOfThought(MigrationSignature)` correctly (only the type is checked, not the signature binding)

**False-confidence tests:** None. These are pure structural tests and are honest about what they verify.

### `test_mipro.py` -- 10 tests

**Covers:**
- ABC conformance, optimize method existence
- Return type (string), extracted instructions forwarding
- Zero-shot config: max_bootstrapped_demos=0, max_labeled_demos=0 (PII safety)
- Config wiring: target_model, mipro_auto, None->light fallback, num_threads
- Trainset construction: expected_response field, prompt-as-input, empty trainset

**Misses:**
- No test for `lm_extra_kwargs` forwarding (GEPA tests have this, MIPROv2 doesn't)
- No test for the `dspy.context(lm=target_lm)` context manager correctly scoping the LM

**False-confidence tests:** None.

### `test_pipeline_optimizer.py` -- 8 tests

**Covers:**
- `_build_signature`: field creation, docstring, empty fields
- `PipelineProgram`: init creates predictors, predict_{name} attributes, forward returns Prediction, missing input defaults to empty string, topological execution order
- `optimize_pipeline`: calls GEPA.compile, returns dict keyed by module names

**Misses:**
- No test for `optimize_pipeline` extracting per-module instructions (the `extended_signature.instructions` path vs `extract_optimized_instructions` fallback)
- No test for `optimize_pipeline` with `lm_extra_kwargs` conflict filtering
- No test for `optimize_pipeline` with non-empty trainset
- No test for `PipelineProgram.forward` output field accumulation across modules (testing that step1's output is passed as step2's input)
- `_build_signature` doesn't verify field types (InputField vs OutputField) -- only field name presence

**False-confidence tests:** `test_optimize_pipeline_returns_dict` -- patches `dspy.context` without configuring `__enter__`/`__exit__`, relying on MagicMock's default behavior. If the `dspy.context` usage changes (e.g., to a `with` statement variant), the mock would silently break.

### `test_pipeline_config.py` -- 9 tests

**Covers:**
- YAML loading: nested `pipeline:` key, top-level dict, field values
- DAG validation: linear chain, parallel branches, cycle detection, missing dependency, field-not-produced-by-upstream, empty modules, single module, defaults

**Misses:**
- No test for invalid YAML content (malformed YAML, missing required fields)
- No test for duplicate module names
- No test for self-referencing depends_on (module depends on itself)
- No test for diamond DAG patterns (A->B, A->C, B->D, C->D)

**False-confidence tests:** None.

### `test_teacher_student.py` -- 7 tests

**Covers:**
- `optimize()` raises NotImplementedError
- `generate_teacher_demos`: returns PromptPair list, empty trainset, uses first module input field, source_model set correctly
- `pipeline_optimize`: calls optimize_pipeline with teacher demos, returns dict[str, str]

**Misses:**
- No test for `generate_teacher_demos` when dspy.LM returns a non-list (the `else` branch: `str(raw_response)`)
- No test for `generate_teacher_demos` when dspy.LM returns empty list (the `raw_response[0] if raw_response else ""` path)
- No test for `generate_teacher_demos` with non-string prompts (list[dict] prompt type)
- No test for teacher_bootstrapped metadata being set on returned demos

**False-confidence tests:** None.

### `test_feedback.py` -- 12 tests

**Covers:**
- `build_feedback_map`: empty trainset, single/multiple pairs, None filtering, str key encoding, list->JSON encoding, sort_keys stability
- `prepend_feedback`: None known_issue pass-through, non-None prepending, exact format, empty base, empty known_issue string, return type

**Misses:**
- No test for duplicate prompts in trainset (which feedback wins?)
- No test for very large feedback strings (memory/performance)

**False-confidence tests:** None. These are pure function tests with clear assertions.

### `test_improvement.py` -- 24 tests

**Covers:**
- `build_improvement_scorer`: return type, one-per-objective, score normalization, non-empty feedback, correct judge_model, litellm failure graceful handling, unparseable response, single/multiple objectives, prompt/expected/actual passed to LLM
- `build_improvement_feedback`: combined output, empty scores pass-through, objective descriptions in output, numeric scores in output, feedback text in output, structured format
- `compute_blended_score`: default weight, custom weight, weight=0 pure equivalence, weight=1 pure improvement, unit interval, edge cases (perfect+zero, zero+perfect), multi-score averaging
- `ImprovementObjective`: default weight, custom weight, common objectives, empty objectives scorer, description preserved, optional params

**Misses:**
- No test for `_parse_score_and_feedback` edge cases: score outside 1-5, decimal scores like "Score: 3.5", "Score:" with no number
- No test for `_escape_xml` function (prompt injection via XML in user content)
- No test for PipelineContext cost tracking (`ctx.add_cost` path in `_score_objective`)
- No test for `expected_response=None` path in `_score_objective` (baseline_section omitted)
- No test for `compute_blended_score` with empty improvement_scores list (returns `(1-w)*eq + w*0`)

**False-confidence tests:** None.

---

## 3. Risk Ranking

| # | Failure Mode | Manifestation | Likelihood x Blast | Existing Test? | Severity |
|---|---|---|---|---|---|
| 1 | **DSPy version incompatibility breaks instruction extraction** | `extract_optimized_instructions` raises InstructionExtractionError in prod. Optimizer returns no result. Migration fails entirely. | DSPy is pre-1.0 and changes its internal layout often. | Partially -- tests mock the structure, but a real DSPy upgrade could silently break the `predict.signature.instructions` path without tests catching it. | **CRITICAL** |
| 2 | **Metric returns invalid score/feedback format** | GEPA/MIPROv2 internal error during compilation. Silent optimization degradation (GEPA can't reflect if feedback is wrong). Bad optimized prompts go to production. | Medium (import fallback chain is fragile, blended score math could produce NaN/Inf). | Partially -- threshold bands tested but no integration test of the full metric with improvement objectives wired in. | **HIGH** |
| 3 | **GEPA timeout leaves background thread running indefinitely** | Memory leak, thread exhaustion in long-running server. `executor.shutdown(wait=False)` means GEPA thread is never killed. | High in CI/server scenarios. | No -- no test verifies shutdown behavior or thread lifecycle. | **HIGH** |
| 4 | **LLM-as-judge prompt injection via improvement scoring** | Adversarial prompt content inside `<prompt>` or `<response>` XML tags breaks out, manipulates scores. `_escape_xml` only escapes `&`, `<`, `>` -- doesn't handle CDATA, encoding tricks. | Low likelihood but high blast (wrong migration recommendation). | No -- `_escape_xml` is untested. | **HIGH** |
| 5 | **Cost overrun during GEPA optimization** | No cost tracking inside GEPAOptimizer. GEPA makes unbounded LLM calls (controlled only by `max_metric_calls`). With expensive models, a "heavy" preset could cost hundreds of dollars. | Medium -- users may not set `gepa_max_metric_calls` or `max_cost_usd`. | No -- no test for cost accumulation or cost-cap enforcement during optimization. | **HIGH** |
| 6 | **Known-issue weight divides by config value but score is already blended** | When both `improvement_objectives` and `known_issue_weight` are active, the blended score gets divided by the weight, which double-penalizes. The ordering in `migration_metric` is: compute blended score, then divide by weight. | Low (requires both features active simultaneously). | No -- no test for the interaction of improvement blending + known-issue weighting. | **MEDIUM** |
| 7 | **Pipeline DAG field validation misses transitive dependencies** | A module in a diamond DAG can read a field from a non-direct-upstream module if it's in `available_fields` from earlier topological processing. The validation is order-dependent. | Low (DAG configs are rare, and the validation is correct for stated semantics). | Partially -- no diamond DAG test. | **MEDIUM** |
| 8 | **Teacher LM call failure silently produces empty demos** | `generate_teacher_demos` has no error handling for dspy.LM call failures. A single LLM error would crash the entire pipeline_optimize call. | Medium in production. | No -- no test for LM call failure during teacher generation. | **MEDIUM** |
| 9 | **MIPROv2 lm_extra_kwargs not filtered for conflict keys** | MIPROv2 doesn't create a reflection_lm, so no conflict filtering needed currently. But if MIPROv2 adds reflection support, the same bug that was fixed in GEPA could appear. | Low (future risk only). | No. | **LOW** |
| 10 | **Empty trainset with on_iteration callback causes no-op tracker** | When trainset is empty + on_iteration is provided, the tracker check `len(trainset) > 0` prevents tracker creation. The on_iteration callback is silently never called. Caller may wait forever for progress. | Low (empty trainset is uncommon). | No explicit test for this interaction. | **LOW** |

---

## 4. Test Plan by Tier

### Unit Tests

| ID | Test | Asserts | Status | Time |
|----|-------|---------|--------|------|
| U1 | `gepa_max_metric_calls` branch bypasses `auto` | GEPA kwargs contain `max_metric_calls` and no `auto` key | MISSING | S |
| U2 | `on_iteration` callback wiring in GEPAOptimizer | Callback fires with correct (iteration, total, score) after GEPA metric calls | MISSING | M |
| U3 | `iteration_history_out` populated after optimize() | Output list contains tracker history dicts | MISSING | M |
| U4 | `_iteration_capturing_metric` swallows extraction failures | Metric returns valid score when `extract_optimized_instructions` raises | MISSING | S |
| U5 | `_iteration_capturing_metric` appends to best_intermediate | best_intermediate grows after each metric call | MISSING | S |
| U6 | metric with `improvement_objectives` produces blended score | Score = (1-w)*equivalence + w*avg(improvement), feedback includes improvement section | MISSING | M |
| U7 | metric with non-string prompt + known-issue feedback map | JSON key encoding matches, score divided by weight | MISSING | S |
| U8 | `_parse_score_and_feedback` edge cases | Score: 3.5 -> 0.625, Score: 6 -> None, "Score:" with no digit -> None | MISSING | S |
| U9 | `_escape_xml` with angle brackets and ampersands | `<script>` -> `&lt;script&gt;`, `&` -> `&amp;` | MISSING | S |
| U10 | `_score_objective` without expected_response | baseline_section omitted from LLM prompt | MISSING | S |
| U11 | `compute_blended_score` with empty improvement_scores | Returns `(1-w)*equivalence + w*0.0` | MISSING | S |
| U12 | `build_feedback_map` with duplicate prompts | Last-write-wins or defined behavior | MISSING | S |
| U13 | `generate_teacher_demos` non-list LM response | `str(raw_response)` path exercised | MISSING | S |
| U14 | `generate_teacher_demos` empty list LM response | Returns PromptPair with `response=""` | MISSING | S |
| U15 | `generate_teacher_demos` metadata includes `teacher_bootstrapped` | `demo.metadata["teacher_bootstrapped"] == True` | MISSING | S |
| U16 | MIPROv2 `lm_extra_kwargs` forwarding | Extra kwargs passed to dspy.LM | MISSING | S |
| U17 | `MigrationProgram.forward()` with mocked LM | Returns dspy.Prediction with `response` field | MISSING | M |
| U18 | `validate_dag` diamond pattern | A->B, A->C, B->D, C->D succeeds with valid order | MISSING | S |
| U19 | `validate_dag` self-referencing dependency | Module depends on itself raises ValueError | MISSING | S |
| U20 | `validate_dag` duplicate module names | Defined behavior (error or dedup) | MISSING | S |
| U21 | `PipelineProgram.forward` output field accumulation | step1 output available as step2 input with correct value | PARTIAL (execution order tested, value propagation not) | S |
| U22 | `optimize_pipeline` per-module instruction extraction | extended_signature path and fallback path both return correct per-module instructions | MISSING | M |
| U23 | `extract_optimized_instructions` primary path (EXISTS) | Already covered | EXISTS | -- |
| U24 | `extract_optimized_instructions` fallback path (EXISTS) | Already covered | EXISTS | -- |
| U25 | `extract_optimized_instructions` error path (EXISTS) | Already covered | EXISTS | -- |
| U26 | IterationTracker callback timing (EXISTS) | Already covered | EXISTS | -- |
| U27 | IterationTracker thread safety (EXISTS) | Already covered | EXISTS | -- |
| U28 | Metric threshold bands (EXISTS) | Already covered | EXISTS | -- |
| U29 | Known-issue weight (EXISTS) | Already covered | EXISTS | -- |
| U30 | GEPA wiring tests (EXISTS) | Already covered | EXISTS | -- |
| U31 | MIPROv2 zero-shot config (EXISTS) | Already covered | EXISTS | -- |
| U32 | Feedback map/prepend (EXISTS) | Already covered | EXISTS | -- |
| U33 | Improvement scorer/blending (EXISTS) | Already covered | EXISTS | -- |

### Contract Tests

| ID | Test | Asserts | Status | Time |
|----|-------|---------|--------|------|
| C1 | Optimizer ABC contract: GEPAOptimizer | `optimize(train_set, val_set, config) -> str` | EXISTS | -- |
| C2 | Optimizer ABC contract: MIPROv2Optimizer | `optimize(train_set, val_set, config) -> str` | EXISTS | -- |
| C3 | Optimizer ABC contract: TeacherStudentOptimizer raises NotImplementedError | `optimize()` not supported, `pipeline_optimize()` required | EXISTS | -- |
| C4 | DSPy metric contract: returns `dspy.Prediction(score, feedback)` | GEPA/MIPROv2 depend on this exact return type | EXISTS | -- |
| C5 | DSPy Example contract: trainset uses `expected_response` + `.with_inputs("prompt")` | Metric reads `gold.expected_response`, GEPA injects `prompt` | EXISTS | -- |
| C6 | `pipeline.optimize_prompt()` consumer contract | Calls GEPAOptimizer or MIPROv2Optimizer based on config.mipro_auto | MISSING | M |
| C7 | `MigrationResult.optimization_iterations` contract | GEPAOptimizer populates iteration history that `build_result` can consume | MISSING | M |
| C8 | `optimize_pipeline` return contract | Returns `dict[str, str]` mapping module_name -> instructions | EXISTS | -- |

### Integration Tests

| ID | Test | Asserts | Status | Time |
|----|-------|---------|--------|------|
| I1 | GEPAOptimizer with local Ollama endpoint (tiny model) | `optimize()` returns non-empty string, GEPA completes within timeout, no exceptions | MISSING | L |
| I2 | MIPROv2Optimizer with local Ollama endpoint | Same as I1 but via MIPROv2 | MISSING | L |
| I3 | `build_migration_metric` with real `string_similarity` (no mocks) | Score is in [0,1], feedback is non-empty, identical strings score ~1.0 | MISSING | S |
| I4 | `build_migration_metric` with real `compute_bertscore` (if installed) | Score is in [0,1], semantically similar strings score > 0.7 | MISSING | M |
| I5 | `PipelineProgram.forward()` with local Ollama endpoint | Multi-module execution produces non-empty Prediction fields | MISSING | L |
| I6 | Full `optimize_prompt()` through pipeline with gepa_max_metric_calls=5 | Returns non-empty instructions within budget | MISSING | XL |

### Property-Based Tests

| ID | Test | Asserts | Status | Time |
|----|-------|---------|--------|------|
| P1 | `compute_blended_score` for arbitrary (eq, [scores], weight) | Result always in [0.0, 1.0] for any valid inputs | MISSING | S |
| P2 | `build_feedback_map` with arbitrary PromptPair lists | Map length <= input length, all keys are strings, no None values | MISSING | S |
| P3 | `prepend_feedback(base, issue)` | Result always starts with "KNOWN ISSUE:" when issue is not None, result == base when issue is None | MISSING | S |
| P4 | `_parse_score_and_feedback` with arbitrary strings | Never raises, normalized score always None or in [0, 1] | MISSING | M |
| P5 | `validate_dag` with randomly generated DAGs | No false positives (valid DAGs accepted), no false negatives (cyclic DAGs rejected) | MISSING | M |
| P6 | `migration_metric` score always in [0, 1] for any sem_score and known_issue_weight | Clamping works regardless of input values | MISSING | S |
| P7 | `IterationTracker` callback count = call_count // trainset_size | For any trainset_size > 0 and any number of calls | MISSING | S |

### End-to-End Tests

| ID | Test | Asserts | Status | Time | Est. Cost |
|----|-------|---------|--------|------|-----------|
| E1 | GEPA optimization with OpenAI GPT-4o-mini (gepa_max_metric_calls=10) | Returns valid optimized instructions, metric scores improve or stay stable, no errors | MISSING | XL | ~$0.50 |
| E2 | MIPROv2 optimization with OpenAI GPT-4o-mini | Same as E1 via MIPROv2 path | MISSING | XL | ~$0.50 |
| E3 | Full Migrator.run() with GEPA timeout (real API, short timeout) | GEPATimeoutWithResult raised with usable instructions | MISSING | XL | ~$1.00 |
| E4 | Improvement scorer with real judge model | Scores vary meaningfully between good and bad responses | MISSING | L | ~$0.20 |

---

## 5. Synthetic Data Generation Strategy

### What "realistic" means for this boundary

For optimization testing, "realistic" means:
1. **PromptPair data** that has varied prompt structures (simple questions, multi-turn conversations, JSON-producing prompts, long-form generation)
2. **Score distributions** that match production patterns (most scores cluster 0.6-0.9, rare outliers at extremes)
3. **DSPy compiled program structures** that match what GEPA/MIPROv2 actually produce (signature field layouts, named_predictors iteration)
4. **LLM response patterns** that include typical failure modes (truncation, refusal, format drift, hallucination)

### How we generate it

**Tier 1: Static fixtures (no cost, instant)**
- Curated `PromptPair` lists covering each `OutputType` (JSON, CLASSIFICATION, SHORT_TEXT, LONG_TEXT)
- 20-50 pairs per type, manually written to exercise edge cases (empty responses, unicode, very long prompts, list-of-dict prompts)
- Stored in `tests/fixtures/synthetic_pairs/` as JSONL files
- Include pairs with and without `feedback` field for known-issue testing

**Tier 2: Parametric generators (no cost, instant)**
- Hypothesis strategies for `PromptPair`, `ImprovementScore`, `ImprovementObjective`, `PipelineConfig`
- Score distributions via `st.floats(min_value=0.0, max_value=1.0)` for metric testing
- Random DAG generation via `st.lists(st.text())` + random edge insertion for validate_dag

**Tier 3: VCR cassettes (one-time cost, then free)**
- Record real API responses from cheap models (GPT-4o-mini, Claude Haiku) for:
  - `build_improvement_scorer` calls (10-20 cassettes covering various objective types)
  - `generate_teacher_demos` LM calls (5-10 cassettes)
  - DSPy GEPA/MIPROv2 compilation traces (2-3 cassettes with gepa_max_metric_calls=5)
- Use `vcrpy` or `pytest-recording` to record/replay
- Store in `tests/cassettes/optimize/`
- Tag cassettes with DSPy version and model version for rot detection

### How we keep it stable

- Static fixtures are version-controlled and immutable
- Hypothesis seeds are fixed in CI (`@settings(database=None)` + explicit seed)
- VCR cassettes have a `recorded_with` header containing DSPy version, model version, and date
- CI job runs cassette freshness check weekly: re-record one cassette and diff against stored version

### How we prevent rot

- Fixtures include a `schema_version` field; a test asserts that all fixtures match current `PromptPair` schema
- VCR cassettes have a max-age of 90 days; CI warns if any cassette exceeds this
- Property-based tests with `@settings(max_examples=200)` catch drift in invariants that fixed data would miss

### Cost profile

- Tier 1 + Tier 2: $0
- Tier 3 initial recording: ~$2-5 (cheap models, small budgets)
- Tier 3 refresh (quarterly): ~$2-5
- Total annual synthetic data cost: <$25

---

## 6. Fixtures, Fakes, and Mocks

### New fixtures needed

| Fixture | Scope | Location | Type | Notes |
|---------|-------|----------|------|-------|
| `fake_compiled_program` | function | `tests/test_optimize/conftest.py` | Factory fixture | Configurable mock with `predict.signature.instructions`, `named_predictors()`, optional `predict_{name}` attrs. Replaces repeated mock setup in test_gepa, test_mipro, test_pipeline_optimizer. |
| `minimal_migration_config` | function | `tests/test_optimize/conftest.py` | Factory fixture | `_make_config(tmp_path, **overrides)` -- already duplicated across 4 test files. Centralize. |
| `sample_prompt_pairs` | function | `tests/test_optimize/conftest.py` | Factory fixture | `_make_pairs(n, output_type)` -- duplicated across 3 files. |
| `fake_dspy_context` | function | `tests/test_optimize/conftest.py` | Context manager mock | The 3-line `mock_ctx.return_value.__enter__/__exit__` pattern is repeated in every test class. |
| `fake_litellm_response` | function | `tests/test_optimize/conftest.py` | Factory function | `_make_litellm_response(text)` from test_improvement.py -- useful for metric integration tests too. |
| `fake_metric` | function | `tests/test_optimize/conftest.py` | Callable | Returns `dspy.Prediction(score=X, feedback="stub")`. Already exists as `_make_stub_metric` in test_metric.py but not shared. |
| `pipeline_config_factory` | function | `tests/test_optimize/conftest.py` | Factory fixture | Linear, diamond, parallel pipeline configs for reuse across test_pipeline_config, test_pipeline_optimizer, test_teacher_student. |

### Contract-verified fakes

| Fake | Verifies against | Notes |
|------|------------------|-------|
| `FakeDSPyLM` | `dspy.LM` | Returns configurable responses for `__call__(messages=...)`. Tracks call count, messages, and model string. Used instead of `MagicMock(return_value=["response"])` to catch interface drift. |
| `FakeGEPAOptimizer` | `dspy.GEPA` | `.compile(program, trainset)` returns a compiled mock with configurable instructions. Calls metric N times to simulate real GEPA behavior. Used for integration-tier timeout testing. |

### Cross-subagent coordination points

| Shared fixture | Used by | Coordination needed |
|----------------|---------|---------------------|
| `sample_pairs` from `tests/conftest.py` | All subsystems | COORDINATION POINT: Currently has 3 pairs (CLASSIFICATION, SHORT_TEXT, JSON). Optimization tests need LONG_TEXT pairs too. Adding to root conftest affects all subsystems. |
| `MigrationConfig` from `config.py` | All subsystems | COORDINATION POINT: Any new config field added by optimization (e.g., a hypothetical `cost_tracking` field) must be backward-compatible with existing tests that construct MigrationConfig without it. |
| `PromptPair` from `core/types.py` | All subsystems | COORDINATION POINT: If `feedback` field semantics change, both feedback.py tests and evaluate subsystem tests need updating. |

---

## 7. Gaps You Can't Close

1. **DSPy internal behavior verification**: We mock `dspy.GEPA` and `dspy.MIPROv2` entirely. We cannot test whether our kwargs are correctly interpreted by DSPy. A DSPy update that renames `auto` to `preset` or changes `reflection_lm` semantics would be invisible to our unit tests. Only E2E tests with real DSPy catch this.

2. **Real GEPA optimization quality**: We can test that GEPA runs and returns instructions, but we cannot test whether those instructions are *good*. Optimization quality depends on the specific LLM, prompt distribution, and GEPA's internal search strategy -- none of which can be synthesized.

3. **Thread safety of `best_intermediate` reads during timeout**: The source code notes that reads of program state inside `extract_optimized_instructions` are "best-effort snapshots (not a guaranteed consistent state)." Testing this race condition reliably is not possible without instrumenting DSPy internals. NEEDS_HUMAN_REVIEW: Is the snapshot-consistency risk accepted as a known limitation, or should we add a lock around the extraction?

4. **Background thread cleanup after timeout**: When GEPA times out, the background thread keeps running. We cannot test that it eventually terminates or that it doesn't corrupt shared state. Python's `ThreadPoolExecutor` doesn't support thread cancellation. NEEDS_HUMAN_REVIEW: Should we document the maximum expected runtime for background GEPA threads, or add a process-level kill after 2x the timeout?

5. **Cost tracking accuracy**: Improvement scorer tracks costs via `ctx.add_cost()`, but GEPA/MIPROv2 don't track LLM call costs at all. LiteLLM tracks costs internally but DSPy doesn't expose them. Without instrumentation inside DSPy, we can't test cost accuracy.

6. **BERTScore/embedding model version sensitivity**: The metric's score depends on which BERTScore model or sentence-transformer model is loaded. Different versions produce different scores. We can't pin model versions in tests without downloading multi-GB models in CI.

---

## 8. Cost and Time Estimate

### Write-time per item

| Tier | Count (new) | Avg time | Total |
|------|-------------|----------|-------|
| Unit (S) | 15 | 20 min | 5 hrs |
| Unit (M) | 6 | 45 min | 4.5 hrs |
| Contract (M) | 2 | 45 min | 1.5 hrs |
| Integration (S) | 1 | 30 min | 0.5 hrs |
| Integration (M) | 1 | 60 min | 1 hr |
| Integration (L) | 3 | 2 hrs | 6 hrs |
| Integration (XL) | 1 | 3 hrs | 3 hrs |
| Property-based (S) | 4 | 30 min | 2 hrs |
| Property-based (M) | 3 | 60 min | 3 hrs |
| E2E (XL) | 3 | 3 hrs | 9 hrs |
| E2E (L) | 1 | 2 hrs | 2 hrs |
| Fixtures/fakes | 7 | 30 min | 3.5 hrs |
| VCR cassette setup | 1 | 3 hrs | 3 hrs |
| **Total** | | | **~44 hrs** |

### CI time per run

| Tier | Time | When |
|------|------|------|
| Unit + Contract + Property | ~15 seconds | Every push |
| Integration (local Ollama) | ~3-5 minutes | Every push (if Ollama available, skip otherwise) |
| E2E (real APIs) | ~10-15 minutes | Nightly or manual trigger |

### Real API cost

| Test | Model | Est. cost per run |
|------|-------|-------------------|
| E1 (GEPA GPT-4o-mini) | openai/gpt-4o-mini | ~$0.50 |
| E2 (MIPROv2 GPT-4o-mini) | openai/gpt-4o-mini | ~$0.50 |
| E3 (GEPA timeout) | openai/gpt-4o-mini | ~$1.00 |
| E4 (Improvement scorer) | openai/gpt-4o-mini | ~$0.20 |
| **Total per nightly run** | | **~$2.20** |
| **Monthly (30 runs)** | | **~$66** |

---

## 9. Path to Production

### Current production readiness level

**Level 2 of 5: "Structurally tested, not behaviorally verified."**

All source modules exist and have test files. The wiring between components (ABC contracts, DSPy type expectations, config forwarding) is well-tested. But:
- No test verifies that optimization actually produces useful output
- No test runs with a real LLM
- Cost tracking during optimization is absent
- Thread safety during timeout is assumed but not proven
- The improvement scorer's LLM-as-judge prompt is untested for injection resistance

### Gap between current and production-hardened

| Gap | Severity | Fix |
|-----|----------|-----|
| No real LLM integration test | High | I1/I2 with local Ollama |
| No cost tracking in GEPA/MIPROv2 | High | Instrument DSPy LM calls or use LiteLLM callbacks |
| No injection defense testing | High | U9 + adversarial prompt test |
| Metric improvement-blend path untested | Medium | U6 |
| No VCR cassettes for deterministic replay | Medium | Cassette recording setup |
| gepa_max_metric_calls path untested | Medium | U1 |
| Background thread lifecycle untested | Low | Document as accepted risk or add process-level timeout |

### Gates -- concrete conditions for production traffic

1. **Gate 1 (dev confidence)**: All unit tests pass, property tests find no new failures, `_escape_xml` tested with injection payloads
2. **Gate 2 (local integration)**: I1/I2 pass with Ollama (qwen2.5:0.5b or similar tiny model), optimization returns non-trivial instructions
3. **Gate 3 (API integration)**: E1/E2 pass with GPT-4o-mini, VCR cassettes recorded, cost per optimization is within 2x of estimate
4. **Gate 4 (production-ready)**: E3 timeout test passes, cost tracking instrumented, improvement scorer injection test passes, monitoring dashboards for optimization duration/cost/score exist

### Ordered sequence of work

1. **Week 1**: Centralize fixtures (conftest.py), write U1-U9 (pure logic gaps). No API costs. ~10 hrs.
2. **Week 2**: Write property-based tests P1-P7, write contract tests C6-C7. No API costs. ~7 hrs.
3. **Week 3**: Write U10-U22, remaining unit coverage. No API costs. ~10 hrs.
4. **Week 4**: Set up VCR cassette infrastructure, record initial cassettes. ~$5 API cost. ~6 hrs.
5. **Week 5**: Write integration tests I1-I4 (local Ollama). No API costs (local model). ~8 hrs.
6. **Week 6**: Write E1-E4, set up nightly CI job. ~$3 API cost. ~11 hrs.

### Smallest slice for next level

**Highest-impact smallest slice**: Write U1 (gepa_max_metric_calls branch), U6 (metric with improvement objectives), U9 (_escape_xml), and P1 (blended score property test). These four tests cover the three highest-severity gaps with ~3 hours of work and zero API cost.

### Dependencies on other boundaries

- **Evaluate boundary**: metric.py imports `compute_bertscore`, `compute_embedding_sim`, `string_similarity`. If their signatures change, metric tests break. The evaluate team should publish a contract test for these functions.
- **Config boundary**: `MigrationConfig` field additions/removals affect all optimizer tests that construct configs. Changes should be backward-compatible.
- **Pipeline/Migrator boundary**: `test_gepa_timeout.py` already tests Migrator-level handling. If `Migrator.run()` changes its error handling for `GEPATimeoutWithResult`, timeout tests need updating. The pipeline team owns this.
- **Safety boundary**: If PII scanning is added to the optimization step (scanning optimized prompts for PII), the optimization boundary expands. Currently PII scanning happens downstream in `pipeline.py`.
