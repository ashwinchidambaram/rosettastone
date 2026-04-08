# RosettaStone Tier 2: Production Readiness Technical Specification

**Author**: Principal Engineer / Measurement Scientist
**Date**: 2026-04-01
**Status**: Implementation-ready specification
**Prerequisite**: Tier 1 (P0 foundation) completed -- intermediate DB writes, task queue, cost guardrails, multi-user isolation all in place.

---

## Table of Contents

1. [Part 1: Human-Labeled Validation Dataset](#part-1-human-labeled-validation-dataset)
   - [1a. Label Schema Design](#1a-label-schema-design)
   - [1b. Dataset Composition](#1b-dataset-composition)
   - [1c. Pair Collection Pipeline](#1c-pair-collection-pipeline)
   - [1d. Labeling Methodology](#1d-labeling-methodology)
   - [1e. Calibration Methodology](#1e-calibration-methodology)
   - [1f. Integration with RosettaStone](#1f-integration-with-rosettastone)
2. [Part 2: Multi-Run Evaluation](#part-2-multi-run-evaluation)
3. [Part 3: Per-Prompt Regression Report](#part-3-per-prompt-regression-report)
4. [Part 4: Actual Cost Tracking](#part-4-actual-cost-tracking)
5. [Part 5: Shadow Deployment Tooling](#part-5-shadow-deployment-tooling)
6. [Part 6: GEPA Timeout](#part-6-gepa-timeout)
7. [Implementation Priority and Effort Estimates](#implementation-priority-and-effort-estimates)

---

## Part 1: Human-Labeled Validation Dataset

### Motivation

The recommendation engine in `src/rosettastone/decision/recommendation.py` uses four hardcoded thresholds to decide GO / CONDITIONAL / NO_GO:

```python
DEFAULT_THRESHOLDS = {
    "json": 0.95,
    "classification": 0.90,
    "short_text": 0.80,
    "long_text": 0.75,
}
```

These thresholds were set by engineering intuition. They have never been validated against ground truth. If the `json` threshold of 0.95 is too aggressive, the system rejects safe migrations. If the `long_text` threshold of 0.75 is too permissive, the system rubber-stamps migrations that would break in production. Both failure modes are expensive -- false negatives waste engineering time re-running migrations that were fine; false positives put broken migrations into production.

The Wilson confidence interval machinery in `statistics.py` is mathematically sound, but the operating point it is compared against is unvalidated. Calibrating these thresholds from labeled data is the single highest-leverage improvement to RosettaStone's decision quality.

---

### 1a. Label Schema Design

**Decision: Multi-dimensional labeling with a primary binary judgment.**

We use a two-layer label scheme:

**Layer 1 -- Primary Judgment (binary)**:
- `PRODUCTION_SAFE`: The target response could replace the source response in production without user-visible degradation.
- `NOT_PRODUCTION_SAFE`: The target response has a deficiency that would be noticed by end users or break downstream systems.

This is the label we calibrate thresholds against. It maps directly to the recommendation engine's purpose: should we GO or not?

**Layer 2 -- Diagnostic Dimensions (1-5 Likert each)**:
- `semantic_equivalence`: Do both responses convey the same meaning? (1=completely different meaning, 5=identical meaning)
- `format_equivalence`: Do both responses follow the same structure/format? (1=completely different format, 5=identical format)
- `functional_equivalence`: Would both responses produce the same downstream behavior in an application? (1=would break downstream, 5=functionally interchangeable)

**Why this scheme and not alternatives:**

- **Binary only** (Label A) is insufficient. A binary label tells us whether a pair passes or fails, but not *why* it fails. Without dimensional labels, we cannot diagnose whether the composite score or a specific metric (BERTScore, JSON field match, etc.) is the calibration bottleneck.

- **Likert only** (Label B) has worse inter-rater reliability than binary judgments. Likert scales for subjective quality are noisy -- one labeler's "3" is another's "4". But binary "would this break production?" is a more concrete judgment that people agree on.

- **Multi-dimensional only** (Label C) without a primary binary loses the direct connection to the recommendation output. We need a clean binary for ROC analysis.

The two-layer scheme gives us the best of both worlds: a reliable binary for threshold calibration (Layer 1) and diagnostic dimensions for understanding *what the metrics are measuring vs. what matters to humans* (Layer 2).

**Data structure:**

```python
# src/rosettastone/calibration/types.py

from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, Field
from rosettastone.core.types import OutputType


class ProductionSafety(StrEnum):
    SAFE = "PRODUCTION_SAFE"
    UNSAFE = "NOT_PRODUCTION_SAFE"


class DimensionalScores(BaseModel):
    """Diagnostic Likert scores (1-5) for understanding failure modes."""
    semantic_equivalence: int = Field(ge=1, le=5)
    format_equivalence: int = Field(ge=1, le=5)
    functional_equivalence: int = Field(ge=1, le=5)


class HumanLabel(BaseModel):
    """A single labeler's annotation for one pair."""
    labeler_id: str
    production_safety: ProductionSafety
    dimensional_scores: DimensionalScores
    confidence: int = Field(ge=1, le=3, description="Labeler's self-assessed confidence: 1=unsure, 2=fairly sure, 3=certain")
    notes: str = ""


class LabeledPair(BaseModel):
    """A response pair with its evaluation scores and human labels."""
    pair_id: str  # deterministic hash of (prompt, source_response, target_response)
    prompt: str
    source_response: str
    target_response: str
    source_model: str
    target_model: str
    output_type: OutputType

    # RosettaStone's automated scores for this pair
    composite_score: float
    metric_scores: dict[str, float]  # e.g. {"bertscore_f1": 0.87, "embedding_sim": 0.92}

    # Human labels (one per labeler; minimum 3)
    labels: list[HumanLabel]

    # Adjudicated ground truth (set after inter-rater analysis)
    adjudicated_safety: ProductionSafety | None = None


class CalibrationDataset(BaseModel):
    """The complete calibration dataset."""
    version: str  # semver, e.g. "1.0.0"
    created_at: str  # ISO 8601
    description: str
    pairs: list[LabeledPair]

    def pairs_by_output_type(self, ot: OutputType) -> list[LabeledPair]:
        return [p for p in self.pairs if p.output_type == ot]
```

---

### 1b. Dataset Composition

**Target: 600 labeled pairs total.**

The number 600 is not arbitrary. Here is the reasoning:

1. **Per-type ROC analysis** requires enough samples at every score range to trace a meaningful curve. With 4 output types, we need at least 100 pairs per type to have ~10 pairs in each 0.1-wide score bucket across [0.0, 1.0]. This gives us 400 minimum.

2. **Inter-rater reliability** measurement (Krippendorff's alpha) requires at least 50 items to produce a stable alpha estimate. We will exceed this per type.

3. **Adversarial and edge cases** need explicit representation (they will not appear organically). We allocate 200 pairs specifically for boundary and failure cases.

4. **Statistical power for threshold selection**: to distinguish between threshold 0.80 and 0.85 with 80% power at alpha=0.05, we need ~150 pairs per type in the critical score range (0.65-0.95). The 150/type allocation satisfies this.

**Allocation:**

| Output Type | Normal Pairs | Adversarial/Edge | Total |
|-------------|-------------|-----------------|-------|
| `json` | 100 | 50 | 150 |
| `classification` | 100 | 50 | 150 |
| `short_text` | 100 | 50 | 150 |
| `long_text` | 100 | 50 | 150 |
| **Total** | **400** | **200** | **600** |

**Score distribution requirement**: Pairs must NOT cluster around one score range. The dataset must span the full [0.0, 1.0] composite score distribution roughly uniformly. Without this, the ROC analysis has blind spots. The collection pipeline (1c) is designed to produce this distribution intentionally.

**Normal pairs by model family** (100 per output type, distributed as):

| Model Pair Direction | Count per Type | Rationale |
|---------------------|---------------|-----------|
| GPT-4o -> Claude Sonnet | 25 | Primary commercial migration path |
| Claude Sonnet -> GPT-4o | 15 | Reverse path for coverage |
| Ollama qwen3:8b -> qwen3.5:4b | 30 | Free to generate, highest volume |
| qwen3.5:4b -> qwen3:8b | 15 | Upgrade path (should mostly pass) |
| Same model -> same model | 15 | Positive control (should be near-perfect scores) |

**Adversarial/edge cases** (50 per output type, designed to stress specific failure modes):

| Case Type | Count per Type | Description |
|-----------|---------------|-------------|
| Near-threshold | 15 | Pairs with composite scores in [threshold-0.10, threshold+0.10] |
| Format divergence | 10 | Semantically equivalent but structurally different (e.g., JSON with reordered keys, bulleted list vs. paragraph) |
| Capability gap | 10 | Source model can do it, target model fails (code, math, multilingual) |
| Empty/error responses | 5 | Target returned empty string, error message, or refusal |
| Length mismatch | 5 | Correct content but dramatically different verbosity |
| Subtle semantic drift | 5 | Plausible but factually incorrect target response |

---

### 1c. Pair Collection Pipeline

The collection pipeline has three stages: generate, score, select.

**Stage 1: Generate raw pairs**

We need (prompt, source_response, target_response) triples. There are three sources:

**Source A -- Existing sample data through RosettaStone pipeline (primary)**:
Run actual RosettaStone migrations using the existing Ollama models and the sample datasets. This produces both baseline (pre-optimization) and validation (post-optimization) response pairs. Both are valuable -- pre-optimization pairs tend to score lower, providing coverage at the low end of the score distribution.

```bash
# Generate pre- and post-optimization pairs for local models
uv run rosettastone migrate \
  --data examples/sample_data.jsonl \
  --from ollama/qwen3:8b \
  --to ollama/qwen3.5:4b \
  --output-dir calibration/raw/qwen_to_qwen35

# The validation_results in the output contain scored pairs
```

**Source B -- Synthetic pair generation for specific score ranges**:
To fill gaps in the score distribution (especially the low-score region 0.0-0.4, which real migrations rarely produce), we synthetically degrade responses:

```python
# scripts/generate_calibration_pairs.py

"""Generate response pairs spanning the full score distribution.

Strategies for synthetic degradation:
1. Truncation: cut the response at 25%, 50%, 75%
2. Paraphrase drift: ask a smaller model to "rephrase but change some details"
3. Format corruption: strip JSON formatting, change delimiters
4. Wrong answer: ask a model for the answer to a DIFFERENT question
5. Partial correctness: replace named entities with wrong values
"""
```

**Source C -- Commercial API pairs (when API keys available)**:
For the 40 pairs per type allocated to GPT-4o <-> Claude paths, we run actual commercial API calls. These are expensive ($0.01-0.05 per pair) but necessary for calibrating the thresholds on the model families users actually care about. Total cost: ~$20-40 for 160 commercial pairs.

**Stage 2: Score all pairs with RosettaStone evaluators**

Every generated pair is run through `CompositeEvaluator._score()` to produce the full metric vector (bertscore_f1, embedding_sim, json_valid, json_field_match, exact_match, string_similarity, llm_judge_score). The composite score is computed. This is the score the thresholds will operate on.

```python
# scripts/score_calibration_pairs.py

from rosettastone.evaluate.composite import CompositeEvaluator
from rosettastone.config import MigrationConfig

config = MigrationConfig(
    source_model="ollama/qwen3:8b",
    target_model="ollama/qwen3.5:4b",
    local_only=True,
)
evaluator = CompositeEvaluator(config)

# Score each pair and record all metrics
for pair in raw_pairs:
    scores = evaluator._score(
        expected=pair.source_response,
        actual=pair.target_response,
        output_type=pair.output_type,
    )
    composite = evaluator._composite_score(scores, pair.output_type)
    pair.metric_scores = scores
    pair.composite_score = composite
```

**Stage 3: Select pairs for labeling (stratified sampling)**

From the full corpus of scored pairs, select 600 pairs using stratified sampling to ensure uniform coverage of the score range:

```python
# For each output type (150 pairs each):
#   - Divide [0.0, 1.0] into 10 buckets of width 0.1
#   - Sample 15 pairs per bucket (10 buckets x 15 = 150)
#   - If a bucket has fewer than 15 pairs, oversample from adjacent buckets
#   - Ensure adversarial/edge case quotas are met
```

This stratification is critical. Without it, most pairs would have scores in the 0.7-0.95 range (the "boring" region where most migrations land), and we would have almost no pairs in the 0.0-0.5 range where the most important threshold decisions happen.

**Collection pipeline script:**

```python
# scripts/build_calibration_dataset.py

"""End-to-end calibration dataset builder.

Usage:
    uv run python scripts/build_calibration_dataset.py \
        --output calibration/dataset_v1.json \
        --ollama-pairs 400 \
        --synthetic-pairs 200 \
        --commercial-pairs 0  # set to 160 when API keys available
"""
```

The script outputs a JSON file conforming to the `CalibrationDataset` schema, with `labels: []` (empty, ready for human annotation).

---

### 1d. Labeling Methodology

#### Labeler Pool

**Minimum: 3 independent labelers per pair.** This is the standard minimum for computing inter-rater reliability with Krippendorff's alpha.

**Who labels:**

| Labeler Type | Count | Role |
|-------------|-------|------|
| Engineers familiar with LLM behavior | 2 | Primary labelers; understand what "production-safe" means for API responses |
| Domain-naive labeler (QA or product) | 1 | Control for expertise bias; ensures labels are not over-technical |

All three labelers see identical instructions and interface. They do not see each other's labels until adjudication.

#### Labeling Interface

A simple web-based annotation tool (or spreadsheet for MVP) that shows:

1. The original prompt
2. The source model's response (labeled "Current Production Response")
3. The target model's response (labeled "Proposed Replacement")
4. The output type (shown as context, not editable)
5. Input fields for the label schema

The interface does NOT show: composite scores, metric scores, model names, or any RosettaStone evaluation output. Labelers must judge equivalence blind to the automated assessment.

#### Labeling Rubric (Complete)

---

**ROSETTASTONE CALIBRATION LABELING RUBRIC v1.0**

**Your task**: You are evaluating whether a new LLM response ("Proposed Replacement") could safely replace the current production response ("Current Production Response") without users or downstream systems noticing a degradation.

**Context**: An automated system is migrating from one LLM to another. For each prompt, you see the response from the old model and the response from the new model. Your job is to judge whether the new response is good enough to ship.

---

**STEP 1: Primary Judgment**

Answer this question: *"If I were the engineer responsible for this system, would I be comfortable deploying the Proposed Replacement to production?"*

- **PRODUCTION_SAFE**: Yes. The replacement is equivalent or better. Users would not notice a difference, and downstream systems would not break. Minor stylistic differences (word choice, sentence structure, capitalization) do NOT make a response unsafe unless they change meaning or break a parser.

- **NOT_PRODUCTION_SAFE**: No. There is a concrete problem:
  - The replacement gets a fact wrong that the original got right
  - The replacement is missing critical information present in the original
  - The replacement breaks a structural format that downstream code would parse (e.g., invalid JSON, missing required fields, wrong delimiter)
  - The replacement refuses to answer when the original did answer
  - The replacement is empty or an error message
  - The replacement introduces harmful, biased, or inappropriate content not present in the original

**Key principle**: *"Different but equally good" is PRODUCTION_SAFE. "Different and worse in a way that matters" is NOT_PRODUCTION_SAFE.*

---

**STEP 2: Diagnostic Dimensions** (rate each 1-5)

**Semantic Equivalence** -- Do both responses convey the same information/meaning?

| Score | Description | Example |
|-------|------------|---------|
| 1 | Completely different meaning or topic | Q: "Capital of France?" A1: "Paris" A2: "The Eiffel Tower is 330m tall" |
| 2 | Same topic but substantially wrong | Q: "Capital of France?" A1: "Paris" A2: "Lyon" |
| 3 | Partially correct, missing key information | Q: "Three states of matter?" A1: "Solid, liquid, gas" A2: "Solid and liquid" |
| 4 | Same meaning, minor differences in detail or emphasis | A1: "37.78 degrees C" A2: "approximately 37.8 C" |
| 5 | Identical meaning, possibly different wording | A1: "The capital of France is Paris." A2: "Paris is France's capital." |

**Format Equivalence** -- Do both responses follow the same structure?

| Score | Description | Example |
|-------|------------|---------|
| 1 | Completely different format | A1: `{"name": "Alice"}` A2: `Name: Alice` |
| 2 | Same general format, major structural differences | A1: `{"name": "Alice", "age": 30}` A2: `{"fullName": "Alice"}` |
| 3 | Same format, moderate differences (extra/missing fields, different nesting) | A1: `{"name": "Alice"}` A2: `{"name": "Alice", "extra": true}` |
| 4 | Same format, minor differences (whitespace, key ordering, quoting) | A1: `{"name":"Alice"}` A2: `{ "name": "Alice" }` |
| 5 | Identical format | A1: `{"name": "Alice"}` A2: `{"name": "Alice"}` |

**Functional Equivalence** -- Would both responses produce the same behavior in a downstream system?

| Score | Description | Example |
|-------|------------|---------|
| 1 | Would cause a crash or completely wrong downstream behavior | A1: `{"status": "ok"}` A2: `Not valid JSON` |
| 2 | Would produce wrong results but not crash | A1: `"Negative"` A2: `"Neutral"` (sentiment classifier) |
| 3 | Would produce degraded but partially acceptable results | Correct answer but wrapped in extra text a parser might struggle with |
| 4 | Would produce identical results in most contexts, edge cases might differ | A1: `"37.78"` A2: `"37.8"` (depends on required precision) |
| 5 | Would produce identical downstream behavior in all contexts | Responses are interchangeable for any consumer |

---

**STEP 3: Confidence**

Rate your confidence in your primary judgment:
- 1 = Unsure (could go either way; I would want to discuss this one)
- 2 = Fairly sure (I can see why someone might disagree, but I think my judgment is right)
- 3 = Certain (this is clearly safe / clearly unsafe, no ambiguity)

---

**STEP 4: Notes** (optional)

If anything about this pair is unusual, note it. Examples: "The prompt is ambiguous so both answers are valid", "The replacement is actually better than the original", "This requires domain expertise I don't have."

---

**SPECIAL CASES**

- **Both responses are wrong**: If the original production response is also wrong, the replacement being equally wrong is PRODUCTION_SAFE (the migration did not make things worse).
- **Replacement is strictly better**: Mark as PRODUCTION_SAFE with a note.
- **Ambiguous prompts**: If the prompt is so vague that many valid responses exist, judge whether the replacement is within the same space of reasonable answers.
- **Code output**: For code, PRODUCTION_SAFE requires functional equivalence (same behavior when executed), not textual equivalence. Different variable names, formatting, or approaches are fine if they compute the same result.

---

#### Inter-Rater Agreement

**Metric: Krippendorff's alpha for the primary binary judgment.**

| Alpha Range | Interpretation | Action |
|-------------|---------------|--------|
| >= 0.80 | Good agreement | Proceed to adjudication |
| 0.67 - 0.79 | Acceptable but noisy | Review rubric, discuss disagreement patterns, re-label worst 10% |
| < 0.67 | Unacceptable | Rubric revision needed; hold calibration until agreement improves |

We also compute Cohen's kappa for each labeler pair to identify if one labeler is an outlier.

For the Likert dimensions, we use Krippendorff's alpha with ordinal distance metric. These do not need to be as high (>= 0.60 is acceptable for diagnostic dimensions).

#### Disagreement Resolution

For pairs where labelers disagree on the primary judgment:

1. **2-of-3 majority**: If two labelers agree, their consensus is the adjudicated label. The disagreement and dissenting label are preserved in metadata.

2. **3-way split** (theoretically impossible with binary labels and 3 labelers, but can happen if a labeler marks "unsure" via confidence=1): Escalate to a fourth labeler who sees all three sets of notes but not the labels. Their judgment breaks the tie.

3. **Systematic disagreements**: If the same pair types cause repeated disagreement (e.g., "is a minor numerical rounding difference SAFE?"), add a rubric amendment with an explicit ruling and re-label the disputed pairs.

#### LLM-as-Labeler (Augmentation)

**Yes, we use LLM labeling as a supplement, not a replacement.**

**Protocol:**

1. Label all 600 pairs with 3 human labelers (ground truth).
2. Also label all 600 pairs with Claude Sonnet 4 and GPT-4o as automated labelers (using the same rubric, presented as a system prompt).
3. Compute agreement between LLM labels and adjudicated human labels.
4. If LLM agreement with human ground truth is >= 0.85 kappa:
   - Future dataset expansions (v2, v3) can use LLM labeling for 80% of pairs, with human labels on a 20% validation subset.
   - This reduces the cost of growing the dataset from ~$3/pair (human) to ~$0.02/pair (LLM).
5. If LLM agreement < 0.85 kappa:
   - LLM labels are used only as a pre-screening triage (flag likely-UNSAFE pairs for prioritized human review) but not as ground truth.

**LLM labeling prompt:**

```
You are an expert evaluator for an LLM migration tool. You will be shown an original prompt, the current production response, and a proposed replacement response.

[INSERT FULL RUBRIC FROM ABOVE]

Return your evaluation as JSON:
{
  "production_safety": "PRODUCTION_SAFE" or "NOT_PRODUCTION_SAFE",
  "semantic_equivalence": 1-5,
  "format_equivalence": 1-5,
  "functional_equivalence": 1-5,
  "confidence": 1-3,
  "notes": "..."
}
```

---

### 1e. Calibration Methodology

Once we have the labeled dataset with adjudicated ground truth (`PRODUCTION_SAFE` / `NOT_PRODUCTION_SAFE`), we calibrate thresholds as follows.

#### Step 1: Per-Type ROC Analysis

For each output type independently:

1. Extract all (composite_score, adjudicated_safety) pairs.
2. Compute the ROC curve: at every possible threshold t in [0.0, 1.0] with step 0.01, compute:
   - **True Positive Rate (TPR)** = P(score >= t | SAFE) -- correctly identified safe migrations
   - **False Positive Rate (FPR)** = P(score >= t | UNSAFE) -- dangerously approved unsafe migrations
   - **Precision** = P(SAFE | score >= t) -- of the migrations we approve, how many are actually safe
   - **Recall** = P(score >= t | SAFE) = TPR
3. Plot the ROC curve and compute AUC. If AUC < 0.70 for any output type, the composite score is not a useful discriminator for that type, and we need to revisit the metric weights or add new metrics.

#### Step 2: Operating Point Selection

The operating point is chosen based on the cost asymmetry of errors:

| Error Type | Description | Business Cost |
|-----------|-------------|---------------|
| **False Positive (FP)** | System says GO but migration is actually unsafe | HIGH -- broken production responses, user-facing degradation, potential revenue impact |
| **False Negative (FN)** | System says NO_GO but migration was actually fine | MEDIUM -- wasted engineering time, delayed cost savings from migration |

Because false positives are more costly than false negatives, we optimize for **high precision at acceptable recall**, not for maximum F1.

**Target operating points per type:**

| Output Type | Max False Positive Rate | Min Recall | Rationale |
|-------------|------------------------|-----------|-----------|
| `json` | 2% | 70% | JSON is parsed by downstream code; a bad JSON migration causes hard crashes. Very low FP tolerance. |
| `classification` | 5% | 75% | Classifications drive routing/logic; moderate FP tolerance. |
| `short_text` | 8% | 80% | User-facing but not parsed; humans are more forgiving of stylistic variation. |
| `long_text` | 10% | 80% | Highest tolerance for variation; long text naturally varies more. |

**Algorithm:**

```python
def find_optimal_threshold(
    scores: list[float],
    labels: list[bool],  # True = PRODUCTION_SAFE
    max_fpr: float,
    min_recall: float,
) -> float:
    """Find the highest threshold that satisfies both constraints.

    Scans thresholds from 1.0 down to 0.0. Returns the first threshold
    where FPR <= max_fpr AND recall >= min_recall. If no threshold
    satisfies both, returns the threshold that minimizes
    (fpr - max_fpr)^2 + (min_recall - recall)^2.
    """
    best_threshold = 0.5  # fallback
    best_distance = float("inf")

    for t_int in range(100, -1, -1):
        t = t_int / 100.0
        tp = sum(1 for s, l in zip(scores, labels) if s >= t and l)
        fp = sum(1 for s, l in zip(scores, labels) if s >= t and not l)
        fn = sum(1 for s, l in zip(scores, labels) if s < t and l)
        tn = sum(1 for s, l in zip(scores, labels) if s < t and not l)

        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)

        if fpr <= max_fpr and recall >= min_recall:
            return t  # highest threshold satisfying both constraints

        distance = max(0, fpr - max_fpr)**2 + max(0, min_recall - recall)**2
        if distance < best_distance:
            best_distance = distance
            best_threshold = t

    return best_threshold
```

#### Step 3: Confidence Calibration

Beyond the threshold, we also calibrate the **confidence interval interpretation**. The current system uses the Wilson CI lower bound as the decision criterion: if CI_lower < threshold, the result is CONDITIONAL even if the point estimate passes. We validate that this approach correctly handles small-sample uncertainty by:

1. Bootstrapping: for each output type, randomly sample N pairs from the labeled dataset (N = 10, 20, 30, 50) and compute the recommendation. Compare against the "full dataset" recommendation.
2. If the CI approach correctly downgrades uncertain small-sample results to CONDITIONAL at least 90% of the time, the approach is validated.
3. If not, we adjust the z-score in `wilson_interval()` (currently 1.96 for 95% CI) or adjust `MIN_RELIABLE_SAMPLES` (currently 30).

#### Step 4: Metric Weight Validation

The labeled data also lets us validate the metric weights in `METRIC_WEIGHTS`:

```python
METRIC_WEIGHTS = {
    "bertscore_f1": 0.5,
    "embedding_sim": 0.5,
    "llm_judge_score": 0.3,
    # ...
}
```

For each output type, we fit a logistic regression:

```
P(SAFE) = sigmoid(w1 * bertscore_f1 + w2 * embedding_sim + w3 * llm_judge + ...)
```

The fitted weights tell us which metrics are actually predictive of human safety judgments. If the fitted weights diverge significantly from the hardcoded weights, we update `METRIC_WEIGHTS`. This is a one-time offline analysis, not a runtime operation.

#### Step 5: Ongoing Recalibration

Thresholds are not set once and forgotten:

1. **Every 6 months** or after **100 new production migrations**: add new labeled pairs from actual production migrations and re-run the calibration analysis.
2. **On new model family**: when a new model family is added (e.g., Gemini, Mistral), collect 50 pairs per output type for that family and verify the existing thresholds still hold.
3. **Threshold drift alert**: track the false positive rate on new data. If it exceeds 2x the target FPR for any type across 20+ consecutive pairs, emit a calibration-needed warning in migration reports.

---

### 1f. Integration with RosettaStone

#### Storage

The calibration dataset is stored in the repository at `calibration/datasets/`:

```
calibration/
  datasets/
    v1.json              # CalibrationDataset, checked into git
    v1_raw_pairs/        # .gitignored, regenerable from scripts
  results/
    v1_roc_analysis.json # Threshold analysis results
    v1_thresholds.json   # Computed thresholds
  scripts/
    generate_pairs.py    # Stage 1: generate raw pairs
    score_pairs.py       # Stage 2: score with evaluators
    select_pairs.py      # Stage 3: stratified selection
    run_calibration.py   # Threshold computation from labeled data
    label_with_llm.py    # LLM-as-labeler augmentation
```

The `v1.json` dataset file (~600 pairs, ~2MB) is checked into git. This is small enough to version-control and ensures reproducibility. Raw intermediate files are `.gitignored`.

#### CalibrationDataset Loader

```python
# src/rosettastone/calibration/loader.py

from __future__ import annotations
import json
from importlib.resources import files
from pathlib import Path
from rosettastone.calibration.types import CalibrationDataset


_DATASET_DIR = Path(__file__).parent.parent.parent.parent / "calibration" / "datasets"


def load_calibration_dataset(version: str = "v1") -> CalibrationDataset:
    """Load a calibration dataset by version."""
    path = _DATASET_DIR / f"{version}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration dataset {version} not found at {path}. "
            f"Run 'uv run python calibration/scripts/generate_pairs.py' to create it."
        )
    data = json.loads(path.read_text())
    return CalibrationDataset.model_validate(data)
```

#### Calibration CLI Command

```python
# New subcommand: uv run rosettastone calibrate

@app.command()
def calibrate(
    dataset: str = typer.Option("v1", help="Calibration dataset version"),
    output: Path = typer.Option("calibration/results", help="Output directory"),
    max_fpr_json: float = typer.Option(0.02, help="Max FPR for JSON type"),
    max_fpr_classification: float = typer.Option(0.05, help="Max FPR for classification type"),
    max_fpr_short_text: float = typer.Option(0.08, help="Max FPR for short_text type"),
    max_fpr_long_text: float = typer.Option(0.10, help="Max FPR for long_text type"),
):
    """Compute calibrated thresholds from a labeled dataset."""
    # 1. Load dataset
    # 2. Run ROC analysis per type
    # 3. Find optimal thresholds
    # 4. Write results to output/thresholds.json
    # 5. Print comparison: old thresholds vs. new thresholds
```

#### Threshold Application

New thresholds are NOT auto-applied. The `calibrate` command outputs a `thresholds.json` that an engineer reviews and then updates `DEFAULT_THRESHOLDS` in `recommendation.py` and `DEFAULT_WIN_THRESHOLDS` in `composite.py`. This is intentional -- threshold changes affect every migration and should be reviewed.

The `MigrationConfig.win_thresholds` field already supports per-run overrides, so users who want to use custom thresholds can do so without modifying the defaults.

#### Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| `src/rosettastone/calibration/__init__.py` | CREATE | Package init |
| `src/rosettastone/calibration/types.py` | CREATE | `HumanLabel`, `LabeledPair`, `CalibrationDataset` Pydantic models |
| `src/rosettastone/calibration/loader.py` | CREATE | Dataset loading utility |
| `src/rosettastone/calibration/threshold_optimizer.py` | CREATE | ROC analysis, threshold computation |
| `calibration/datasets/.gitkeep` | CREATE | Placeholder for dataset storage |
| `calibration/scripts/generate_pairs.py` | CREATE | Raw pair generation |
| `calibration/scripts/score_pairs.py` | CREATE | Score pairs with evaluators |
| `calibration/scripts/select_pairs.py` | CREATE | Stratified pair selection |
| `calibration/scripts/run_calibration.py` | CREATE | End-to-end calibration pipeline |
| `calibration/scripts/label_with_llm.py` | CREATE | LLM-as-labeler automation |
| `src/rosettastone/cli.py` | MODIFY | Add `calibrate` subcommand |
| `.gitignore` | MODIFY | Add `calibration/datasets/*_raw_pairs/` |

#### Tests Required

| Test File | Tests |
|-----------|-------|
| `tests/test_calibration/test_types.py` | Validate Pydantic models serialize/deserialize correctly; validate field constraints (Likert 1-5, confidence 1-3) |
| `tests/test_calibration/test_threshold_optimizer.py` | Test `find_optimal_threshold` with synthetic data: known-good threshold recovery; edge case with all-SAFE or all-UNSAFE labels; verify FPR constraint is respected |
| `tests/test_calibration/test_loader.py` | Test load with valid dataset; test FileNotFoundError on missing version; test round-trip (write then load) |

---

## Part 2: Multi-Run Evaluation

### Problem

LLM responses are non-deterministic. A single evaluation of each pair gives a point estimate of quality that may be misleading. A pair that scores 0.92 on one run might score 0.78 on the next because the target model generated a different response. The current pipeline evaluates each pair exactly once (`CompositeEvaluator.evaluate()` calls `litellm.completion()` once per pair).

### Design

#### Configuration

Add to `MigrationConfig`:

```python
# In src/rosettastone/config.py

class MigrationConfig(BaseModel):
    # ... existing fields ...

    # Multi-run evaluation
    eval_runs: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of evaluation runs per pair. Higher values improve reliability but increase cost linearly.",
    )
    eval_aggregation: Literal["mean", "median", "p25"] = Field(
        default="median",
        description="How to aggregate scores across runs. 'median' is robust to outliers. 'p25' is conservative (pessimistic).",
    )
    variance_flag_threshold: float = Field(
        default=0.15,
        description="Standard deviation threshold above which a prompt is flagged as non-deterministic.",
    )
```

**Default is 1 run** (backward compatible -- no behavior change unless explicitly configured). Recommended production setting is 3 runs.

#### Data Structures

Add to `EvalResult`:

```python
# In src/rosettastone/core/types.py

class EvalResult(BaseModel):
    # ... existing fields ...

    # Multi-run fields (populated when eval_runs > 1)
    run_scores: list[float] = []           # composite_score from each run
    run_metric_scores: list[dict[str, float]] = []  # per-metric scores from each run
    score_std: float = 0.0                  # standard deviation across runs
    is_non_deterministic: bool = False      # flagged if std > variance_flag_threshold
```

Add to `MigrationResult`:

```python
class MigrationResult(BaseModel):
    # ... existing fields ...

    non_deterministic_count: int = 0  # number of pairs flagged as non-deterministic
    eval_runs: int = 1                # how many runs were performed
```

#### Algorithm

Modify `CompositeEvaluator.evaluate()`:

```python
def evaluate(self, test_set, optimized_prompt=None):
    results = []
    eval_runs = self.config.eval_runs if hasattr(self.config, 'eval_runs') else 1

    for pair in test_set:
        if eval_runs == 1:
            # Current behavior, unchanged
            result = self._evaluate_single(pair, optimized_prompt)
            results.append(result)
        else:
            # Multi-run: evaluate N times, aggregate
            run_results = []
            for _ in range(eval_runs):
                run_result = self._evaluate_single(pair, optimized_prompt)
                if run_result is not None:
                    run_results.append(run_result)

            if not run_results:
                continue

            # Aggregate
            aggregated = self._aggregate_runs(run_results, pair)
            results.append(aggregated)

    return results

def _aggregate_runs(self, run_results, pair):
    """Aggregate multiple evaluation runs into a single EvalResult."""
    import statistics

    run_scores = [r.composite_score for r in run_results]
    run_metric_scores = [r.scores for r in run_results]

    # Aggregate composite score
    if self.config.eval_aggregation == "mean":
        composite = statistics.mean(run_scores)
    elif self.config.eval_aggregation == "p25":
        sorted_scores = sorted(run_scores)
        idx = max(0, int(len(sorted_scores) * 0.25))
        composite = sorted_scores[idx]
    else:  # median (default)
        composite = statistics.median(run_scores)

    std = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
    threshold = self._get_threshold(pair.output_type or detect_output_type(pair.response))

    return EvalResult(
        prompt_pair=pair,
        new_response=run_results[0].new_response,  # use first run's response
        scores=run_results[0].scores,  # use first run's detailed metrics
        composite_score=composite,
        is_win=composite >= threshold,
        details=run_results[0].details,
        run_scores=run_scores,
        run_metric_scores=run_metric_scores,
        score_std=std,
        is_non_deterministic=std > self.config.variance_flag_threshold,
    )
```

#### Parallelization

Multi-run evaluation multiplies LLM calls by N. To mitigate latency:

1. **Within-pair parallelism**: Use `concurrent.futures.ThreadPoolExecutor` to run N evaluations of the same pair concurrently. LiteLLM handles connection pooling.
2. **Max concurrency**: Cap at `config.num_threads * eval_runs` total concurrent calls to avoid rate limiting.
3. **Cost impact**: With `eval_runs=3` on a 50-pair test set, cost triples from ~$0.50 to ~$1.50 for commercial APIs. This is acceptable and visible in the preflight cost estimate.

#### Report Changes

Add a "Non-Deterministic Prompts" section to the markdown and HTML reports:

```
## Non-Deterministic Prompts

{{ non_deterministic_count }} of {{ total_test_cases }} prompts showed high variance
across {{ eval_runs }} evaluation runs (std > {{ variance_flag_threshold }}).

These prompts may produce inconsistent results in production and warrant human review.

| # | Composite Score | Std Dev | Run Scores |
|---|----------------|---------|------------|
{% for r in flagged_non_deterministic %}
| {{ loop.index }} | {{ r.composite_score }} | {{ r.score_std }} | {{ r.run_scores }} |
{% endfor %}
```

#### Files to Modify

| File | Change |
|------|--------|
| `src/rosettastone/config.py` | Add `eval_runs`, `eval_aggregation`, `variance_flag_threshold` fields |
| `src/rosettastone/core/types.py` | Add `run_scores`, `run_metric_scores`, `score_std`, `is_non_deterministic` to `EvalResult`; add `non_deterministic_count`, `eval_runs` to `MigrationResult` |
| `src/rosettastone/evaluate/composite.py` | Refactor `evaluate()` to support multi-run; add `_evaluate_single()` and `_aggregate_runs()` |
| `src/rosettastone/core/pipeline.py` | Pass through `eval_runs` in `build_result()` |
| `src/rosettastone/report/templates/report.md.jinja` | Add non-deterministic prompts section |
| `src/rosettastone/report/templates/report.html.jinja` | Add non-deterministic prompts section |

#### Tests

| Test | Description |
|------|-------------|
| `tests/test_evaluate/test_multi_run.py` | Mock LiteLLM to return different scores across runs; verify median/mean/p25 aggregation; verify variance flagging at threshold boundary; verify eval_runs=1 produces identical output to current behavior |

---

## Part 3: Per-Prompt Regression Report

### Problem

The current report shows the 5 worst-scoring validation cases (in `report.md.jinja` "Worst Regressions" section), but this is a post-optimization analysis only. It does not compare per-prompt baseline vs. optimized scores to identify which specific prompts **regressed** due to GEPA optimization. A prompt that scored 0.65 in both baseline and validation is not a regression -- it was always bad. A prompt that scored 0.95 baseline but 0.72 post-optimization is a genuine regression that demands attention.

### Design

#### Data Structure

Add a new model for per-prompt comparison:

```python
# src/rosettastone/core/types.py

class PromptRegression(BaseModel):
    """Per-prompt comparison between baseline and optimized scores."""
    prompt_index: int
    output_type: str
    baseline_score: float
    optimized_score: float
    delta: float  # optimized - baseline (negative = regression)
    baseline_is_win: bool
    optimized_is_win: bool
    status: Literal["improved", "stable", "regressed", "at_risk"]
    metric_deltas: dict[str, float] = {}  # per-metric change
```

Add to `MigrationResult`:

```python
class MigrationResult(BaseModel):
    # ... existing fields ...

    prompt_regressions: list[PromptRegression] = []
    regression_count: int = 0
    at_risk_count: int = 0
```

#### Status Classification

```python
def classify_prompt_status(baseline_score: float, optimized_score: float, threshold: float) -> str:
    delta = optimized_score - baseline_score

    if delta >= 0.05:
        return "improved"
    elif delta >= -0.05:
        return "stable"
    elif optimized_score >= threshold:
        return "regressed"  # got worse but still passes
    else:
        return "at_risk"  # got worse AND now fails threshold
```

The `at_risk` flag triggers when a prompt both regressed AND fell below its output type's threshold. This is the most actionable signal: GEPA optimization made this specific prompt worse, and it is now failing.

#### Sorting

Regressions are sorted by `delta` ascending (worst regression first). `at_risk` prompts always sort before `regressed` prompts regardless of delta magnitude.

#### Report Section

Add to the markdown template after "Worst Regressions":

```
## Per-Prompt Regression Analysis

{{ regression_count }} prompt(s) regressed after optimization.
{{ at_risk_count }} prompt(s) are AT RISK (regressed below threshold).

{% if at_risk_prompts %}
### AT RISK Prompts (require human review)

| # | Output Type | Baseline | Optimized | Delta | Status |
|---|------------|----------|-----------|-------|--------|
{% for r in at_risk_prompts %}
| {{ r.prompt_index }} | {{ r.output_type }} | {{ "%.3f"|format(r.baseline_score) }} | {{ "%.3f"|format(r.optimized_score) }} | {{ "%+.3f"|format(r.delta) }} | AT RISK |
{% endfor %}
{% endif %}

{% if regressed_prompts %}
### Regressed Prompts (degraded but still passing)

| # | Output Type | Baseline | Optimized | Delta |
|---|------------|----------|-----------|-------|
{% for r in regressed_prompts[:10] %}
| {{ r.prompt_index }} | {{ r.output_type }} | {{ "%.3f"|format(r.baseline_score) }} | {{ "%.3f"|format(r.optimized_score) }} | {{ "%+.3f"|format(r.delta) }} |
{% endfor %}
{% endif %}
```

#### Connection to TestCaseRecord

The `TestCaseRecord` DB model already has `phase` ("baseline" or "validation") and `composite_score`. To connect per-prompt regressions to the DB:

1. After computing `prompt_regressions`, store the `status` field in `TestCaseRecord.details_json` for each validation-phase record.
2. Add a query endpoint `GET /api/v1/migrations/{id}/regressions` that joins baseline and validation `TestCaseRecord` rows by prompt index and returns the regression list.

#### Files to Modify

| File | Change |
|------|--------|
| `src/rosettastone/core/types.py` | Add `PromptRegression` model; extend `MigrationResult` |
| `src/rosettastone/core/pipeline.py` | Compute regressions in `build_result()` by pairing baseline and validation results |
| `src/rosettastone/report/templates/report.md.jinja` | Add regression analysis section |
| `src/rosettastone/report/templates/report.html.jinja` | Add regression analysis section with color coding |
| `src/rosettastone/server/api/migrations.py` | Add regressions endpoint |

#### Tests

| Test | Description |
|------|-------------|
| `tests/test_core/test_regressions.py` | Test status classification logic; verify sorting (at_risk before regressed); verify delta computation; edge case: baseline and validation have different number of results (skipped pairs) |

---

## Part 4: Actual Cost Tracking

### Problem

`MigrationResult.cost_usd` is always 0.0 because `PipelineContext.costs` is never populated during the pipeline run. The preflight step estimates cost, but actual cost is never recorded. This means the report's "Cost Summary" section is always `$0.0000`, which is misleading and useless.

### Design

#### Where LiteLLM Exposes Cost

LiteLLM exposes cost data in the response object:

```python
response = litellm.completion(model="openai/gpt-4o", messages=messages)

# Cost is available at:
response._hidden_params.get("response_cost")  # float, in USD
# Also available via:
response.usage.prompt_tokens
response.usage.completion_tokens
```

The `response_cost` field is the most reliable -- LiteLLM computes it from its internal pricing table. If unavailable (e.g., for Ollama), we fall back to 0.0.

#### Cost Accumulator

Add a thread-safe cost accumulator to `PipelineContext`:

```python
# src/rosettastone/core/context.py

import threading

@dataclass
class PipelineContext:
    # ... existing fields ...
    _cost_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_cost(self, phase: str, cost: float) -> None:
        """Thread-safe cost accumulation."""
        with self._cost_lock:
            self.costs[phase] = self.costs.get(phase, 0.0) + cost
```

#### Capture Points

Cost is captured at every point where `litellm.completion()` is called:

1. **`CompositeEvaluator.evaluate()`** -- both baseline and validation phases. This is the highest-volume call site (2 x test_set_size calls minimum).
2. **`LLMJudgeEvaluator.score()`** -- 2 calls per pair (bidirectional).
3. **GEPA optimization** -- DSPy manages its own LiteLLM calls internally. We cannot easily intercept individual calls inside DSPy, but we can use LiteLLM's callback mechanism.

**For direct calls (evaluators):**

```python
# In CompositeEvaluator.evaluate(), after litellm.completion():
response = litellm.completion(model=self.config.target_model, messages=messages, **extra_kwargs)
cost = getattr(response, '_hidden_params', {}).get('response_cost', 0.0) or 0.0
if self._ctx is not None:
    self._ctx.add_cost("evaluation", cost)
```

**For DSPy/GEPA calls (callback-based):**

LiteLLM supports a global success callback. We register a callback before GEPA runs and deregister it after:

```python
# In GEPAOptimizer.optimize(), or in the migrator before calling optimize_prompt():

import litellm

accumulated_gepa_cost = 0.0

def _cost_callback(kwargs, completion_response, start_time, end_time):
    nonlocal accumulated_gepa_cost
    cost = getattr(completion_response, '_hidden_params', {}).get('response_cost', 0.0) or 0.0
    accumulated_gepa_cost += cost

litellm.success_callback = [_cost_callback]
try:
    optimized_prompt = optimize_prompt(train, val, config)
finally:
    litellm.success_callback = []

ctx.add_cost("optimization", accumulated_gepa_cost)
```

#### Report Integration

Update the report template to show actual vs. estimated cost:

```
## Cost Summary

| Phase | Actual Cost (USD) |
|-------|-------------------|
{% for phase, cost in costs.items() %}
| {{ phase }} | ${{ "%.4f" | format(cost) }} |
{% endfor %}
| **Total Actual** | **${{ "%.4f" | format(cost_usd) }}** |
{% if estimated_cost_usd %}
| Preflight Estimate | ${{ "%.4f" | format(estimated_cost_usd) }} |
| Estimate Accuracy | {{ "%.0f" | format(cost_usd / max(estimated_cost_usd, 0.0001) * 100) }}% |
{% endif %}
```

#### MigrationResult Changes

```python
class MigrationResult(BaseModel):
    # ... existing fields ...
    cost_breakdown: dict[str, float] = {}  # {"evaluation": 0.12, "optimization": 0.45, ...}
    estimated_cost_usd: float = 0.0        # from preflight, for comparison
```

#### Files to Modify

| File | Change |
|------|--------|
| `src/rosettastone/core/context.py` | Add `_cost_lock` and `add_cost()` method |
| `src/rosettastone/evaluate/composite.py` | Accept `PipelineContext` in constructor; capture cost after each `litellm.completion()` call |
| `src/rosettastone/evaluate/llm_judge.py` | Accept `PipelineContext`; capture cost |
| `src/rosettastone/core/migrator.py` | Register/deregister LiteLLM callback around GEPA; pass ctx to evaluators |
| `src/rosettastone/core/pipeline.py` | Pass ctx through `evaluate_baseline()` and `evaluate_optimized()`; update `build_result()` to include `cost_breakdown` |
| `src/rosettastone/core/types.py` | Add `cost_breakdown`, `estimated_cost_usd` to `MigrationResult` |
| `src/rosettastone/report/templates/report.md.jinja` | Update cost section |
| `src/rosettastone/report/templates/report.html.jinja` | Update cost section with actual vs. estimated |

#### Tests

| Test | Description |
|------|-------------|
| `tests/test_core/test_cost_tracking.py` | Mock `litellm.completion` to return `_hidden_params["response_cost"]`; verify costs accumulate in context; verify thread safety with concurrent adds; verify fallback to 0.0 when cost not available |
| `tests/test_evaluate/test_composite_cost.py` | Verify cost is captured per pair in evaluator; verify cost flows through to `MigrationResult.cost_breakdown` |

---

## Part 5: Shadow Deployment Tooling

### Problem

After RosettaStone recommends GO, the engineer still faces the riskiest step: deploying the new model to production. RosettaStone's evaluation was on a test set; production traffic may differ. Shadow deployment -- running both models in parallel and comparing results on real traffic before switching -- is the standard mitigation.

### Design

#### 5a. Shadow Configuration File

At the end of every migration that results in a GO or CONDITIONAL recommendation, RosettaStone generates a `shadow_config.yaml`:

```yaml
# shadow_config.yaml -- generated by RosettaStone
# Migration: openai/gpt-4o -> anthropic/claude-sonnet-4
# Recommendation: GO (confidence: 92.3%)
# Generated: 2026-04-01T14:30:00Z

shadow:
  source_model: "openai/gpt-4o"
  target_model: "anthropic/claude-sonnet-4"
  optimized_prompt: |
    You are a helpful assistant that provides concise, accurate responses.
    [... full optimized prompt ...]

  # Shadow mode returns the source model's response to the caller.
  # The target model's response is logged for comparison only.
  primary: "source"  # "source" or "target"

  # Sampling rate: what fraction of requests to shadow (1.0 = all)
  sample_rate: 1.0

  # Duration: how long to run shadow mode before making a switch decision
  duration_hours: 72

  # Comparison logging
  log_path: "./shadow_logs/"
  log_format: "jsonl"  # compatible with RosettaStone re-ingestion

  # Rollback
  rollback:
    trigger: "manual"  # "manual" or "auto"
    # Auto-rollback triggers (if trigger: "auto"):
    auto_rollback_if:
      error_rate_exceeds: 0.05      # 5% error rate
      latency_p95_exceeds_ms: 5000  # 5s p95
      score_below: 0.70             # average composite score

  # Model endpoints (for the proxy)
  endpoints:
    source:
      provider: "openai"
      model: "gpt-4o"
      # API key from environment: OPENAI_API_KEY
    target:
      provider: "anthropic"
      model: "claude-sonnet-4"
      # API key from environment: ANTHROPIC_API_KEY
```

**Generation logic**: Added to `generate_report()` in `pipeline.py`. Only generated when recommendation is GO or CONDITIONAL.

#### 5b. Shadow Proxy Script

A lightweight HTTP proxy that sits between the application and the LLM API:

```python
# scripts/shadow_proxy.py

"""Shadow deployment proxy for RosettaStone migrations.

Usage:
    uv run python scripts/shadow_proxy.py --config shadow_config.yaml --port 8080

The proxy:
1. Receives OpenAI-compatible API requests on localhost:8080
2. Forwards to BOTH source and target models concurrently
3. Returns the primary model's response to the caller
4. Logs both responses for comparison

Your application points its LLM API base URL to localhost:8080 instead of
the real API. No application code changes needed beyond the base URL.
"""
```

**Architecture:**

```
Application  -->  Shadow Proxy (localhost:8080)
                    |                    |
                    v                    v
              Source Model          Target Model
              (primary)             (shadow)
                    |                    |
                    v                    v
              Response returned    Response logged
              to application       for comparison
```

**Key implementation details:**

- The proxy is a `uvicorn` + `FastAPI` app with a single `POST /v1/chat/completions` endpoint.
- Source and target calls run concurrently via `asyncio.gather()`. The primary model's response is returned immediately; the shadow model's response is fire-and-forget (does not block the caller).
- If the shadow model call fails, the failure is logged but does not affect the primary response.
- The proxy adds a `X-Shadow-Request-Id` header to both outgoing requests for correlation.

```python
# Simplified core logic:

@app.post("/v1/chat/completions")
async def shadow_completion(request: Request):
    body = await request.json()

    # Run both models concurrently
    source_task = asyncio.create_task(call_model(config.source, body))
    target_task = asyncio.create_task(call_model(config.target, body, optimized_prompt=config.optimized_prompt))

    # Wait for primary (source); don't block on shadow
    source_response = await source_task
    try:
        target_response = await asyncio.wait_for(target_task, timeout=30.0)
    except (asyncio.TimeoutError, Exception) as e:
        target_response = {"error": str(e)}

    # Log comparison
    await log_comparison(body, source_response, target_response)

    # Return primary response to caller
    return source_response
```

#### 5c. Shadow Comparison Log Format

Logs are JSONL, one line per request, compatible with RosettaStone's JSONL ingestion:

```json
{
  "request_id": "shadow-uuid-1234",
  "timestamp": "2026-04-01T14:30:00.123Z",
  "prompt": [{"role": "user", "content": "What is the capital of France?"}],
  "source_model": "openai/gpt-4o",
  "target_model": "anthropic/claude-sonnet-4",
  "source_response": "The capital of France is Paris.",
  "target_response": "Paris is the capital of France.",
  "source_latency_ms": 234,
  "target_latency_ms": 189,
  "source_tokens": {"prompt": 12, "completion": 8},
  "target_tokens": {"prompt": 12, "completion": 7},
  "source_cost_usd": 0.000045,
  "target_cost_usd": 0.000032
}
```

This format has two key properties:
1. It can be directly re-ingested by RosettaStone as a JSONL data file (the `prompt` and `response` fields are present).
2. It includes both source and target responses, enabling post-hoc evaluation without re-calling the APIs.

#### 5d. Shadow-to-Recalibration Feedback Loop

Shadow logs feed back into RosettaStone via:

```bash
# After shadow period, evaluate the shadow logs:
uv run rosettastone evaluate-shadow \
  --shadow-logs ./shadow_logs/ \
  --output ./shadow_report/

# If shadow data is high quality, add to calibration dataset:
uv run python calibration/scripts/add_shadow_data.py \
  --shadow-logs ./shadow_logs/ \
  --dataset calibration/datasets/v1.json \
  --output calibration/datasets/v2.json
```

The `evaluate-shadow` command scores all shadow log pairs using the same evaluators as the migration pipeline and generates a mini-report with win rate, regressions, and a GO/NO_GO recommendation for proceeding with the full cutover.

#### Files to Create

| File | Description |
|------|-------------|
| `scripts/shadow_proxy.py` | Shadow deployment HTTP proxy |
| `scripts/shadow_config_template.yaml` | Template for shadow config |
| `src/rosettastone/shadow/__init__.py` | Shadow tooling package |
| `src/rosettastone/shadow/config.py` | Shadow config Pydantic model |
| `src/rosettastone/shadow/log_format.py` | Log entry model + writer |
| `src/rosettastone/shadow/evaluator.py` | Shadow log evaluation pipeline |
| `src/rosettastone/core/pipeline.py` | MODIFY: generate shadow_config.yaml in `generate_report()` |
| `src/rosettastone/cli.py` | MODIFY: add `evaluate-shadow` subcommand |

#### Tests

| Test | Description |
|------|-------------|
| `tests/test_shadow/test_proxy.py` | Test proxy routes requests to both models; test primary response returned; test shadow failure does not block primary; test log output format |
| `tests/test_shadow/test_log_format.py` | Test JSONL serialization; test re-ingestion compatibility |
| `tests/test_shadow/test_evaluator.py` | Test shadow log evaluation produces valid report |

---

## Part 6: GEPA Timeout

### Problem

GEPA optimization (`dspy.GEPA`) can run indefinitely depending on the auto preset (light=25 iterations, heavy=100) and the LLM response times. With `gepa_auto="heavy"` and a slow model, the optimization step can take hours. There is no timeout, and the only way to stop it is to kill the process, losing all intermediate progress.

### Design

#### Configuration

```python
# In src/rosettastone/config.py

class MigrationConfig(BaseModel):
    # ... existing fields ...

    gepa_timeout_seconds: int = Field(
        default=600,
        ge=30,
        description="Maximum wall-clock time for GEPA optimization in seconds. "
                    "If exceeded, the best intermediate result found so far is used.",
    )
```

Default of 600 seconds (10 minutes) is generous for `gepa_auto="light"` (typically completes in 2-5 minutes) and provides a safety net for heavier presets.

#### Timeout Implementation

DSPy's GEPA optimizer does not expose a native timeout. We wrap it in a thread-based timeout:

```python
# src/rosettastone/optimize/gepa.py

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


class GEPAOptimizer(Optimizer):
    def optimize(self, train_set, val_set, config, on_iteration=None) -> str:
        # ... existing LM setup ...

        # Track best intermediate result via the iteration callback
        best_intermediate: list[str] = []  # mutable container for thread access

        original_callback = on_iteration

        def _tracking_callback(iteration: int, total: int, score: float):
            # DSPy may update the program in-place; we extract instructions
            # after each iteration as a checkpoint
            nonlocal compiled
            try:
                from rosettastone.optimize.utils import extract_optimized_instructions
                instructions = extract_optimized_instructions(compiled)
                if instructions:
                    best_intermediate.clear()
                    best_intermediate.append(instructions)
            except Exception:
                pass
            if original_callback:
                original_callback(iteration, total, score)

        # Run GEPA with timeout
        timeout = getattr(config, 'gepa_timeout_seconds', 600)
        compiled = None
        timed_out = False

        def _run_gepa():
            nonlocal compiled
            with dspy.context(lm=target_lm):
                optimizer = dspy.GEPA(**gepa_kwargs)
                compiled = optimizer.compile(program, trainset=trainset)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_gepa)
            try:
                future.result(timeout=timeout)
            except FuturesTimeoutError:
                timed_out = True
                # Future's thread will eventually finish but we don't wait for it
                # The thread will be cleaned up when the executor shuts down

        if timed_out:
            if best_intermediate:
                # Use best intermediate result
                import logging
                logging.getLogger(__name__).warning(
                    "GEPA optimization timed out after %ds; using best intermediate result",
                    timeout,
                )
                return best_intermediate[0]
            else:
                raise TimeoutError(
                    f"GEPA optimization timed out after {timeout}s with no intermediate results. "
                    f"Try increasing gepa_timeout_seconds or reducing gepa_auto intensity."
                )

        # Normal completion
        from rosettastone.optimize.utils import extract_optimized_instructions
        return extract_optimized_instructions(compiled)
```

**Important caveat about thread cancellation**: Python threads cannot be forcibly killed. The GEPA thread will continue running in the background until it completes or the process exits. The timeout only prevents the *caller* from waiting longer than the specified duration. In practice, this is acceptable because:

1. The GEPA thread will complete its current iteration and then the executor's `__exit__` allows it to finish naturally.
2. The `with dspy.context(lm=target_lm)` scope is thread-local, so it does not interfere with subsequent operations.
3. If process-level cleanup is needed, Python's `atexit` handlers and the GIL ensure orderly shutdown.

A more robust approach would use `multiprocessing` instead of threading, allowing actual process termination. This can be a follow-up improvement if the threading approach causes problems in practice.

#### Warning in Report

When timeout occurs, add a warning to the pipeline context:

```python
if timed_out:
    ctx.warnings.append(
        f"GEPA optimization timed out after {timeout}s. "
        f"The optimized prompt may be suboptimal. "
        f"Consider increasing gepa_timeout_seconds (current: {timeout}) "
        f"or reducing gepa_auto from '{config.gepa_auto}' to 'light'."
    )
```

This warning appears in the report's "Pipeline Warnings" section and is factored into the recommendation reasoning.

#### Fallback Behavior Summary

| Scenario | Behavior |
|----------|----------|
| GEPA completes within timeout | Normal flow, no change |
| GEPA times out, intermediate result exists | Use intermediate, emit warning, continue pipeline |
| GEPA times out, no intermediate result | Raise `TimeoutError`, migration fails with clear error message |

#### Files to Modify

| File | Change |
|------|--------|
| `src/rosettastone/config.py` | Add `gepa_timeout_seconds` field |
| `src/rosettastone/optimize/gepa.py` | Wrap GEPA in ThreadPoolExecutor with timeout; intermediate result tracking |
| `src/rosettastone/core/migrator.py` | Catch `TimeoutError` from optimizer; add warning to context |

#### Tests

| Test | Description |
|------|-------------|
| `tests/test_optimize/test_gepa_timeout.py` | Mock DSPy GEPA to sleep indefinitely; verify timeout fires after configured seconds; verify intermediate result is used when available; verify TimeoutError raised when no intermediate; verify normal completion is unaffected by timeout |

---

## Implementation Priority and Effort Estimates

### Priority Order

| Priority | Part | Rationale | Effort |
|----------|------|-----------|--------|
| **1** | Part 4: Actual Cost Tracking | Lowest effort, highest immediate value. Cost is literally $0.00 in every report today. Unblocks cost-based decisions. No dependencies. | 1-2 days |
| **2** | Part 6: GEPA Timeout | Safety net for runaway optimization. Simple implementation, prevents real-world production incidents. No dependencies. | 1 day |
| **3** | Part 3: Per-Prompt Regression Report | Medium effort, high diagnostic value. Uses only data already available in the pipeline. Makes the existing report dramatically more useful. | 2-3 days |
| **4** | Part 2: Multi-Run Evaluation | Medium effort, important for reliability. Requires careful refactoring of `CompositeEvaluator` but no new subsystems. | 3-4 days |
| **5** | Part 1: Human-Labeled Validation Dataset | Highest total effort but highest long-term value. The data collection takes calendar time (weeks, not days). Start the pair generation scripts early; labeling can happen in parallel with Parts 2-4 implementation. | 3-4 weeks (calendar), ~10 days engineering |
| **6** | Part 5: Shadow Deployment Tooling | Largest scope, least urgent. Only valuable after migrations are actually being deployed. Can be deferred until a real deployment is imminent. | 5-7 days |

### Implementation Phases

**Phase A (Week 1)**: Parts 4 + 6 in parallel. Both are self-contained, low-risk changes.
- Cost tracking: modify evaluators and context.
- GEPA timeout: modify optimizer only.
- Gate: `uv run pytest tests/ -v` passes, cost appears in reports, timeout is exercised in tests.

**Phase B (Weeks 2-3)**: Parts 3 + 2 in parallel.
- Per-prompt regression: new models + pipeline logic + report templates.
- Multi-run evaluation: refactor CompositeEvaluator + new config fields.
- Gate: both features visible in migration reports, multi-run mode tested with Ollama.

**Phase C (Weeks 2-5, overlapping with B)**: Part 1 data collection starts in Week 2.
- Week 2: Generate and score raw pairs (scripts).
- Week 3: Stratified selection, prepare labeling interface.
- Weeks 3-4: Human labeling (calendar time).
- Week 5: Calibration analysis, threshold computation.
- Gate: calibrated thresholds published; ROC curves reviewed; inter-rater alpha >= 0.80.

**Phase D (Week 5-6)**: Part 5 shadow tooling.
- Build proxy, config generator, log evaluator.
- Gate: end-to-end shadow test with Ollama models.

### Total Effort

| Part | Engineering Days |
|------|-----------------|
| Part 1: Calibration Dataset | 10 |
| Part 2: Multi-Run Evaluation | 4 |
| Part 3: Per-Prompt Regression | 3 |
| Part 4: Cost Tracking | 2 |
| Part 5: Shadow Tooling | 6 |
| Part 6: GEPA Timeout | 1 |
| **Total** | **26 engineering days** |

Calendar time: 5-6 weeks, accounting for labeling calendar time and parallel execution of independent parts.

### Dependencies Graph

```
Part 4 (Cost Tracking) -----> no dependencies
Part 6 (GEPA Timeout)  -----> no dependencies
Part 3 (Regressions)   -----> no dependencies
Part 2 (Multi-Run)     -----> no dependencies
Part 1 (Calibration)   -----> Part 2 (uses multi-run for more stable calibration scores)
Part 5 (Shadow)        -----> Part 4 (shadow logs include cost data)
                        -----> Part 1 (shadow data feeds into calibration dataset)
```

Parts 2, 3, 4, and 6 are fully independent and can be implemented by separate engineers simultaneously. Part 1 benefits from Part 2 being complete (multi-run scores are more stable for calibration) but can start pair generation immediately. Part 5 is the only part with meaningful dependencies.
