# RosettaStone — UI Vision

> Designing for decisions, not data. Every screen answers a question a human actually asks.

---

## Design Philosophy

The current brief designs around **data** — metrics, scores, charts, percentiles, confidence intervals. This vision designs around **decisions** — should we switch? is anything about to break? are we spending too much? did the migration work?

Three principles:

**1. Design for day 1, not day 100.** A first-time user with zero migrations should feel welcomed, not overwhelmed. The UI should teach you what it does by guiding you through it, not by showing you an empty cockpit full of instruments you don't understand yet.

**2. Answer the question, then show the evidence.** Lead with the answer ("Safe to switch" / "3 models need attention" / "You could save $400/month"). Put the supporting data behind a click, not in front of the answer.

**3. Two audiences, two experiences — not a toggle on the same page.** Engineers need to debug and drill down. Executives need a document they can forward. The engineer gets an interactive dashboard. The executive gets a shareable artifact (PDF/one-pager), not a stripped-down version of the engineer's dashboard. The "executive view" isn't a view — it's an export.

---

## Information Architecture

The platform has four sections, each answering a core question:

```
┌──────────────────────────────────────────────────────────┐
│  RosettaStone                                            │
│                                                          │
│  [Models]    [Migrations]    [Costs]    [Alerts]         │
│                                                          │
│  Each tab = one question a human asks                    │
└──────────────────────────────────────────────────────────┘
```

| Section | Question it answers | Who uses it |
|---|---|---|
| **Models** | "What are we running and what's out there?" | Both |
| **Migrations** | "Did the migration work? Should we ship it?" | Engineer (primary), Exec (via export) |
| **Costs** | "Are we spending wisely?" | Both |
| **Alerts** | "Is anything about to break?" | Both |

The **Models** page is the default landing page — not Migrations. This is a critical reframe. Most of the time, users aren't in the middle of a migration. They're monitoring their model landscape. The landing page should reflect what they do most often.

---

## Page 1: Models (Landing Page — `/`)

**Question:** "What are we running, and is there anything we should know?"

This page is the heartbeat of the platform. It's where a user comes on a Monday morning to check the health of their model landscape. It's also where a first-time user lands and immediately understands what this tool is about.

### First-Time Experience (Zero State)

When a user first opens RosettaStone with no data, they don't see empty tables and zero-count cards. They see a warm, guided setup:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Welcome to RosettaStone                                 │
│                                                          │
│  Let's set up your model landscape.                      │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Which models are you currently using?             │  │
│  │                                                    │  │
│  │  [openai/gpt-4o          ] [+ Add model]           │  │
│  │  [anthropic/claude-sonnet-4] [+ Add model]         │  │
│  │                                                    │  │
│  │  (We use LiteLLM model identifiers)                │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Or, if you've already run a migration:                  │
│  [Import from migration results]                         │
│                                                          │
│  Or, connect your infrastructure:                        │
│  [Connect Redis cache]  [Connect LangSmith]              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Once models are registered, the page transforms into the model landscape view.

### Active State — Model Landscape

The page has three zones, each answering a sub-question:

**Zone 1: Attention Bar** (only appears when something needs action)

A single, calm banner at the top. Not an alert wall — a prioritized summary.

```
┌──────────────────────────────────────────────────────────┐
│ ⚠  2 things need your attention                         │
│    • gpt-4o-0613 is deprecated — retiring Apr 15        │
│    • Claude Haiku 4.5 costs 40% less than your current  │
│      Haiku — consider upgrading                         │
│                                                     [→]  │
└──────────────────────────────────────────────────────────┘
```

This replaces the "Fleet Status Cards" from the original brief. Instead of showing counts of GO/CONDITIONAL/NO_GO (which are migration-centric and meaningless when you're not migrating), it shows what actually matters: things that need human attention.

**Zone 2: Your Models** (the core of the page)

A clean card grid showing each model you've registered. Each card is a self-contained health check:

```
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│  openai/gpt-4o                  │  │  anthropic/claude-sonnet-4      │
│  ● Active                       │  │  ● Active                       │
│                                 │  │                                 │
│  Context: 128K tokens           │  │  Context: 200K tokens           │
│  Cost: $2.50/1M input           │  │  Cost: $3.00/1M input           │
│  Features: Tools, Vision, JSON  │  │  Features: Tools, JSON          │
│                                 │  │                                 │
│  No issues detected             │  │  No issues detected             │
│                                 │  │                                 │
│  [Run Migration →]              │  │  [Run Migration →]              │
└─────────────────────────────────┘  └─────────────────────────────────┘

┌─────────────────────────────────┐
│  openai/gpt-4o-0613             │
│  ⚠ Deprecated — retiring Apr 15│
│                                 │
│  Context: 128K tokens           │
│  Cost: $2.50/1M input           │
│                                 │
│  Recommended replacement:       │
│  openai/gpt-4o (latest)         │
│                                 │
│  [Start Migration →]            │
└─────────────────────────────────┘
```

Key design decisions:
- Status is human language ("Active", "Deprecated — retiring Apr 15"), not color codes
- Features are listed as capabilities the engineer cares about
- The deprecation card shows the recommended replacement — actionable, not just informational
- "Run Migration" is always one click away
- Cards auto-populate model info from LiteLLM's model registry (context window, pricing, capabilities)

**Zone 3: Model Explorer** (below the fold — discovery, not urgency)

A searchable, filterable table of available models the user isn't currently using. This is the "what's out there?" view. Think of it as a curated model catalog.

```
┌──────────────────────────────────────────────────────────┐
│  Explore Models                                          │
│  [Search models...]   [Provider ▾]  [Capability ▾]      │
│                                                          │
│  Model              Provider   Cost/1M   Context  Match  │
│  ─────────────────────────────────────────────────────── │
│  claude-sonnet-4.6  Anthropic  $3.00     200K     95%   │
│  gemini-2.5-flash   Google     $0.15     1M       88%   │
│  gpt-4o-mini        OpenAI     $0.15     128K     82%   │
│  llama-4-maverick   Meta       $0.10*    128K     79%   │
│                                                          │
│  * Self-hosted pricing estimate                          │
│                                                          │
│  "Match" = capability overlap with your registered       │
│  models. Higher = easier migration.                      │
└──────────────────────────────────────────────────────────┘
```

The "Match" column is unique to RosettaStone — it pre-computes how compatible a candidate model is with your current setup based on capability overlap (tools, vision, JSON mode, context window). This answers the question "how hard would it be to switch to this?" before you even start.

---

## Page 2: Migrations (`/migrations`)

**Question:** "What migrations have I run, and how did they go?"

### Migration List

Clean, scannable table. Not a dashboard with fleet status cards — a list of things you've done, with clear outcomes.

```
┌──────────────────────────────────────────────────────────┐
│  Migrations                            [+ New Migration] │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  gpt-4o → claude-sonnet-4              Safe to ship│  │
│  │  92% confidence · 156 test cases · $2.34 · 2m ago  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  gpt-4o-0613 → gpt-4o               Needs review  │  │
│  │  78% confidence · 43 test cases · $1.12 · 1d ago   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  gpt-3.5-turbo → gpt-4o-mini           Do not ship│  │
│  │  61% confidence · 89 test cases · $0.87 · 3d ago   │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

Key design decisions:
- Three recommendation states only: **Safe to ship** (green), **Needs review** (amber), **Do not ship** (red). NOT "GO/CONDITIONAL/NO_GO" — those are engineer jargon, not human language.
- Each card shows the four things you need at a glance: model pair, recommendation, confidence, metadata
- The whole card is clickable → migration detail page
- Empty state: "No migrations yet. Run `rosettastone migrate` from your terminal, or click '+ New Migration' to start one here."

### Migration Detail (`/migrations/{id}`)

This is where the engineer spends most of their time. The page answers: "Should I ship this, and if not, what's wrong?"

**The Answer First Pattern:**

The page opens with the answer, not the data:

```
┌──────────────────────────────────────────────────────────┐
│  ← Migrations                                           │
│                                                          │
│  gpt-4o → claude-sonnet-4                                │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │                                                    │  │
│  │              ✓ Safe to Ship                        │  │
│  │                                                    │  │
│  │  The target model matches or exceeds the source    │  │
│  │  in 92% of test cases. JSON outputs pass at 100%.  │  │
│  │  Free-text responses show minor tone differences   │  │
│  │  but maintain semantic equivalence.                │  │
│  │                                                    │  │
│  │  [Export Report ▾]  [View Optimized Prompt]        │  │
│  │                                                    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
```

The recommendation reasoning is right there — in plain English, not behind a tab. The "Export Report" dropdown offers: Markdown, PDF, HTML, and **Executive Summary** (a one-page PDF designed to be forwarded to leadership — this replaces the persona toggle).

**Below the Answer: The Evidence**

The rest of the page provides supporting evidence, organized by the questions an engineer asks in order:

**Section 1: "How much better is this than no optimization?"**

Three numbers in a row — simple, no jargon:

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   92%         │  │   85%         │  │   +7%         │
│   Confidence  │  │   Baseline    │  │   Improvement │
│   (after opt) │  │   (before)    │  │   from GEPA   │
└──────────────┘  └──────────────┘  └──────────────┘
```

A first-time user reads left to right and immediately understands: "The optimization made things 7% better."

**Section 2: "Where does it work and where doesn't it?"**

A breakdown by output type — the most actionable view. This is the decision table, but presented as cards, not a dense table:

```
┌─────────────────────────────────────────────────────────┐
│  Results by Output Type                                 │
│                                                         │
│  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │ JSON Outputs    PASS│  │ Free Text (Short)  PASS │   │
│  │ 48/48 (100%)        │  │ 71/78 (91%)             │   │
│  │ All fields match    │  │ Minor tone differences  │   │
│  └─────────────────────┘  └─────────────────────────┘   │
│                                                         │
│  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │ Classification PASS │  │ Free Text (Long)   WARN │   │
│  │ 23/24 (96%)         │  │ 5/6 (83%)               │   │
│  │ 1 mismatch          │  │ Low sample size (6)     │   │
│  └─────────────────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Notice: no Wilson CIs, no P10/P50/P90, no "tier badges" ("unreliable"/"directional"/"likely representative"). Instead, plain language: "Low sample size (6)" tells you the same thing as a badge without requiring you to learn a badge system. If the user wants statistical detail, it's one click away (expand → shows CI, percentiles, metric spread).

**Section 3: "Show me the failures"**

A focused list of test cases that failed, sorted worst-first. This is the debugging view — only losses, not the full grid.

```
┌──────────────────────────────────────────────────────────┐
│  Needs Attention (14 of 156 test cases)                  │
│  [Show all test cases]                                   │
│                                                          │
│  Worst regression · Classification · Score: 0.31         │
│  "Expected: 'urgent' → Got: 'high_priority'"             │
│  [View diff →]                                           │
│                                                          │
│  Score: 0.45 · Free Text · Tone mismatch                 │
│  Response is correct but significantly more formal        │
│  [View diff →]                                           │
│                                                          │
│  ... (12 more)                                           │
└──────────────────────────────────────────────────────────┘
```

The "Show all test cases" link expands to the full filterable grid (equivalent to E.6 in the original brief, but accessed by choice, not by default). This is the key UX difference: the original brief shows everything and expects you to filter. This shows failures and lets you expand if you want more.

**Section 4: Diff View**

When you click "View diff" on any test case, a slide-over panel opens (matching the original brief's E.7/E.8 concept — that part was good):

```
┌─────────────────────────┬───────────────────────────────┐
│  Expected (gpt-4o)      │  Actual (claude-sonnet-4)     │
│                         │                               │
│  The quarterly revenue  │  The quarterly revenue        │
│  report shows a [-12%]  │  report shows a [+12%]        │ ← red/green highlighting
│  increase in Q3...      │  increase in Q3, driven       │
│                         │  primarily by...              │
│                         │                               │
│  Score: 0.72            │  Evaluators: BERTScore,       │
│  Output type: Free Text │  Embedding Similarity         │
└─────────────────────────┴───────────────────────────────┘
```

Side-by-side on desktop, unified on mobile. Same as original brief — this was well-designed.

**Section 5: Details (collapsed by default)**

Everything the power user needs, tucked away:
- Full configuration (model params, GEPA settings, data source)
- Optimized prompt text (in a code block)
- Score distributions (histograms)
- Statistical details (CIs, percentiles, metric decomposition)
- Pipeline warnings and safety scan results
- Cost breakdown

This section is the "original brief's E.1 through E.10" — all that detail exists, but it's behind a "Show Details" expand, not the default view. The 80% of users who just need the answer and the failures never scroll here. The 20% who need statistical rigor can find it.

---

## Page 3: Costs (`/costs`)

**Question:** "Are we spending wisely on LLM APIs?"

This page doesn't exist in the original brief at all. It's new and it's what makes this a platform, not just a migration tool.

### Data Source

RosettaStone can populate this from:
- Redis cache metadata (token counts per request, model identifier)
- Observability platform data (LangSmith, Braintrust logs)
- User-provided monthly spend figures (manual input for MVP)
- LiteLLM pricing data (always available, no integration needed)

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  Cost Overview                         [This month ▾]    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  $1,247       │  │  $312         │  │  $935         │  │
│  │  This month   │  │  Potential    │  │  After        │  │
│  │  (estimated)  │  │  savings      │  │  optimization │  │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                          │
│  Cost by Model                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  gpt-4o           ████████████████████  $823 (66%) │  │
│  │  claude-sonnet-4   ██████               $312 (25%) │  │
│  │  gpt-4o-mini       ██                   $112  (9%) │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Optimization Opportunities                              │
│  ┌────────────────────────────────────────────────────┐  │
│  │  💡 Switch gpt-4o → gpt-4o-mini for classification │  │
│  │     tasks. Estimated savings: $180/month.           │  │
│  │     These tasks don't require frontier reasoning.   │  │
│  │     [Evaluate this →]                               │  │
│  │                                                     │  │
│  │  💡 Switch gpt-4o → claude-haiku-4.5 for short      │  │
│  │     text generation. 60% cheaper, similar quality.  │  │
│  │     [Evaluate this →]                               │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

The "Optimization Opportunities" section is where RosettaStone becomes genuinely valuable between migrations. It analyzes your usage patterns and suggests cost-saving model switches — each one linking directly to a migration evaluation. "We noticed 40% of your GPT-4o calls are simple classifications that a cheaper model could handle. Want us to test that?"

For MVP, this page can start with just the LiteLLM pricing comparison (no real usage data needed). Show: "If you switched from X to Y, here's what you'd save based on current pricing." The real usage data integrations come in Phase 4.

---

## Page 4: Alerts (`/alerts`)

**Question:** "Is anything about to break?"

This page tracks model deprecations, pricing changes, and capability updates across providers. Powered by the deprecations.info API + LiteLLM model registry + periodic scraping of provider changelogs.

```
┌──────────────────────────────────────────────────────────┐
│  Alerts                                                  │
│                                                          │
│  Active Alerts (2)                                       │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  🔴 Deprecation: gpt-4o-0613                       │  │
│  │     Retiring April 15, 2026 (26 days)              │  │
│  │     Replacement: gpt-4o (latest)                   │  │
│  │     You have 3 systems using this model.           │  │
│  │     [Start Migration →]                            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  🟡 Price Change: anthropic/claude-sonnet-4        │  │
│  │     Price dropping from $3.00 to $2.50/1M input    │  │
│  │     Effective April 1, 2026                        │  │
│  │     No action needed — your costs will decrease.   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  🟢 New Model: claude-opus-4.6                     │  │
│  │     Released February 5, 2026                      │  │
│  │     Benchmark improvements: +12% on reasoning      │  │
│  │     Compatible with your current Claude setup.     │  │
│  │     [Explore →]                                    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Notification Settings                                   │
│  ☑ Email me about deprecations affecting my models       │
│  ☑ Email me about price changes > 10%                    │
│  ☐ Email me about new model releases                     │
└──────────────────────────────────────────────────────────┘
```

Three severity levels, tied to action required:
- **Red (Deprecation):** You must act or your system breaks. Shows countdown.
- **Yellow (Change):** Something changed that may affect you. Usually informational.
- **Green (Opportunity):** A new model or price change that could benefit you.

Every alert is actionable — it either links to "Start Migration" or "No action needed" with an explanation. No alert exists just to inform — it either requires action or explicitly says it doesn't.

---

## Executive Experience: The Export, Not the Toggle

The original brief has a persona toggle between Engineer and Executive views. I think this is wrong. Here's why:

An executive doesn't bookmark a dashboard URL and check it weekly. They receive a link or a PDF from their engineer who says "here's the migration report, we're recommending we switch." The executive experience is a *document*, not an *app*.

RosettaStone should generate a **one-page Executive Summary** as a beautifully formatted PDF/HTML that the engineer exports and shares. It contains:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  RosettaStone Migration Report                           │
│  gpt-4o → claude-sonnet-4                                │
│  March 20, 2026                                          │
│                                                          │
│  ─────────────────────────────────────────────────────── │
│                                                          │
│  Recommendation: Safe to Switch                          │
│                                                          │
│  The target model matches or exceeds our current model   │
│  in 92% of test cases (156 cases evaluated). Quality is  │
│  maintained across all output types. JSON outputs pass   │
│  at 100%. Free-text responses show minor stylistic       │
│  differences but preserve meaning and accuracy.          │
│                                                          │
│  ─────────────────────────────────────────────────────── │
│                                                          │
│  Quality: 92% match    Cost Impact: -$140/month          │
│  Risk: Low             Timeline: Ready to deploy         │
│                                                          │
│  ─────────────────────────────────────────────────────── │
│                                                          │
│  What improves:                                          │
│  • 15% faster response times                             │
│  • Native JSON mode (no parsing failures)                │
│  • 200K context window (up from 128K)                    │
│                                                          │
│  What to watch:                                          │
│  • Slightly more formal tone in customer-facing text     │
│  • 4 test cases showed classification differences        │
│                                                          │
│  ─────────────────────────────────────────────────────── │
│                                                          │
│  Prepared by RosettaStone v0.1.0                         │
│  Full technical report: [link]                           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

This replaces the entire Executive View (X.1 through X.6) from the original brief. The AI-generated narrative is preserved (it's the recommendation paragraph), but it lives in a shareable document, not an interactive dashboard that a VP will never bookmark.

The migration detail page still has a "Generate Executive Report" button — but it produces a PDF, not a view toggle.

---

## Design Principles for Implementation

### Visual Language

- **No jargon in primary UI.** "Safe to ship" not "GO." "92% match" not "91.0% [87%, 94%] (Wilson CI)." Technical details exist one click deeper.
- **Three colors, three meanings.** Green = good/safe. Amber = needs attention. Red = action required/blocked. No fourth color. No gradients of severity within a color.
- **Cards over tables for primary views.** Tables are for data exploration (test case grid). Cards are for decision-making (model cards, migration cards, alert cards). Default to cards; use tables only when the user is actively searching/filtering.
- **Progressive disclosure everywhere.** Answer → Evidence → Detail. Three levels max. The answer is always visible. Evidence is one click. Detail is two clicks.

### Onboarding Flow

1. First visit with no data → guided setup (register models)
2. First visit with CLI migration data → auto-populated Models page with migration results
3. Return visit → Models page showing health of registered models + any new alerts

### Empty States

Every empty state teaches the user what to do:
- No models: "Register your first model to start monitoring" + setup wizard
- No migrations: "Run `rosettastone migrate` to evaluate a model switch" + example command
- No alerts: "All clear — no deprecations or issues affecting your models" (with green checkmark — this is good news, not a missing feature)
- No cost data: "Connect a data source to track spending" + integration options

### Responsive Design

- Desktop: Full layout, side-by-side diffs, multi-column cards
- Tablet: Stacked cards, condensed tables
- Mobile: Simplified — answer card, recommendations, link to full detail. Mobile is for checking status, not debugging migrations.

---

## How This Maps to the Build Phases

| Phase | UI Scope |
|---|---|
| Phase 1 (MVP) | No web UI — CLI only with markdown report |
| Phase 2 | No web UI — CLI with richer output |
| Phase 3 | **Models page** (registered models, basic info), **Migrations page** (list + detail with answer-first pattern, diff view, test case grid), **Executive PDF export**, basic responsive layout |
| Phase 4 | **Costs page** (with LangSmith/Braintrust data), **Alerts page** (deprecation tracking via API), Model Explorer with match scores, optimization opportunities |
| Phase 5 | Pipeline migration views, A/B testing dashboard, migration versioning/history, team features, notification system |

---

## What Changed From the Original Brief

| Original Brief | This Vision | Why |
|---|---|---|
| Dashboard with fleet status cards | Models page with health cards | Fleet status is migration-centric; model health is always relevant |
| Persona toggle (Engineer/Executive) | Executive PDF export | Execs don't use dashboards — they read documents |
| 10 sub-sections in Engineer View | Answer → Evidence → Detail | Progressive disclosure beats information density |
| Wilson CI, P10/P50/P90, tier badges | Plain language ("Low sample size") | First-time users shouldn't need a statistics degree |
| GO/CONDITIONAL/NO_GO | Safe to ship / Needs review / Do not ship | Human language over engineer jargon |
| Migration-only dashboard | Models + Migrations + Costs + Alerts | Platform users come back between migrations |
| Empty state = empty tables | Guided setup wizard | Day 1 experience matters as much as day 100 |
| Dense filter bar + lazy-loaded table | Focused failure list + expandable full grid | Show failures by default, full grid on demand |
| Alert banner with severity badges | Attention bar with plain language priorities | "2 things need your attention" > "2 HIGH severity alerts" |
| Cost only shown in migration context | Dedicated Costs page with optimization suggestions | Cost visibility is a standalone value proposition |
| Deprecation as a migration trigger | Dedicated Alerts page with countdown timers | Deprecation awareness is continuous, not per-migration |
