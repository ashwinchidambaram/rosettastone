# RosettaStone — UI Vision (AC Brand Edition)

> Designing for decisions, not data. Every screen answers a question a human actually asks. Built on the AC Adaptive Palette System with full dark/light mode support.

---

## Design Foundation

### Brand Mode: Space-Forward with Earth Grounding

Per the AC Brand Guide, RosettaStone is a technical/dev tool project. This means:

- **Dark mode is the default** (Space-Forward mode) — engineers live in dark themes
- **Light mode is available** via toggle — for daytime use, screen sharing, and executive report generation
- **Terracotta 400 (#D4745E)** is the primary accent for CTAs and brand moments
- **Sage 400 (#8B9D83)** maps to success/active/pass states
- **Error 400 (#D85650)** for blockers and failures (warm red that matches the earth palette)
- **Gold 400 (#D4A574)** for warnings and attention states
- **Info 400 (#7A9FB5)** for informational and running states
- Earth tones provide warmth and grounding; monochromatic space tones provide structure

### Dark/Light Mode Mapping

```
Light Mode                    Dark Mode (default)
─────────────────────────────────────────────────
Sand 100 (#F9F7F4) bg    →   Space Black (#1A1A1A) bg
Sand 200 (#F3EFE8) surface →  Charcoal (#2C2C2C) surface
White (#FFFFFF) cards     →   Graphite (#3D3D3D) cards
Charcoal (#2C2C2C) text  →   Sand 300 (#E8DCC4) text
Gray 700 (#616161) secondary → Gray 400 (#BDBDBD) secondary
Gray 300 (#E0E0E0) borders → Slate Dark (#4A4A4A) borders

Terracotta 400 (#D4745E)  →   Same (accent colors don't change)
Sage 400 (#8B9D83)        →   Same
Gold 400 (#D4A574)        →   Same
Error 400 (#D85650)       →   Same
Info 400 (#7A9FB5)        →   Same
```

### Typography

- **Primary:** Inter (400, 500, 600, 700)
- **Code/Monospace:** JetBrains Mono (diffs, prompts, technical content)
- **Base size:** 16px, line-height 1.6 for body
- **KPI numbers:** Inter Semibold, 32-48px, tabular figures
- **Navigation:** Inter Medium 500, 16px

### Component Patterns (from AC Brand Guide)

- **Cards:** Standard Card (12px radius, 24px padding, subtle shadow) for data cards. Elevated Card (16px radius, 32px padding) for recommendation cards. Colored Accent Card (4px left border in status color) for status-driven cards.
- **Buttons:** Primary (Terracotta 400 fill, Sand 100 text), Secondary (outline, hover → Terracotta), Ghost (text-only, Sand 200 hover bg). On dark: add subtle glow on hover.
- **Nav bar:** 72px height, sticky, border-bottom. Active link: Terracotta 500 with 2px bottom border.
- **Icons:** Lucide Icons, 24px standard, 2px stroke, inherit text color or Gray 700.
- **Status dots:** 8px colored circles. Sage 400 = active/pass. Error 400 = fail/blocked. Gold 400 = warning/attention. Info 400 = running/informational.
- **Spacing:** 8px base unit. Card padding 24px. Section gaps 48-64px. Generous whitespace — "default to MORE space, not less."

### Dark Mode-Specific Enhancements

- Cards get `1px solid rgba(189, 189, 189, 0.1)` border and darker shadow
- Interactive cards on hover: `border-color: rgba(212, 116, 94, 0.3)` with subtle terracotta glow
- CTAs get a subtle `box-shadow: 0 0 16px rgba(212, 116, 94, 0.3)` glow
- Status badges maintain their color but get slightly increased saturation for dark bg readability
- Theme toggle: sun/moon icon in the nav bar, right side near settings

---

## Design Philosophy

Three principles:

**1. Design for day 1, not day 100.** A first-time user with zero migrations should feel welcomed, not overwhelmed. The UI teaches by guiding, not by showing empty scaffolding.

**2. Answer the question, then show the evidence.** Lead with the answer ("Safe to switch" / "3 models need attention" / "You could save $400/month"). Supporting data is behind a click, not in front of the answer.

**3. Two audiences, two experiences.** Engineers get an interactive dashboard with drill-down capability. Executives get a shareable document (PDF export), not a stripped-down dashboard. The "executive view" is an export, not a toggle.

---

## Information Architecture

Four sections via the top nav bar:

```
┌──────────────────────────────────────────────────────────┐
│  🪨 RosettaStone    Models  Migrations  Costs  Alerts  ☀/🌙 ⚙ │
└──────────────────────────────────────────────────────────┘
```

| Section | Question | Primary user |
|---|---|---|
| **Models** (landing) | "What are we running and what's out there?" | Both |
| **Migrations** | "Did the migration work? Should we ship it?" | Engineer |
| **Costs** | "Are we spending wisely?" | Both |
| **Alerts** | "Is anything about to break?" | Both |

**Models** is the default landing page. Most of the time, users aren't migrating — they're monitoring their model landscape.

---

## Page 1: Models (Landing Page — `/`)

**Question:** "What are we running, and is there anything we should know?"

### First-Time Experience (Zero State)

Centered welcome experience on the Space Black (dark) or Sand 100 (light) background:

- Large heading (Inter Semibold 36px): "Welcome to RosettaStone"
- Subheading (Inter Regular 18px, secondary text color): "Let's set up your model landscape."
- A Standard Card containing:
  - "Which models are you currently using?"
  - Input fields with placeholder text in Gray 500
  - "+ Add another model" ghost link in Terracotta 400
  - Helper text in Caption size (12px): "We use LiteLLM model identifiers (e.g., openai/gpt-4o)"
  - Primary button: "Set up models" (Terracotta 400 fill)
- Secondary options below: "Import from migration results" | "Connect Redis cache" | "Connect LangSmith" as ghost text links

### Active State — Model Landscape

**Zone 1: Attention bar** — Only when action needed. Full-width, subtle Gold 100 (light) or Gold with low opacity (dark) background. Left Gold 400 border accent. Plain language: "2 things need your attention" with arrow link. Dismissible.

**Zone 2: Your Models** — Card grid (2 columns desktop, 1 mobile). Each model is a Standard Card:
- Model name as H4 (Inter Medium 18px)
- Status dot (8px, Sage 400 for active, Gold 400 for deprecated) + status text
- Specs as Body Small (14px secondary text): context window, cost, capabilities
- Deprecated cards get a 4px Gold 400 left border and show recommended replacement
- "Run migration →" ghost link in Terracotta 400 at bottom of each card
- "+ Add model" card with dashed border

**Zone 3: Explore Models** — Below the fold. Search bar (AC Brand input style: Sand 100 bg, 2px Gray 400 border, focus → Terracotta 400 border) + filter dropdowns. Clean table with model candidates, "Match %" column showing capability overlap. Info icon explaining Match with tooltip.

---

## Page 2: Migrations (`/migrations`)

**Question:** "What migrations have I run, and how did they go?"

### Migration List

Stacked full-width Colored Accent Cards with status-colored left borders:

- **Safe to ship** → 4px Sage 400 left border, Sage badge
- **Needs review** → 4px Gold 400 left border, Gold badge
- **Do not ship** → 4px Error 400 left border, Error badge

Each card shows: model pair (H4), recommendation badge, confidence %, test case count, cost, timestamp. Entire card clickable → detail page. On hover in dark mode: subtle terracotta glow border.

Empty state: centered text with "No migrations yet. Run `rosettastone migrate` to get started." (command in JetBrains Mono) + "+ New Migration" primary button.

### Migration Detail (`/migrations/{id}`)

**The answer first:**

Elevated Card (16px radius, 32px padding) with status-colored left border. Contains:
- Large status line (Inter Semibold 28px): "✓ Safe to ship" / "✗ Do not ship" / "⚠ Needs review"
- Recommendation reasoning (Body Regular 16px, 1.6 line-height)
- Two buttons: "Export report ▾" (Primary) and "View optimized prompt" (Secondary)

**KPI row** — Three Elevated Cards:
- Confidence % (large number, Sage/Gold/Error colored based on threshold)
- Baseline % (neutral — Gray 700 in light, Gray 400 in dark)
- Improvement delta (Sage 400 if positive, Error 400 if negative)

Numbers displayed at 32px Inter Semibold with tabular figures. Subtitles at 14px Body Small.

**Results by output type** — 2×2 card grid. Each card:
- Output type name (H4)
- Win rate as large number
- "PASS" (Sage 400 badge) / "FAIL" (Error 400 badge) / "WARN" (Gold 400 badge)
- One-line plain language description
- Failing cards get Error 400 left border accent

**Needs attention** — Focused list of failures, worst first. Each item:
- Score (color-coded) + output type + one-line description
- "View diff →" terracotta ghost link
- "Show all test cases" expandable link at bottom

**Details section** — Collapsed by default (Ghost button: "Configuration & details ▸"). Contains: full config as key-value table, optimized prompt in JetBrains Mono code block, score distribution histograms, statistical details (CIs, percentiles), pipeline warnings, safety scan results.

---

## Page 3: Costs (`/costs`)

**Question:** "Are we spending wisely on LLM APIs?"

**Three KPI Elevated Cards:**
- Current month spend (neutral)
- Potential savings (Sage 400 accent)
- Projected after optimization (neutral)

**Cost by model** — Horizontal bar chart. Bars use brand-appropriate colors (Terracotta 300 for primary model, Sage 300 for secondary, Gold 300 for tertiary). Labels show model name, dollar amount, percentage.

**Optimization opportunities** — Stacked Colored Accent Cards with Sage 50 (light) or subtle Sage tint (dark) background. Lightbulb icon (Lucide). Each card: suggestion text, estimated savings, "Evaluate this →" terracotta ghost link.

---

## Page 4: Alerts (`/alerts`)

**Question:** "Is anything about to break?"

Three alert types, each a Colored Accent Card:
- **Deprecation** (Error 400 left border): Model name, retirement date with countdown, affected systems count, replacement suggestion, "Start migration →" primary button
- **Price change** (Gold 400 left border): Model name, price change details, effective date, "No action needed" with Sage checkmark if informational
- **New model** (Sage 400 left border): Model name, release date, key improvements, compatibility note, "Explore →" ghost link

**Notification settings** at bottom: checkboxes for email preferences (deprecations, price changes, new releases).

---

## Executive Experience: The Export

Not a view toggle — a shareable PDF document. Generated via "Export report ▾ → Executive summary" on any migration detail page.

**Design the PDF in light mode always** (Sand 100 background, Charcoal text) — PDFs are printed/screenshared, so light mode is appropriate regardless of the user's dashboard preference.

**Layout:**
- Header: "RosettaStone Migration Report" (Caption, uppercase tracking), model pair (H1), date
- Recommendation: Large text in status color. Reasoning paragraph below.
- Four metric boxes: Quality Match, Cost Impact, Risk Level, Deployment Status
- Two columns: "What improves" (Sage accent) and "What to watch" (Gold accent)
- Footer: version, link to full technical report

Professional, clean, forwardable. Like a one-page consulting deliverable.

---

## Dark/Light Mode Toggle Behavior

- **Default:** Dark mode (Space-Forward, per AC Brand Guide for technical projects)
- **Toggle location:** Sun/moon icon in the nav bar, right side
- **Persistence:** Saved to localStorage, respects `prefers-color-scheme` system setting on first visit
- **Transition:** 200ms ease on all color properties when toggling
- **Scope:** Affects the entire dashboard UI. Does NOT affect exported PDFs (always light mode).
- **Implementation:** CSS custom properties for all colors, toggled via a `data-theme="light"` attribute on `<html>`

---

## How This Maps to Build Phases

| Phase | UI Scope |
|---|---|
| Phase 1 (MVP) | No web UI — CLI only with markdown report |
| Phase 2 | No web UI — CLI with richer output |
| Phase 3 | **Models page**, **Migrations page** (list + detail), **Executive PDF export**, dark/light mode toggle |
| Phase 4 | **Costs page**, **Alerts page**, Model Explorer with match scores, notification settings |
| Phase 5 | Pipeline migration views, A/B testing dashboard, migration versioning, team features |
