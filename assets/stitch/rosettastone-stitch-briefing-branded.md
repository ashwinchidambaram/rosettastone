# RosettaStone — Stitch UI Briefing (AC Brand Edition)

This document has two parts:
1. **DESIGN.MD** — Design system context for Stitch to maintain consistency
2. **Screen Prompts** — Individual prompts for each screen, one at a time

---

# Part 1: DESIGN.MD

## Product Context

RosettaStone is a model intelligence platform for AI engineering teams. It helps teams manage their LLM (Large Language Model) landscape — tracking which models they're running, monitoring for deprecations and cost changes, and automating the process of migrating between models.

Four main sections via top navigation:
- **Models** (landing page) — "What are we running and what's out there?"
- **Migrations** — "Did the migration work? Should we ship it?"
- **Costs** — "Are we spending wisely?"
- **Alerts** — "Is anything about to break?"

## Design Principles

- **Answer first, evidence behind.** Every page leads with the answer. Supporting data is one click deeper.
- **Human language, not jargon.** "Safe to ship" not "GO." "92 percent match" not "91.0% [87%, 94%] (Wilson CI)."
- **Progressive disclosure.** Answer → Evidence → Detail. Three levels max.
- **Generous whitespace.** Like the vast space between celestial bodies — designs should breathe. Default to MORE space, not less.

## Brand System: AC Adaptive Palette

This is a developer tool using the **Space-Forward** brand mode. The design must support both dark mode (default) and light mode.

### Color Palette

**Earth Tone Accents (same in both modes):**
- Terracotta 400: #D4745E — Primary CTAs, brand accent, action links
- Sage 400: #8B9D83 — Success, active, pass states
- Gold 400: #D4A574 — Warnings, attention states
- Error 400: #D85650 — Failures, blockers, destructive states (warm red)
- Info 400: #7A9FB5 — Informational, running states, links

**Dark Mode (default):**
- Background: Space Black #1A1A1A
- Surface/secondary bg: Charcoal #2C2C2C
- Cards: Graphite #3D3D3D with 1px border rgba(189,189,189,0.1) and shadow 0 4px 16px rgba(26,26,26,0.4)
- Primary text: Warm Sand #E8DCC4
- Secondary text: Gray 400 #BDBDBD
- Borders/dividers: Slate Dark #4A4A4A
- Card hover: border-color rgba(212,116,94,0.3) with subtle terracotta glow
- CTA glow on hover: box-shadow 0 0 16px rgba(212,116,94,0.3)

**Light Mode:**
- Background: Sand 100 #F9F7F4 (warm off-white)
- Surface/secondary bg: Sand 200 #F3EFE8 (cream)
- Cards: White #FFFFFF with 1px border #E0E0E0 and shadow 0 2px 8px rgba(44,44,44,0.06)
- Primary text: Charcoal #2C2C2C
- Secondary text: Gray 700 #616161
- Borders/dividers: Gray 300 #E0E0E0
- Card hover: shadow 0 4px 16px rgba(44,44,44,0.08), translateY(-2px)

**Status Badge Colors:**
- Pass/Active: Sage 400 #8B9D83 background, Sage 800 #333A2E text (light mode) / Sage 100 #E3E8DF text (dark mode)
- Fail/Blocked: Error 400 #D85650 background, white text
- Warning/Review: Gold 400 #D4A574 background, Gold 800 #5E422A text (light) / Gold 100 #F7E9D1 text (dark)
- Info/Running: Info 400 #7A9FB5 background, white text

### Typography

- **Primary font:** Inter (weights 400, 500, 600, 700). Fallback: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto
- **Monospace:** JetBrains Mono (for code blocks, diffs, prompts, commands)
- **Page titles:** Inter Semibold 600, 28-36px, letter-spacing -0.01em
- **Section headings:** Inter Semibold 600, 22-28px
- **Card headings:** Inter Medium 500, 18px
- **Body text:** Inter Regular 400, 16px, line-height 1.6
- **Secondary/metadata:** Inter Regular 400, 14px, line-height 1.5
- **Captions/labels:** Inter Regular 400, 12px, letter-spacing 0.01em
- **KPI numbers:** Inter Semibold 600, 32-48px, tabular figures
- **Navigation links:** Inter Medium 500, 16px, letter-spacing 0.005em

### Spacing

- Base unit: 8px
- Card internal padding: 24px (standard) or 32px (elevated)
- Gaps between cards: 16-24px
- Section spacing: 48-64px
- Container padding: 48px desktop, 24px mobile
- Border radius: 8px (buttons, inputs), 12px (standard cards), 16px (elevated cards)

### Component Patterns

**Navigation bar:** 72px height, sticky top. Background: Charcoal (dark) / Sand 100 (light) with 1px bottom border. Logo "RosettaStone" on left with stone icon. Four nav links center-left. Active link: Terracotta 500 color with 2px bottom border. Theme toggle (sun/moon icon) and settings gear on right side.

**Standard card:** Graphite bg (dark) / White bg (light). 12px border-radius, 24px padding, subtle shadow. 1px border. On hover if interactive: slight lift and border glow (dark) or increased shadow (light).

**Elevated card:** Same as standard but 16px radius, 32px padding, stronger shadow. Used for recommendation cards and KPI displays.

**Colored accent card:** Standard card + 4px left border in status color. Used for model cards with deprecation warnings, alert cards, migration result cards.

**Primary button:** Terracotta 400 background, Sand 100 text. 8px radius, 12px vertical / 24px horizontal padding. Hover: Terracotta 500 + translateY(-1px) + subtle glow (dark mode).

**Secondary button:** Transparent bg, Charcoal (light) / Sand 300 (dark) text, 2px Gray 400 border. Hover: border → Terracotta 400, text → Terracotta 400.

**Ghost link:** No border, Terracotta 400 text. Hover: Sand 200 bg (light) / rgba(212,116,94,0.1) bg (dark).

**Input fields:** Sand 100 bg (light) / Charcoal bg (dark), 2px Gray 400 border, 8px radius. Focus: Terracotta 400 border.

**Status dots:** 8px circles. Sage 400 (active), Gold 400 (deprecated/warning), Error 400 (fail), Info 400 (running).

**Icons:** Lucide Icons, 24px size, 2px stroke weight, inherit text color.

---

# Part 2: Screen-by-Screen Prompts

Generate each screen using the DESIGN.MD above as context. Generate the **dark mode version** first (it's the default), then optionally generate a light mode variant.

---

## Prompt 1: Navigation Shell + Models Page (Dark Mode, Populated)

Design a web application dashboard for "RosettaStone" — a model intelligence platform for AI engineering teams. Use a dark color scheme.

**Overall look:** Dark background (#1A1A1A), warm earth-tone accents, generous whitespace, modern and clean. Think of it as a developer tool that feels warm and human despite being technical. The font is Inter for all text.

**Navigation bar:** Sticky top bar, 72px tall, background #2C2C2C with 1px bottom border in #4A4A4A. Left side: A small stone/gem icon and the text "RosettaStone" in Inter Semibold, color #E8DCC4 (warm sand). Center-left: four nav links — "Models" (active, with #C4624D color and a 2px bottom border in #D4745E), "Migrations", "Costs", "Alerts" (with a small red #D85650 notification dot). Right side: a sun icon for theme toggle, and a gear icon for settings, both in #BDBDBD.

**Page background:** #1A1A1A

**Attention banner:** A full-width subtle banner below the nav. Background: rgba(212, 165, 116, 0.1) with a 4px left border in #D4A574 (gold). Text in #E8DCC4: "2 things need your attention" with a right-arrow. Below in smaller #BDBDBD text: "gpt-4o-0613 is deprecated — retiring Apr 15" and "Claude Haiku 4.5 costs 40 percent less than your current Haiku". The banner should feel calm, not alarming.

**Section heading:** "Your models" in Inter Semibold 22px, color #E8DCC4. Below it, a grid of 4 cards (2 columns):

**Card 1:** Background #3D3D3D, 12px rounded corners, 24px padding, 1px border rgba(189,189,189,0.1). Title: "openai/gpt-4o" in Inter Medium 18px, #E8DCC4. Next to it, an 8px circle in #8B9D83 (sage green) and "Active" in small #BDBDBD text. Below: "Context: 128K tokens · Cost: $2.50/1M input · Features: Tools, Vision, JSON" in 14px #BDBDBD. At the bottom: "Run migration →" in #D4745E (terracotta).

**Card 2:** Same style. "anthropic/claude-sonnet-4", sage green dot, Active. "Context: 200K tokens · Cost: $3.00/1M input · Features: Tools, JSON".

**Card 3:** Same style. "openai/gpt-4o-mini", sage green dot, Active. "Context: 128K tokens · Cost: $0.15/1M input · Features: Tools, JSON".

**Card 4:** Has a 4px left border in #D4A574 (gold) instead of the normal border. Title: "openai/gpt-4o-0613". Gold dot with "Deprecated — retiring Apr 15" in #D4A574 text. Same info section. Below: "Recommended replacement: openai/gpt-4o (latest)" in 14px text, and "Start migration →" in #D4745E.

After the four cards, a "+ Add model" card with a dashed #4A4A4A border and a "+" icon.

**Section: "Explore models"** below. Heading in Inter Semibold 22px. A search input field (background #2C2C2C, 2px border #4A4A4A, placeholder "Search models..." in #9E9E9E) and two dropdown filters ("Provider", "Capability"). Below, a clean table with #2C2C2C header row and #3D3D3D alternating rows. Columns: Model, Provider, Cost/1M Input, Context Window, Match %. Show 5 rows (gemini-2.5-flash, llama-4-maverick, claude-opus-4.6, mistral-large, deepseek-v3). The Match % has a subtle inline progress bar behind the number using terracotta at reduced opacity.

The overall feel should be spacious, dark, warm, and organized — a tool you'd enjoy opening every morning.

---

## Prompt 2: Models Page — First-Time / Empty State (Dark Mode)

Same navigation shell as Prompt 1 but no notification dot on Alerts.

No attention banner. The main area shows a centered welcome experience on the #1A1A1A background:

Large heading in Inter Semibold 36px, #E8DCC4: "Welcome to RosettaStone"
Subheading in Inter Regular 18px, #BDBDBD: "Let's set up your model landscape."

Below, an elevated card: #3D3D3D background, 16px rounded corners, 32px padding, subtle shadow. Inside:
- "Which models are you currently using?" in Inter Medium 18px
- Two input fields with #2C2C2C background, 2px border #4A4A4A, placeholder text "openai/gpt-4o" and "anthropic/claude-sonnet-4" in #9E9E9E
- "+ Add another model" ghost link in #D4745E
- Helper text in 12px #9E9E9E: "We use LiteLLM model identifiers (e.g., openai/gpt-4o)"
- A primary button: "Set up models" with #D4745E background, #F9F7F4 text, 8px radius, subtle glow

Below the card, two lines of subtle links:
"Or import from migration results" in #D4745E
"Or connect your infrastructure:" followed by two secondary outline buttons: "Connect Redis cache" and "Connect LangSmith" with #4A4A4A borders and #BDBDBD text.

The design should feel welcoming and guided — the user immediately knows what to do.

---

## Prompt 3: Migrations List Page (Dark Mode)

Navigation: "Migrations" is now active (terracotta bottom border). Dark background.

Page heading in Inter Semibold 28px, #E8DCC4: "Migrations". Right-aligned primary button: "+ New migration" in #D4745E background.

Three stacked full-width cards, each clickable:

**Card 1:** #3D3D3D background, 12px radius. 4px left border in #8B9D83 (sage — success). Left side: "gpt-4o → claude-sonnet-4" in Inter Medium 18px. Right side: a small rounded badge with #8B9D83 background and dark text: "Safe to ship". Below the title in 14px #BDBDBD: "92% confidence · 156 test cases · $2.34 · 2 minutes ago". On hover: border glows subtly with rgba(139,157,131,0.3).

**Card 2:** Same layout. 4px left border in #D4A574 (gold). "gpt-4o-0613 → gpt-4o". Gold badge: "Needs review". "78% confidence · 43 test cases · $1.12 · 1 day ago".

**Card 3:** Same layout. 4px left border in #D85650 (error red). "gpt-3.5-turbo → gpt-4o-mini". Red badge: "Do not ship". "61% confidence · 89 test cases · $0.87 · 3 days ago".

Below: subtle "Showing 3 migrations" in 12px caption text.

Clean, scannable. Each card communicates the essential info at a glance.

---

## Prompt 4: Migration Detail — "Safe to Ship" (Dark Mode)

Navigation: "Migrations" active. Below nav, breadcrumb: "← Migrations" in #D4745E ghost link.

**Page header:** "gpt-4o → claude-sonnet-4" in Inter Semibold 28px, #E8DCC4. Small rounded "Complete" badge in #8B9D83 next to it.

**Recommendation card:** Elevated card (#3D3D3D, 16px radius, 32px padding) with 4px left border in #8B9D83. Large "✓ Safe to ship" in Inter Semibold 28px, #8B9D83. Below, paragraph in 16px #E8DCC4: "The target model matches or exceeds the source in 92 percent of test cases. JSON outputs pass at 100 percent. Free-text responses show minor tone differences but maintain semantic equivalence." Two buttons below: "Export report ▾" (primary, #D4745E) and "View optimized prompt" (secondary, outline).

**Three KPI cards** in a row (elevated cards, 16px radius):
- "92%" in Inter Semibold 32px #8B9D83, subtitle "Confidence (after optimization)" in 14px #BDBDBD
- "85%" in 32px #BDBDBD, subtitle "Baseline (before optimization)"
- "+7%" in 32px #8B9D83, subtitle "Improvement from GEPA"

**Section: "Results by output type"** — heading in Inter Semibold 22px. Four cards in 2×2 grid:
- "JSON outputs" — "48/48 (100%)" large, sage "PASS" badge, subtitle "All fields match"
- "Free text (short)" — "71/78 (91%)", sage "PASS", "Minor tone differences"
- "Classification" — "23/24 (96%)", sage "PASS", "1 mismatch"
- "Free text (long)" — "5/6 (83%)", gold "WARN" badge, "Low sample size (6)" — this card has a Gold 400 left border

**Section: "Needs attention"** — heading with "(14 of 156 test cases)" in secondary text. List items:
- "Worst regression · Classification · Score: 0.31" with "Expected 'urgent', got 'high_priority'" and "View diff →" in #D4745E
- Second item: "Score: 0.45 · Free text" with description and diff link
- "... 12 more" collapsed. "Show all 156 test cases" expandable link.

**Collapsed section:** "Configuration and details ▸" ghost button that expands to show full config.

---

## Prompt 5: Migration Detail — "Do Not Ship" (Dark Mode)

Same layout as Prompt 4 but with error states:

Recommendation card: 4px left border #D85650. "✗ Do not ship" in Inter Semibold 28px, #D85650. Paragraph explaining failures.

KPIs: "61%" in #D85650, "58%" neutral, "+3%" dim (barely improved — in #BDBDBD, not green).

Output type cards: 2 with Error "FAIL" badges (red left border), 1 with Gold "WARN", 1 with Sage "PASS".

Same structure, red dominant color on the recommendation card makes the answer immediately clear.

---

## Prompt 6: Costs Page (Dark Mode)

Navigation: "Costs" active.

Heading: "Cost overview" with dropdown "This month ▾" on right.

Three elevated KPI cards:
- "$1,247" in 32px #E8DCC4, subtitle "Estimated this month"
- "$312" in 32px #8B9D83, subtitle "Potential savings"
- "$935" in 32px #E8DCC4, subtitle "After optimization"

**Cost by model section:** Horizontal bar chart on dark background. Three bars:
- "openai/gpt-4o" — longest bar in #D4745E at 40% opacity, "$823 (66%)" label
- "anthropic/claude-sonnet-4" — medium bar in #8B9D83 at 40% opacity, "$312 (25%)"
- "openai/gpt-4o-mini" — short bar in #D4A574 at 40% opacity, "$112 (9%)"

**Optimization opportunities:** Two cards with subtle sage-tinted background (rgba(139,157,131,0.05)) and Sage 400 left border accent:
- Lucide lightbulb icon in #D4A574. "Switch gpt-4o → gpt-4o-mini for classification tasks." Subtitle: "Estimated savings: $180/month. These tasks don't need frontier reasoning." "Evaluate this →" in #D4745E.
- Second card same pattern with different suggestion.

---

## Prompt 7: Alerts Page (Dark Mode)

Navigation: "Alerts" active, red notification dot.

Heading: "Alerts"

**Active alerts (2):**

Card 1: #3D3D3D card, 4px left border #D85650 (error red). "Deprecation: gpt-4o-0613" in Inter Medium 18px. "Retiring April 15, 2026 (26 days)" with subtle countdown. "You have 3 systems using this model." "Replacement: openai/gpt-4o (latest)". Primary button: "Start migration →" in #D4745E.

Card 2: 4px left border #D4A574 (gold). "Price change: anthropic/claude-sonnet-4". "Price dropping from $3.00 to $2.50/1M input. Effective April 1, 2026." "No action needed — your costs will decrease." Sage checkmark icon.

**Recent updates:**

Card 3: 4px left border #8B9D83 (sage). "New model: claude-opus-4.6". "Released February 5, 2026. Benchmark improvements: +12 percent reasoning, +8 percent coding. Compatible with your current Claude setup." "Explore →" ghost link.

**Notification settings** section at bottom: Three checkboxes with #D4745E accent when checked:
- ☑ "Email me about deprecations affecting my models"
- ☑ "Email me about price changes over 10 percent"
- ☐ "Email me about new model releases"

---

## Prompt 8: Executive Report PDF (Light Mode)

Design a clean, printable one-page document. This should look like a polished PDF — NOT a dashboard. Use LIGHT colors: #F9F7F4 (warm off-white) background, #2C2C2C (charcoal) text.

**Header:** Small caps "ROSETTASTONE MIGRATION REPORT" in Inter Medium 12px, #616161 (gray), letter-spacing 0.1em. Below: "gpt-4o → claude-sonnet-4" in Inter Semibold 28px, #2C2C2C. Date: "March 20, 2026" in 14px #616161. A thin horizontal line in #E0E0E0.

**Recommendation:** "Recommendation: Safe to switch" in Inter Semibold 22px, #8B9D83 (sage green). Below, paragraph in 16px #2C2C2C with 1.6 line-height: "The target model matches or exceeds our current model in 92 percent of test cases (156 cases evaluated). Quality is maintained across all output types. JSON outputs pass at 100 percent. Free-text responses show minor stylistic differences but preserve meaning and accuracy."

Thin #E0E0E0 divider.

**Four metric boxes** in a row, each with a thin #E0E0E0 border and 16px padding:
- "92%" large in #2C2C2C, label "Quality match" in 12px #616161
- "-$140/mo" large in #8B9D83, label "Cost impact"
- "Low" large in #8B9D83, label "Risk level"
- "Ready" large in #8B9D83, label "Deployment status"

Divider.

**Two columns:**
Left: "What improves" heading in Inter Medium 16px with a small #8B9D83 dot. Bullet list in 14px #2C2C2C: "15 percent faster response times", "Native JSON mode (no parsing failures)", "200K context window (up from 128K)".
Right: "What to watch" heading with #D4A574 dot. Bullet list: "Slightly more formal tone in customer-facing text", "4 classification differences in edge cases".

**Footer:** "Prepared by RosettaStone v0.1.0 · Full technical report: [link]" in 12px #9E9E9E. Subtle horizontal line above.

Professional, warm, clean. A document a VP can read in 60 seconds and forward to their boss.

---

## Prompt 9: Diff View Slide-Over Panel (Dark Mode)

Design a slide-over panel from the right side (50-60 percent page width). Background of the panel: #2C2C2C. The area behind the panel is dimmed with a dark overlay.

**Panel header:** Close button (X icon, #BDBDBD) top right. Left side: a small rounded badge "Free text" in #7A9FB5 (info blue) background, then large "0.72" in Inter Semibold 28px #D4A574 (gold, since it's in warning range), and a small "LOSS" badge in #D85650 background with white text.

**Metric bars:** Three horizontal progress bars on the #2C2C2C background:
- "BERTScore: 0.78" — bar fills to 78% using #8B9D83 (sage) at 40% opacity, text in #BDBDBD
- "Embedding similarity: 0.81" — bar fills to 81% using #8B9D83
- "Composite: 0.72" — bar fills to 72% using #D4A574 (gold, since below threshold)

**Tab toggle:** Two tabs above the diff area — "Side-by-side" (active, #D4745E underline) and "Unified". Tabs in 14px Inter Medium.

**PII warning banner:** Full width of the panel. Subtle #D4A574 background at 10% opacity with gold left border. Small text: "Content may contain sensitive data from production prompts" in 14px #D4A574.

**Diff view:** Two columns. Left header: "Expected (gpt-4o)" in 12px #9E9E9E. Right header: "Actual (claude-sonnet-4)" in 12px #9E9E9E. Below each header, text content in JetBrains Mono 14px on a #1A1A1A background. Show a realistic paragraph about quarterly revenue where a few words differ. Deleted words highlighted with rgba(216,86,80,0.2) (error red at low opacity). Added words highlighted with rgba(139,157,131,0.2) (sage green at low opacity). Unchanged text in #E8DCC4.

The panel should feel focused and technical — this is the engineer's debugging view.

---

## Prompt 10 (Bonus): Light Mode Variant of Models Page

Take the exact same layout and content as Prompt 1, but render it in light mode:

- Background: #F9F7F4 (warm off-white)
- Nav bar background: #F9F7F4 with 1px bottom border #E0E0E0
- Nav text: #2C2C2C, active: #C4624D with terracotta underline
- Cards: White #FFFFFF background, 1px border #E0E0E0, shadow 0 2px 8px rgba(44,44,44,0.06)
- Card text: #2C2C2C primary, #616161 secondary
- Model name: #2C2C2C
- Status dots: Same colors (sage, gold)
- "Run migration →" link: #D4745E
- Attention banner: #F7E9D1 background (light gold) with #D4A574 left border
- Search input: #F9F7F4 background, 2px #BDBDBD border
- Table rows: White and #F3EFE8 alternating

Everything else (layout, spacing, content, type sizes) stays identical. Only colors change. This demonstrates the dark/light mode system working across the same layout.

---

## Notes for Working with Stitch

**Generation order:** Generate screens in order. Prompt 1 establishes the visual language. If Stitch drifts on later screens, reference "use the exact same card style, typography, and color palette as the Models page."

**Key brand colors to reinforce if Stitch drifts:**
- Dark bg: #1A1A1A, Cards: #3D3D3D, Text: #E8DCC4
- Action/links: #D4745E (terracotta)
- Success: #8B9D83 (sage green)
- Warning: #D4A574 (gold)
- Error: #D85650 (warm red)

**Prototyping connections:**
- Model card "Run migration" → New Migration form (future)
- Migration list card click → Migration Detail page
- "View diff →" → Diff slide-over panel
- "Export report → Executive summary" → PDF preview (Prompt 8)
- Theme toggle → Swaps between dark (Prompts 1-7, 9) and light (Prompt 10)

**Responsive:** Design for 1440px desktop. Cards stack to 1 column on mobile. Diff panel goes full-width on mobile. Nav collapses to hamburger on mobile.

**Tone check:** After each screen — would a first-time engineer understand everything without a tutorial? Would it feel warm despite being a dark dev tool? If not, simplify and add whitespace.
