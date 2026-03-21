# RosettaStone Web UI -- Playwright Test Plan

> Definitive QA specification for all UI pages, interactions, and edge cases.
> Target: `tests/test_server/test_playwright_ui.py`

---

## Technical Setup

### Server Startup

```python
# Fixture: start uvicorn as subprocess, wait for healthy, yield base_url, teardown
# Command: uv run uvicorn rosettastone.server.app:create_app --factory --port 8000
# Health check: GET http://localhost:8000/api/v1/health -> {"status": "ok"}
# Use playwright.sync_api with Chromium
# BASE_URL = "http://localhost:8000"
```

### Fixture Pattern

```python
import subprocess, time, signal
import pytest
from playwright.sync_api import sync_playwright, Page, expect

@pytest.fixture(scope="session")
def server():
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "rosettastone.server.app:create_app",
         "--factory", "--port", "8765"],
        cwd="/Users/ashwinchidambaram/dev/projects/rosettastone",
    )
    # Poll health endpoint until ready (max 15s)
    import urllib.request
    for _ in range(30):
        try:
            resp = urllib.request.urlopen("http://localhost:8765/api/v1/health")
            if resp.status == 200:
                break
        except Exception:
            time.sleep(0.5)
    yield "http://localhost:8765"
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)

@pytest.fixture(scope="session")
def browser():
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    yield browser
    browser.close()
    pw.stop()

@pytest.fixture
def page(browser, server):
    page = browser.new_page()
    yield page
    page.close()
```

### General Verification Patterns

- `page.goto(url)` -- navigate
- `page.locator(selector)` -- find elements
- `expect(locator).to_be_visible()` -- visibility assertion
- `expect(locator).to_have_text(text)` -- text content assertion
- `expect(locator).to_have_attribute(name, value)` -- attribute assertion
- `expect(locator).to_have_count(n)` -- element count assertion
- `page.wait_for_load_state("networkidle")` -- wait for HTMX requests
- `page.locator(selector).click()` -- interaction

---

## 1. Models Dashboard (`/ui/`)

### 1.1 URL and Route
- **GET** `/ui/`
- Route handler: `dashboard()` in `migrations.py`
- Template: `models.html` extends `base.html`
- Context: `models=DUMMY_MODELS`, `alerts=DUMMY_ALERTS`, `active_nav="models"`

### 1.2 Elements to Verify Exist

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Page title | `<title>` | Contains "Model Intelligence" |
| Alerts banner section | `section` containing "things need your attention" | Visible |
| Alert warning icon | `.material-symbols-outlined` with text "warning" inside alert section | Visible |
| "Your models" heading | `h2` with text "Your models" | Visible |
| Active instances counter | `span` with text matching "3 ACTIVE INSTANCES" | Visible (3 of 4 models are active) |
| Model cards (4 total) | Cards inside the grid | Count = 4 model cards + 1 "Add model" card |
| Active model cards (3) | Cards with green "Active" label | Count = 3 |
| Deprecated model card (1) | Card with gold "Deprecated" label | Count = 1 |
| "Add model" button card | Button with text "Add model" | Visible |
| "Explore models" heading | `h2` with text "Explore models" | Visible |
| Search input | `input[placeholder*="Search 2,450+ models"]` | Visible |
| Provider filter button | Button containing "Provider" | Visible |
| Capability filter button | Button containing "Capability" | Visible |
| Explore table | `table` element | Visible |
| Table headers | `th` elements | 5 headers: Model Name, Provider, Cost/1M, Context, Match % |
| Table rows | `tbody tr` | Count = 5 (gemini-2.5-flash, llama-4-maverick, claude-opus-4.6, mistral-large, deepseek-v3) |

### 1.3 Data-Driven Checks

**Models (from DUMMY_MODELS):**

| Model ID | Provider | Status | Context | Cost/1M | Extra |
|---|---|---|---|---|---|
| `openai/gpt-4o` | OpenAI | active | 128K | $2.50 | -- |
| `anthropic/claude-sonnet-4` | Anthropic | active | 200K | $3.00 | -- |
| `openai/gpt-4o-mini` | OpenAI | active | 128K | $0.15 | -- |
| `openai/gpt-4o-0613` | OpenAI | deprecated | 8K | $5.00 | retirement_date="Apr 15, 2026", replacement="openai/gpt-4o" |

Verify each model ID appears in the page body.

**Active model cards** should display: Context value, Cost/1M value, Provider name.
- Check: `page.locator("text=128K")` visible (appears twice for gpt-4o and gpt-4o-mini)
- Check: `page.locator("text=$2.50")` visible
- Check: `page.locator("text=$3.00")` visible
- Check: `page.locator("text=OpenAI")` visible
- Check: `page.locator("text=Anthropic")` visible

**Deprecated model card** should display:
- "Deprecated" label text
- Retirement date: "Retiring Apr 15, 2026"
- Recommended upgrade: "openai/gpt-4o (latest)"
- "Start migration" link

**Alerts banner (from DUMMY_ALERTS -- 3 alerts):**
- Heading: "3 things need your attention"
- Alert messages visible:
  - "Model retiring in 26 days"
  - "Price decreased 17%"
  - "New model available"

**Explore models table (hardcoded in template):**

| Model | Provider | Cost | Context |
|---|---|---|---|
| gemini-2.5-flash | Google | $0.10 | 1M+ |
| llama-4-maverick | Meta | $0.00 | 64K |
| claude-opus-4.6 | Anthropic | $15.00 | 200K |
| mistral-large | Mistral | $4.00 | 32K |
| deepseek-v3 | DeepSeek | $0.25 | 128K |

Verify each model name appears in a `td` element.

### 1.4 Navigation Checks
- Active model cards: "Run migration" link exists (href="#")
- Deprecated card: "Start migration" link exists (href="#")
- "Add model" button is clickable (no navigation, just verify exists)

### 1.5 Interactive Behavior Checks
- None specific to this page beyond cross-page (nav, theme toggle)

### 1.6 Visual/Styling Checks
- Active nav: "Models" nav link should have terracotta color (`color: rgb(212, 116, 94)` / `#D4745E`) and bottom border
- Deprecated card should have gold left border (`border-left-color` matching `#D4A574`)
- Active status dots should be green (emerald-500)
- Match % progress bars: check `div` with `style` attribute containing `width: 98%` etc.

### 1.7 Edge Cases
- None for this page (empty state is a separate route)

---

## 2. Models Empty State (`/ui/?empty=true`)

### 2.1 URL and Route
- **GET** `/ui/?empty=true`
- Route handler: `dashboard()` with `empty="true"` query param
- Template: `models_empty.html` extends `base.html`
- Context: `active_nav="models"` (no models/alerts data passed)

### 2.2 Elements to Verify Exist

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Page title | `<title>` | Contains "Welcome" |
| Hero heading | `h1` with text "Welcome to RosettaStone" | Visible |
| Subtitle | `p` with text "Let's set up your model landscape." | Visible |
| Onboarding card | `div` with heading "Which models are you currently using?" | Visible |
| Model input 1 | `input[placeholder="openai/gpt-4o"]` | Visible |
| Model input 2 | `input[placeholder="anthropic/claude-sonnet-4"]` | Visible |
| "+ Add another model" button | Button containing "+ Add another model" | Visible |
| LiteLLM hint | Text "We use LiteLLM model identifiers" | Visible |
| "Set up models" button | Button with text "Set up models" | Visible |
| Import link | Link with text containing "Or import from migration results" | Visible |
| Infrastructure section | Text "Or connect your infrastructure:" | Visible |
| "Connect Redis cache" button | Button containing "Connect Redis cache" | Visible |
| "Connect LangSmith" button | Button containing "Connect LangSmith" | Visible |

### 2.3 Data-Driven Checks
- No dynamic data -- all static content. Verify the static text strings above.

### 2.4 Navigation Checks
- Import link (`href="#"`)
- Active nav state should still be "models"

### 2.5 Interactive Behavior Checks
- Input fields should be focusable and accept text
  - **Trigger**: Click on input, type "test-model"
  - **Expected**: Input value changes to "test-model"
  - **Verify**: `expect(page.locator('input[placeholder="openai/gpt-4o"]')).to_have_value("test-model")`
- "Set up models" button should be clickable
  - **Trigger**: Click the button
  - **Expected**: No crash (button has no handler yet, just verify no error)

### 2.6 Visual/Styling Checks
- Background glow gradient (`.empty-glow` style defined in `<style>` block)
- "Set up models" button should have terracotta background (`background-color: #D4745E`)

### 2.7 Edge Cases
- Visiting `/ui/?empty=false` or any other value should render the normal dashboard (not empty state)

---

## 3. Migrations List (`/ui/migrations`)

### 3.1 URL and Route
- **GET** `/ui/migrations`
- Route handler: `migrations_page()` in `migrations.py`
- Template: `migrations.html` extends `base.html`
- Context: `migrations=DUMMY_MIGRATIONS` (fallback when DB empty), `active_nav="migrations"`

### 3.2 Elements to Verify Exist

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Page title | `<title>` | Contains "Migrations" |
| Page heading | `h1` with text "Migrations" | Visible |
| Subtitle | `p` containing "Evaluate and deploy" | Visible |
| "+ New migration" button | Button containing "+ New migration" | Visible |
| Migration cards | `.group.relative` divs within the list | Count = 3 |
| Pagination footer | Text "Showing 3 migrations" | Visible |
| Previous page button (disabled) | Button with `chevron_left` icon, `disabled` attribute | Visible, disabled |
| Next page button | Button with `chevron_right` icon | Visible |

### 3.3 Data-Driven Checks

**Migration 1 (Safe to ship):**
- Source/target: "gpt-4o" and "claude-sonnet-4" visible in the card
- Arrow entity between them
- Recommendation badge: text "Safe to ship"
- Confidence: "92% confidence"
- Test cases: "156 test cases"
- Cost: "$2.34"
- Time: "2 minutes ago"

**Migration 2 (Needs review):**
- Source/target: "gpt-4o-0613" and "gpt-4o"
- Recommendation badge: text "Needs review"
- Confidence: "78% confidence"
- Test cases: "43 test cases"
- Cost: "$1.12"
- Time: "1 day ago"

**Migration 3 (Do not ship):**
- Source/target: "gpt-3.5-turbo" and "gpt-4o-mini"
- Recommendation badge: text "Do not ship"
- Confidence: "61% confidence"
- Test cases: "89 test cases"
- Cost: "$0.87"
- Time: "3 days ago"

### 3.4 Navigation Checks
- Each migration card is wrapped in `<a href="/ui/migrations/{{ id }}">`:
  - Card 1: `a[href="/ui/migrations/1"]`
  - Card 2: `a[href="/ui/migrations/2"]`
  - Card 3: `a[href="/ui/migrations/3"]`
- Clicking card 1 should navigate to `/ui/migrations/1`
  - **Trigger**: `page.locator('a[href="/ui/migrations/1"]').click()`
  - **Expected**: URL changes to contain `/ui/migrations/1`
  - **Verify**: `expect(page).to_have_url(re.compile(r"/ui/migrations/1"))`

### 3.5 Interactive Behavior Checks
- Migration cards should be clickable (navigation -- covered above)

### 3.6 Visual/Styling Checks

**Left border colors by recommendation:**
- Safe to ship: left border `bg-[#8B9D83]` (sage green, `rgb(139, 157, 131)`)
  - Selector: First card's `div.absolute.left-0` with `background-color`
- Needs review: left border `bg-[#D4A574]` (gold, `rgb(212, 165, 116)`)
- Do not ship: left border `bg-[#D85650]` (critical red, `rgb(216, 86, 80)`)

**Badge colors:**
- "Safe to ship" badge: `bg-[#8B9D83] text-[#131313]`
- "Needs review" badge: `bg-[#D4A574] text-[#131313]`
- "Do not ship" badge: `bg-[#D85650] text-white`

**Icons by recommendation:**
- Safe to ship: `verified` icon
- Needs review: `warning` icon
- Do not ship: `cancel` icon

**Active nav:** "Migrations" link should have terracotta styling

### 3.7 Edge Cases
- Empty migrations state: When no migrations exist and DUMMY_MIGRATIONS is empty, should show "No migrations yet" empty state. (Note: current code always has DUMMY_MIGRATIONS as fallback.)

---

## 4. Migration Detail -- Safe to Ship (`/ui/migrations/1`)

### 4.1 URL and Route
- **GET** `/ui/migrations/1`
- Route handler: `migration_detail_page()` in `migrations.py`
- Template: `migration_detail.html` extends `base.html`, renders "Safe to ship" layout branch
- Context: `migration=DUMMY_MIGRATIONS[0]`, `active_nav="migrations"`

### 4.2 Elements to Verify Exist

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Page title | `<title>` | Contains "Migration Detail" |
| Breadcrumb link | `a[href="/ui/migrations"]` with text "Migrations" | Visible |
| Back arrow | `.material-symbols-outlined` with text "arrow_back" | Visible |
| Page heading | `h1` containing "gpt-4o" and "claude-sonnet-4" | Visible |
| "Complete" status badge | `span` with text "Complete" | Visible |
| "Version History" button | Button containing "Version History" | Visible |
| "Deploy to Prod" button | Button containing "Deploy to Prod" | Visible |
| Recommendation card | Section with "Safe to ship" heading and check_circle icon | Visible |
| Confidence score (big display) | Text "92%" in the recommendation card sidebar | Visible |
| "Confidence Score" label | Text "Confidence Score" | Visible |
| Reasoning text | Text containing "All output types meet or exceed" | Visible |
| "Export report" link | `a` with text "Export report" and href to `/ui/migrations/1/executive` | Visible |
| "View optimized prompt" button | Button with text "View optimized prompt" | Visible |
| KPI grid: Parity Score | "92%" with label "Parity Score" | Visible |
| KPI grid: Baseline | "85%" with label "Baseline" | Visible |
| KPI grid: Improvement | "+7%" with label "Improvement from GEPA" | Visible |
| "Results by output type" heading | `h3` with text "Results by output type" | Visible |
| Per-type cards (4) | Cards in results grid | Count = 4 |
| "Needs Attention" section | Section header "Needs Attention" | Visible |
| Critical count badge | Text "3 CRITICAL" | Visible |
| Regression entries (3) | Regression items | Count = 3 |
| "Worst regression" label | Text "Worst regression" (on first regression only) | Visible |
| "View diff" buttons | Buttons with text "View diff" | Count = 3 |
| "Configuration and details" collapsible | Button with text "Configuration and details" | Visible |
| "Model Metadata" section | Section with heading "Model Metadata" | Visible |

### 4.3 Data-Driven Checks

**Migration data (id=1):**
- source: "gpt-4o", target: "claude-sonnet-4"
- recommendation: "Safe to ship"
- confidence: 92, baseline: 85, improvement: 7
- reasoning: "All output types meet or exceed quality thresholds..."
- cost: "$2.34", test_cases: 156

**Per-type results:**

| Type | Wins/Total | Badge | Description |
|---|---|---|---|
| JSON | 48/48 (100%) | PASS | "All fields match" |
| Text | 89/96 (92%) | PASS | "Strong semantic match" |
| Code | 5/6 (83%) | WARN | "Low sample size (6)" |
| Classification | 4/6 (66%) | WARN | "Low sample size (6)" |

Verify each type name, wins/total ratio, and badge text appears.

**Regressions:**

| tc_id | Score | Title | Expected | Got |
|---|---|---|---|---|
| 42 | 0.31 | Priority classification mismatch | urgent | high_priority |
| 87 | 0.45 | Truncated JSON response | {"status": "complete", ...} | {"status": "done"} |
| 103 | 0.52 | Different code formatting | def foo(): | def foo() ->None: |

Verify regression titles, expected/got text pairs.

**Configuration section (collapsed by default):**
- Source: gpt-4o
- Target: claude-sonnet-4
- Test Cases: 156
- Cost: $2.34

**Model Metadata (hardcoded in template):**
- Context Window: 200k tokens
- Tokenizer: Claude v4 Native
- Provider: Anthropic

### 4.4 Navigation Checks
- Breadcrumb: Clicking "Migrations" navigates to `/ui/migrations`
  - **Trigger**: `page.locator('a[href="/ui/migrations"]').first.click()`
  - **Expected**: URL becomes `/ui/migrations`
- "Export report" link: `a[href="/ui/migrations/1/executive"]`
  - **Trigger**: Click the link
  - **Expected**: Navigates to executive report page

### 4.5 Interactive Behavior Checks

**Collapsible "Configuration and details":**
- **Initial state**: Content hidden (`max-height: 0`, no `expanded` class on `#config-details`)
- **Trigger**: Click the button `[data-action="toggle-collapse"][data-target="config-details"]`
- **Expected result**: `#config-details` gains `expanded` class, content becomes visible
- **Verify**:
  ```python
  config_btn = page.locator('[data-action="toggle-collapse"][data-target="config-details"]')
  config_content = page.locator('#config-details')
  # Before click: aria-expanded="false"
  expect(config_btn).to_have_attribute("aria-expanded", "false")
  config_btn.click()
  # After click: aria-expanded="true", content has "expanded" class
  expect(config_btn).to_have_attribute("aria-expanded", "true")
  expect(config_content).to_have_class(re.compile(r"expanded"))
  # Verify content is visible
  expect(page.locator('#config-details >> text=Source')).to_be_visible()
  ```
- **Toggle back**: Click again, aria-expanded="false", expanded class removed

**HTMX "View diff" buttons (3 regression entries):**
- Each button has `hx-get="/ui/fragments/diff/1/{tc_id}"` `hx-target="#diff-content"` `hx-swap="innerHTML"`
- **Trigger**: Click the first "View diff" button
- **Expected**:
  1. HTMX fires GET to `/ui/fragments/diff/1/42`
  2. Response HTML swapped into `#diff-content`
  3. `htmx:afterSwap` event fires
  4. `app.js` event listener calls `openSlideout()`
  5. `#diff-panel` gets classes `open` (removes `closed`, removes `hidden`)
  6. `#diff-backdrop` becomes visible (removes `hidden`)
  7. `body` gets `overflow: hidden`
- **Verify**:
  ```python
  page.locator('button:has-text("View diff")').first.click()
  page.wait_for_load_state("networkidle")
  # Panel should be visible
  expect(page.locator('#diff-panel')).to_be_visible()
  expect(page.locator('#diff-panel')).to_have_class(re.compile(r"open"))
  expect(page.locator('#diff-backdrop')).to_be_visible()
  # Diff content should contain expected data
  expect(page.locator('#diff-content')).to_contain_text("Classification")
  expect(page.locator('#diff-content')).to_contain_text("0.72")
  ```

### 4.6 Visual/Styling Checks
- Recommendation card left border: green/success (`border-success` class)
- check_circle icon with FILL 1 (filled)
- "Safe to ship" heading in success color
- KPI values in success color for confidence and improvement
- Per-type PASS badges: green (`bg-success/10 text-success border-success/30`)
- Per-type WARN badges: gold/warn (`bg-warn/10 text-warn border-warn/30`)
- Code and Classification cards should have left border (`border-l-4 border-warn`)

### 4.7 Edge Cases
- What if per_type is empty: The "Results by output type" section should not render (`{% if migration.per_type %}` guard)
- What if regressions is empty: "Needs Attention" section should not render

---

## 5. Migration Detail -- Needs Review (`/ui/migrations/2`)

### 5.1 URL and Route
- **GET** `/ui/migrations/2`
- Template: `migration_detail.html`, renders "Needs review / default" layout branch (the `{% else %}` block)
- Context: `migration=DUMMY_MIGRATIONS[1]`

### 5.2 Elements to Verify Exist

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Page heading | `h1` containing "gpt-4o-0613" and "gpt-4o" | Visible |
| "Review" status badge | `span` with text "Review" | Visible |
| Recommendation card | Card with "Needs review" heading and warning icon | Visible |
| Recommendation text | Text "Needs review" in large heading | Visible |
| Warning icon | `.material-symbols-outlined` with text "warning" (in recommendation card) | Visible |
| Reasoning | Text containing "classification accuracy dropped 8%" | Visible |
| "View full report" link | `a[href="/ui/migrations/2/executive"]` | Visible |
| "Review edge cases" button | Button with text "Review edge cases" | Visible |
| KPI: Confidence | "78%" | Visible |
| KPI: Baseline | "72%" | Visible |
| KPI: Improvement | "+6%" | Visible |

### 5.3 Data-Driven Checks
- source: "gpt-4o-0613", target: "gpt-4o"
- recommendation: "Needs review"
- confidence: 78, baseline: 72, improvement: 6
- per_type: empty list (should NOT render "Results by output type")
- regressions: empty list (should NOT render "Regressions to review")
- cost: "$1.12", test_cases: 43

### 5.4 Navigation Checks
- Breadcrumb back to `/ui/migrations`
- "View full report" links to `/ui/migrations/2/executive`

### 5.5 Interactive Behavior Checks
- No collapsible sections (not in this layout branch)
- No HTMX diff buttons (no regressions)

### 5.6 Visual/Styling Checks
- Recommendation card: left border `border-[#D4A574]` (gold)
- "Needs review" heading color: `text-[#D4A574]`
- KPI confidence color: `text-[#D4A574]`
- Warning icon color: `text-[#D4A574]`
- "Review" badge: `bg-[#D4A574]/10 text-[#D4A574] border-[#D4A574]/30`

### 5.7 Edge Cases
- **No per_type section rendered**: Verify `"Results by output type"` does NOT appear
  ```python
  expect(page.locator('text=Results by output type')).to_have_count(0)
  ```
- **No regressions section rendered**: Verify `"Regressions to review"` does NOT appear
  ```python
  expect(page.locator('text=Regressions to review')).to_have_count(0)
  ```

---

## 6. Migration Detail -- Do Not Ship (`/ui/migrations/3`)

### 6.1 URL and Route
- **GET** `/ui/migrations/3`
- Template: `migration_detail.html`, renders "Do not ship" layout branch
- Context: `migration=DUMMY_MIGRATIONS[2]`

### 6.2 Elements to Verify Exist

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Page heading | `h1` containing "gpt-3.5-turbo" and "gpt-4o-mini" with trending_flat icon | Visible |
| "FAILED" badge | `span` with text "FAILED" | Visible |
| "Do not ship" heading | `h2` with text "Do not ship" | Visible |
| Cancel icon | `.material-symbols-outlined` with text "cancel" | Visible |
| Reasoning | Text containing "Critical schema violations in 16%" | Visible |
| "View full report" link | `a[href="/ui/migrations/3/executive"]` | Visible |
| "Adjust optimization settings" button | Button with text "Adjust optimization settings" | Visible |
| KPI: Confidence | "61%" | Visible |
| KPI: Baseline | "58%" | Visible |
| KPI: Improvement | "+3%" | Visible |
| "Results by output type" heading | `h3` with text "Results by output type" | Visible |
| Per-type cards (4) | Results cards | Count = 4 |
| "Critical regressions" heading | `h3` with text "Critical regressions" | Visible |
| Regression count | Text "Showing 3 of 3" | Visible |
| Regression entries (3) | Regression rows | Count = 3 |
| Warning icons in regressions | `.material-symbols-outlined` with text "warning" in red | Count = 3 |
| "Last Evaluated" label | Text "Last Evaluated" | Visible |
| Time ago | Text "3 days ago" | Visible |

### 6.3 Data-Driven Checks

**Header differs from other layouts:**
- Uses `font-mono text-xl` for source/target
- Has `trending_flat` icon (not arrow entity)
- "FAILED" badge with `bg-error/20 text-error`

**Per-type results (id=3):**

| Type | Wins/Total | Badge | Description |
|---|---|---|---|
| JSON | 30/42 (71%) | FAIL | "Schema violations in 12 cases" |
| Text | 22/28 (78%) | WARN | "Semantic drift detected" |
| Code | 3/11 (27%) | FAIL | "Syntax errors in 8 outputs" |
| Classification | 5/8 (62%) | FAIL | "37% accuracy drop" |

**Regressions (id=3):**

| tc_id | Score | Title | Expected | Got |
|---|---|---|---|---|
| 7 | 0.12 | Schema Violation: Missing required key 'metadata' | {"data": ..., "metadata": ...} | {"data": ...} |
| 19 | 0.22 | Syntax error in generated Python | valid Python function | SyntaxError on line 3 |
| 31 | 0.28 | Wrong classification category | billing_dispute | general_inquiry |

**HTMX "View diff" links in regressions:**
- Each regression row has an `<a>` with `hx-get="/ui/fragments/diff/3/{tc_id}"`
- Note: These links have `opacity-0 group-hover:opacity-100` -- only visible on hover

### 6.4 Navigation Checks
- Breadcrumb back to `/ui/migrations`
- "View full report" links to `/ui/migrations/3/executive`

### 6.5 Interactive Behavior Checks

**HTMX diff slideout (on hover + click):**
- **Trigger**: Hover over first regression row, then click "View diff" link
- **Expected**: Slideout panel opens with diff fragment content
- **Verify**:
  ```python
  first_reg = page.locator('.group:has-text("Schema Violation")').first
  first_reg.hover()
  # "View diff" link becomes visible on hover
  diff_link = first_reg.locator('a:has-text("View diff")')
  expect(diff_link).to_be_visible()
  diff_link.click()
  page.wait_for_load_state("networkidle")
  expect(page.locator('#diff-panel')).to_be_visible()
  ```

### 6.6 Visual/Styling Checks
- Recommendation card: `border-l-4 border-[#D85650]` (red)
- "Do not ship" heading: `text-[#D85650]`
- Cancel icon: `text-[#D85650]`
- "FAILED" badge: `bg-error/20 text-error border border-error/30`
- KPI confidence: `text-[#D85650]`
- Per-type FAIL badges: `bg-[#D85650]/10 text-[#D85650] border border-[#D85650]/20`
- Per-type FAIL cards have: `border-l-4 border-[#D85650]`
- Per-type WARN cards have: `border-l-4 border-tertiary-container`
- Warning icons in regressions: `text-[#D85650]`

### 6.7 Edge Cases
- "View diff" links are hidden by default (`opacity-0`) -- test must hover first

---

## 7. Costs Overview (`/ui/costs`)

### 7.1 URL and Route
- **GET** `/ui/costs`
- Route handler: `costs_page()` in `migrations.py`
- Template: `costs.html` extends `base.html`
- Context: `costs=DUMMY_COSTS`, `active_nav="costs"`

### 7.2 Elements to Verify Exist

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Page title | `<title>` | Contains "Costs" |
| Page heading | `h1` with text "Cost overview" | Visible |
| Time period dropdown | Button with text "This month" | Visible |
| KPI: Estimated | Text "$1,247" with label "Estimated this month" | Visible |
| KPI trend | Text "+12.4% vs last month" | Visible |
| KPI: Savings | Text "$312" with label "Potential savings" | Visible |
| KPI savings subtext | Text "2 actionable optimizations" | Visible |
| KPI: After optimization | Text "$935" with label "After optimization" | Visible |
| KPI optimization subtext | Text "Target budget reached" | Visible |
| "Cost by model" heading | `h2` with text "Cost by model" | Visible |
| Model cost bars (3) | Bar chart rows | Count = 3 |
| "Optimization opportunities" heading | `h2` with text "Optimization opportunities" | Visible |
| Opportunity cards (2) | Opportunity entries | Count = 2 |

### 7.3 Data-Driven Checks

**DUMMY_COSTS values:**

KPIs:
- total_month: "$1,247"
- potential_savings: "$312"
- after_optimization: "$935"

Cost by model:

| Model | Cost | Percentage |
|---|---|---|
| gpt-4o | $823 | 66% |
| claude-sonnet-4 | $312 | 25% |
| gpt-4o-mini | $112 | 9% |

Verify model names, cost values, percentage values appear in the page.
Bar widths: verify `style` contains `width: 66%`, `width: 25%`, `width: 9%`.
Bar colors rotate: `#D4745E`, `#8B9D83`, `#D4A574` (via `bar_colors` template variable).

Optimization opportunities:

| Title | Savings | Confidence |
|---|---|---|
| Switch gpt-4o classification to gpt-4o-mini | $187/mo | 94% parity |
| Batch non-urgent gpt-4o requests | $125/mo | No quality impact |

Verify opportunity titles, savings amounts, and confidence notes.
- First opportunity: lightbulb icon
- Second opportunity: bolt icon

### 7.4 Navigation Checks
- "Evaluate this" links on each opportunity (href="#")
- Active nav: "Costs" link styled in terracotta

### 7.5 Interactive Behavior Checks
- "This month" dropdown button exists but no dropdown menu is implemented (just verify button present)

### 7.6 Visual/Styling Checks
- KPI cards use `rounded-2xl` with deep shadow
- Savings value: `text-[#8B9D83]` (sage green)
- Opportunity cards: sage green left border (`border-l-4 border-[#8B9D83]`)
- "Evaluate this" link color: `text-[#D4745E]`

### 7.7 Edge Cases
- None specific (always uses DUMMY_COSTS)

---

## 8. Alerts Hub (`/ui/alerts`)

### 8.1 URL and Route
- **GET** `/ui/alerts`
- Route handler: `alerts_page()` in `migrations.py`
- Template: `alerts.html` extends `base.html`
- Context: `alerts=DUMMY_ALERTS`, `active_nav="alerts"`

### 8.2 Elements to Verify Exist

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Page title | `<title>` | Contains "Alerts" |
| Hero warning icon | Filled `warning` icon | Visible |
| "System Priority: Omega" label | Text "System Priority: Omega" | Visible |
| Alert count heading | `h1` containing "3 Critical Alerts" | Visible |
| "ACTION REQUIRED." text | Text "ACTION REQUIRED." in red | Visible |
| Description paragraph | Text containing "RosettaStone migration protocols" | Visible |
| "Start Batch Migration" button | Button with text "Start Batch Migration" | Visible |
| "View Drift Logs" button | Button with text "View Drift Logs" | Visible |
| "Active Intelligence Reports" heading | `h2` with text "Active Intelligence Reports" | Visible |
| Subtitle | Text "Real-time heuristic monitoring" | Visible |
| "Export Report Data" button | Button with text "Export Report Data" | Visible |
| Alert cards (3) | Alert row cards | Count = 3 |
| "Notification Settings" heading | `h3` with text "Notification Settings" | Visible |
| Checkbox 1: Critical System Alerts | Checkbox, checked | Visible, checked |
| Checkbox 2: Cost & Efficiency Reports | Checkbox, unchecked | Visible, not checked |
| Checkbox 3: Model Performance Drift | Checkbox, checked | Visible, checked |
| "Save Preferences" button | Button with text "Save Preferences" | Visible |

### 8.3 Data-Driven Checks

**DUMMY_ALERTS (3 alerts):**

| Alert # | Type | Model | Key Info |
|---|---|---|---|
| 1 | deprecation | gpt-4o-0613 | "Deprecation Warning: gpt-4o-0613", "Model retiring in 26 days", "Start migration to gpt-4o", "26 days remaining" |
| 2 | price_change | claude-sonnet-4 | "Price Change: claude-sonnet-4", "Price decreased 17%", "No action needed", "$3.00 -> $2.50" |
| 3 | new_model | claude-opus-4.6 | "New Model Available: claude-opus-4.6", "New model available", "+12% reasoning improvement...", "Available" |

Verify each alert card renders the correct title, message, action text, and timeline/impact/status data.

**Per-type rendering in alert cards:**
- Deprecation: red icon (`text-[#D85650]`), `history_toggle_off` icon, red border (`border-[#D85650]`), TIMELINE label, "26 days remaining" in red
- Price change: gold icon (`text-[#D4A574]`), `payments` icon, gold border (`border-[#D4A574]`), IMPACT label, "$3.00 -> $2.50" in gold
- New model: sage icon (`text-[#8B9D83]`), `model_training` icon, sage border (`border-[#8B9D83]`), STATUS label, "Available" in sage

### 8.4 Navigation Checks
- Active nav: "Alerts" link styled in terracotta
- Arrow icon appears on hover for each alert card (opacity transition)

### 8.5 Interactive Behavior Checks

**Checkbox toggling:**
- **Trigger**: Click the unchecked "Cost & Efficiency Reports" checkbox
- **Expected**: Checkbox becomes checked
- **Verify**: `expect(page.locator('input[type="checkbox"]').nth(1)).to_be_checked()`

**Forward arrow on hover:**
- **Trigger**: Hover over first alert card
- **Expected**: `arrow_forward` icon becomes visible (`opacity: 1`)
- **Verify**:
  ```python
  first_alert = page.locator('.group:has-text("Deprecation Warning")').first
  arrow = first_alert.locator('.material-symbols-outlined:has-text("arrow_forward")')
  # Before hover: opacity 0
  first_alert.hover()
  # After hover: opacity 100 (visible)
  expect(arrow).to_be_visible()
  ```

### 8.6 Visual/Styling Checks
- Hero section has gradient background (`bg-gradient-to-br from-error-container/20 to-transparent`)
- "Start Batch Migration" button: gradient terracotta background
- "View Drift Logs" button: ghost border style
- Notification settings section: different background (`bg-surface-container-lowest`)

### 8.7 Edge Cases
- None specific (always uses DUMMY_ALERTS)

---

## 9. Executive Report (`/ui/migrations/1/executive`)

### 9.1 URL and Route
- **GET** `/ui/migrations/{id}/executive`
- Route handler: `executive_report_page()` in `migrations.py`
- Template: `executive_report.html` (standalone, does NOT extend `base.html`)
- Context: `migration=DUMMY_MIGRATIONS[0]`, `report_date=<current date>`

### 9.2 Elements to Verify Exist

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Page title | `<title>` | "Executive Migration Report" |
| Report label | Text "ROSETTASTONE MIGRATION REPORT" | Visible |
| Source/target heading | `h1` containing "gpt-4o" and "claude-sonnet-4" | Visible |
| Report date | Current date in "Month Day, Year" format | Visible |
| Recommendation heading | `h2` containing "Recommendation: Safe to switch" | Visible |
| Check circle icon | `.material-symbols-outlined` with text "check_circle" (filled) | Visible |
| Reasoning text | Text containing "All output types meet or exceed" | Visible |
| Metric box: Quality match | "92%" with label "Quality match" | Visible |
| Metric box: Cost | "$2.34" with label "Cost" | Visible |
| Metric box: Risk level | "Low" with label "Risk level" (92 >= 85) | Visible |
| Metric box: Deployment status | "Ready" with label "Deployment status" | Visible |
| "What improves" section | Heading "What improves" with sage dot | Visible |
| Improvement bullets | "92% parity across 156 test cases", "+7% improvement from GEPA", "Baseline score: 85%" | All visible |
| "What to watch" section | Heading "What to watch" with gold dot | Visible |
| Regression bullets (3) | Max 3 regressions from migration data | Visible |
| Per-type results section | "Results by Output Type" heading | Visible |
| Per-type rows (4) | Rows for JSON, Text, Code, Classification | Count = 4 |
| Version label | "V.04.2-STABLE" | Visible |
| Footer: "Prepared by" | "Prepared by RosettaStone v0.1.0" | Visible |
| Footer: "CONFIDENTIAL" | Text "CONFIDENTIAL" | Visible |
| Report canvas styling | `.report-canvas` div | Visible |

### 9.3 Data-Driven Checks

**This is a standalone page (no nav, no footer from base.html):**
- No navigation bar
- No base.html footer
- Has its own report-specific footer

**Metric boxes (conditional logic):**
- Risk level: confidence 92 >= 85 -> "Low"
- Risk level color: `text-sage` (because "Safe to ship")
- Deployment status: "Ready" (because "Safe to ship")

**"What to watch" section regressions (max 3 from `migration.regressions[:3]`):**
- "Priority classification mismatch (score: 0.31)"
- "Truncated JSON response (score: 0.45)"
- "Different code formatting (score: 0.52)"

**Per-type results:**
- JSON: 48/48 (PASS) -- text-sage
- Text: 89/96 (PASS) -- text-sage
- Code: 5/6 (WARN) -- text-[#D4A574]
- Classification: 4/6 (WARN) -- text-[#D4A574]

### 9.4 Navigation Checks
- This is a standalone page -- no nav links to verify
- No links to other pages (self-contained report)

### 9.5 Interactive Behavior Checks
- None -- this is a print-ready static page

### 9.6 Visual/Styling Checks
- **Light background**: `body` has `background-color: #F9F7F4` and `color: #2C2C2C` (inverted from dark default)
- Report canvas: `max-width: 8.5in`, white background, subtle shadow
- Print media query: `@media print` removes shadow, sets white bg
- `.text-sage` class: `color: #8B9D83`
- `.sage-dot` / `.gold-dot` indicator dots
- Border dividers: `border-color: #E0E0E0`
- Top gradient bar: fixed at top of viewport

### 9.7 Edge Cases
- Test `/ui/migrations/3/executive` (Do not ship):
  - Recommendation: "Recommendation: Do not switch"
  - Cancel icon instead of check_circle
  - Risk level: 61 < 70 -> "High"
  - Deployment status: "Blocked"
- Test `/ui/migrations/2/executive` (Needs review):
  - Recommendation: "Recommendation: Needs review"
  - Warning icon
  - Risk level: 78 >= 70 and < 85 -> "Medium"
  - Deployment status: "Pending"
  - "What to watch": "No critical regressions detected" (empty regressions list)
- Test `/ui/migrations/999/executive` -> 404

---

## 10. Diff Slideout Fragment (`/ui/fragments/diff/1/42`)

### 10.1 URL and Route
- **GET** `/ui/fragments/diff/{migration_id}/{tc_id}`
- Route handler: `diff_fragment()` in `comparisons.py`
- Template: `fragments/diff_slideout.html` (no base.html, HTMX fragment)
- Context: `diff=DUMMY_DIFF` (fallback when DB has no matching records)

### 10.2 Elements to Verify Exist (within the fragment HTML)

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Output type badge | `span` with text "Classification" | Present |
| Composite score (large) | `h2` with text "0.72" | Present |
| WIN/LOSS badge | `span` with text "LOSS" (is_win=False) | Present |
| Close button (X) | `button[data-action="close-slideout"]` with close icon | Present |
| BERTScore label + value | "BERTScore" label, value "0.85" | Present |
| BERTScore progress bar | Bar with `style` `width: 85%` | Present |
| Embedding similarity label + value | "Embedding similarity" label, value "0.79" | Present |
| Embedding bar | Bar with `style` `width: 79%` | Present |
| Composite label + value | "Composite" label, value "0.72" | Present |
| Composite bar | Bar with `style` `width: 72%` | Present |
| Tab: "Side-by-side" | Button text "Side-by-side" (active) | Present |
| Tab: "Unified" | Button text "Unified" (inactive) | Present |
| PII Warning banner | Text "Content may contain sensitive data" | Present |
| Diff view: "Expected" header | Text containing "Expected (gpt-4o)" | Present |
| Diff view: "Actual" header | Text containing "Actual (claude-sonnet-4)" | Present |
| Expected content | JSON with `"priority": "urgent"` | Present |
| Actual content | JSON with `"priority": "high_priority"` | Present |
| Test Case Evidence section | "Test Case Evidence" heading | Present |
| Evidence: Test Case ID | "#42" | Present |
| Evidence: Output Type | "Classification" | Present |
| Evidence: Composite Score | "0.72" | Present |
| Footer: CLOSE button | `button[data-action="close-slideout"]` with text "CLOSE" | Present |
| Footer: APPROVE VARIANCE button | Button with text "APPROVE VARIANCE" | Present |

### 10.3 Data-Driven Checks

**DUMMY_DIFF values:**
```python
{
    "tc_id": 42,
    "is_win": False,
    "composite_score": 0.72,
    "output_type": "Classification",
    "scores": {"bertscore": 0.85, "embedding": 0.79, "composite": 0.72},
    "source_model": "gpt-4o",
    "target_model": "claude-sonnet-4",
    "expected": '{\n  "priority": "urgent",\n  "category": "billing",\n  "confidence": 0.94\n}',
    "actual": '{\n  "priority": "high_priority",\n  "category": "billing",\n  "confidence": 0.91\n}',
}
```

- LOSS badge (not WIN) because `is_win=False`
- LOSS badge has `bg-[#D85650] text-white`
- Progress bars: BERTScore 85%, Embedding 79%, Composite 72%
- Expected/Actual JSON content displayed in side-by-side grid

### 10.4 Navigation Checks
- No page navigation (this is a fragment loaded into a panel)

### 10.5 Interactive Behavior Checks

**Close button (X icon at top):**
- **Trigger**: Click `button[data-action="close-slideout"]` (top right)
- **Expected**: `closeSlideout()` called, panel animates closed
- **Verify**: Panel gets `closed` class, then becomes `hidden`

**Close button (footer CLOSE):**
- Same behavior as X button

**Testing as standalone fragment:**
- This fragment is designed to be loaded via HTMX, but can be tested by direct GET
- `response = page.goto("/ui/fragments/diff/1/42")` -- renders raw HTML
- Better to test integrated with the slideout panel from a migration detail page

### 10.6 Visual/Styling Checks
- LOSS badge: `bg-[#D85650] text-white`
- Composite score: large gold text (`text-[#D4A574]`)
- BERTScore bar: `bg-[#8B9D83]/40` (dim sage)
- Embedding bar: `bg-[#8B9D83]` (full sage)
- Composite bar: `bg-[#D4A574]` (gold)
- PII warning: gold border-left, gold text
- Side-by-side tab active: `border-b-2 border-[#D4745E]`
- Unified tab inactive: `text-on-surface/40`

### 10.7 Edge Cases
- **Content not stored**: When `diff.expected` equals `'Content not stored (run with --store-prompt-content)'`:
  - PII warning should NOT appear
  - Diff view should NOT appear
  - Instead show: "Content not stored" message with `visibility_off` icon
  - Show hint: `--store-prompt-content` code element
  - (To test this, would need a test case where content is null)

---

## 11. Cross-Page: Navigation Bar

### 11.1 Test on Every Page

Test the nav bar on each page: `/ui/`, `/ui/migrations`, `/ui/migrations/1`, `/ui/costs`, `/ui/alerts`.

### 11.2 Elements to Verify

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Nav bar | `nav` element | Fixed position, visible |
| Logo/brand | Text "RosettaStone" | Visible |
| "Models" link | `a[href="/ui/"]` with text "Models" | Visible |
| "Migrations" link | `a[href="/ui/migrations"]` with text "Migrations" | Visible |
| "Costs" link | `a[href="/ui/costs"]` with text "Costs" | Visible |
| "Alerts" link | `a[href="/ui/alerts"]` with text "Alerts" | Visible |
| Alerts notification dot | `span` with red dot near Alerts link (`bg-[#D85650]`) | Visible |
| Theme toggle button | `button[data-action="toggle-theme"]` | Visible |
| Settings button | Button with "settings" icon | Visible |
| Avatar/account icon | Element with "account_circle" icon | Visible |

### 11.3 Active State per Page

For each page, the corresponding nav link should have the terracotta active style (`text-[#D4745E] border-b-2 border-[#D4745E]`):

| Page | Active Link |
|---|---|
| `/ui/` | Models |
| `/ui/?empty=true` | Models |
| `/ui/migrations` | Migrations |
| `/ui/migrations/1` | Migrations |
| `/ui/migrations/2` | Migrations |
| `/ui/migrations/3` | Migrations |
| `/ui/costs` | Costs |
| `/ui/alerts` | Alerts |

**Verify**:
```python
# On /ui/costs:
models_link = page.locator('a[href="/ui/"]')
costs_link = page.locator('a[href="/ui/costs"]')
# Models link should NOT have terracotta border
expect(models_link).not_to_have_class(re.compile(r'border-b-2'))
# Costs link SHOULD have terracotta color
expect(costs_link).to_have_css("color", "rgb(212, 116, 94)")
```

### 11.4 Navigation Functionality

For each nav link:
- **Trigger**: Click the link
- **Expected**: Page navigates to the correct URL
- **Verify**: `expect(page).to_have_url(expected_url)`

Test matrix (from `/ui/`):
1. Click "Migrations" -> URL becomes `/ui/migrations`
2. Click "Costs" -> URL becomes `/ui/costs`
3. Click "Alerts" -> URL becomes `/ui/alerts`
4. Click "Models" -> URL becomes `/ui/`

### 11.5 Note on Executive Report
- The executive report page (`/ui/migrations/1/executive`) does NOT include the nav bar (standalone template, no base.html extension). Verify this:
  ```python
  page.goto(base_url + "/ui/migrations/1/executive")
  expect(page.locator('nav')).to_have_count(0)
  ```

---

## 12. Cross-Page: Theme Toggle

### 12.1 Default State
- Default theme is "dark" (set in `<html class="dark" data-theme="dark">`)
- `data-theme="dark"` on `<html>` element
- `<html>` has `dark` class

### 12.2 Toggle to Light Mode

- **Trigger**: Click `button[data-action="toggle-theme"]`
- **Expected**:
  1. `<html>` element gets `data-theme="light"`
  2. `<html>` element loses `dark` class
  3. `localStorage` key `rosettastone-theme` set to `"light"`
- **Verify**:
  ```python
  page.goto(base_url + "/ui/")
  # Default dark
  expect(page.locator('html')).to_have_attribute("data-theme", "dark")
  expect(page.locator('html')).to_have_class(re.compile(r"dark"))

  # Click toggle
  page.locator('[data-action="toggle-theme"]').click()

  # Now light
  expect(page.locator('html')).to_have_attribute("data-theme", "light")
  # "dark" class removed
  html_classes = page.locator('html').get_attribute("class")
  assert "dark" not in html_classes

  # localStorage updated
  stored = page.evaluate("() => localStorage.getItem('rosettastone-theme')")
  assert stored == "light"
  ```

### 12.3 Toggle Back to Dark Mode

- **Trigger**: Click theme toggle button again
- **Expected**: `data-theme="dark"`, `dark` class restored, localStorage = "dark"

### 12.4 Persistence Across Navigation

- **Trigger**: Toggle to light mode, then navigate to `/ui/costs`
- **Expected**: Page loads with `data-theme="light"` (app.js reads localStorage on load)
- **Verify**:
  ```python
  page.goto(base_url + "/ui/")
  page.locator('[data-action="toggle-theme"]').click()
  expect(page.locator('html')).to_have_attribute("data-theme", "light")

  page.goto(base_url + "/ui/costs")
  expect(page.locator('html')).to_have_attribute("data-theme", "light")
  ```

### 12.5 Visual Verification in Light Mode

After toggling to light mode, verify CSS overrides apply:
- `body` background changes from dark to light (verify computed style)
  ```python
  bg = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
  # Should be white-ish, not #131313
  ```
- Nav bar background changes (white-ish instead of dark)
- Footer background changes

### 12.6 Test on Multiple Pages
- Repeat toggle test on: `/ui/`, `/ui/migrations`, `/ui/costs`, `/ui/alerts`, `/ui/migrations/1`

---

## 13. Cross-Page: Footer

### 13.1 Elements to Verify

On every page that extends `base.html`:

| Element | Selector / Strategy | Assertion |
|---|---|---|
| Footer element | `footer` | Visible |
| Copyright text | Text containing "2024 RosettaStone Intelligence" | Visible |
| "Documentation" link | `a` with text "Documentation" | Visible |
| "API Reference" link | `a` with text "API Reference" | Visible |
| "Status" link | `a` with text "Status" | Visible |
| "Privacy" link | `a` with text "Privacy" | Visible |

### 13.2 Footer styling
- Background: `bg-[#0e0e0e]`
- Border top: `border-t border-[#E5E2E1]/10`
- Copyright text: JetBrains Mono, 10px, uppercase, tracking-widest
- Link text: same style, with hover color change to `#D4745E`

### 13.3 Verify Footer Absent on Executive Report
- `/ui/migrations/1/executive` should NOT have the base.html footer (has its own report footer)
- Verify: No element matching the "2024 RosettaStone Intelligence" text

### 13.4 Test on Multiple Pages
- Verify footer on: `/ui/`, `/ui/?empty=true`, `/ui/migrations`, `/ui/migrations/1`, `/ui/costs`, `/ui/alerts`

---

## 14. Error Handling: 404 for `/ui/migrations/999`

### 14.1 URL and Route
- **GET** `/ui/migrations/999`
- Route handler: `migration_detail_page()` raises `HTTPException(status_code=404)`
- The dummy data fallback looks for `id=999` in `DUMMY_MIGRATIONS`, finds nothing, raises 404

### 14.2 Expected Behavior
- HTTP status code: 404
- Response body contains `"detail": "Migration not found"` (FastAPI default JSON error)
  ```python
  response = page.goto(base_url + "/ui/migrations/999")
  assert response.status == 404
  ```

### 14.3 Additional 404 Tests

| URL | Expected Status | Reason |
|---|---|---|
| `/ui/migrations/999` | 404 | Migration not in DUMMY_MIGRATIONS |
| `/ui/migrations/9999` | 404 | Same |
| `/ui/migrations/999/executive` | 404 | Executive report for missing migration |
| `/ui/migrations/0` | 404 | ID 0 not in DUMMY_MIGRATIONS |
| `/ui/migrations/-1` | 422 or 404 | Negative ID |

---

## 15. Slideout Panel Integration (Full Lifecycle)

This is a dedicated integration test combining migration detail + diff fragment + slideout panel.

### 15.1 Full Open-Close Lifecycle

```python
def test_slideout_full_lifecycle(page, server):
    page.goto(server + "/ui/migrations/1")

    # 1. Panel starts hidden
    expect(page.locator('#diff-panel')).to_be_hidden()
    expect(page.locator('#diff-backdrop')).to_be_hidden()

    # 2. Click "View diff" button
    page.locator('button:has-text("View diff")').first.click()
    page.wait_for_load_state("networkidle")

    # 3. Panel opens
    expect(page.locator('#diff-panel')).to_be_visible()
    expect(page.locator('#diff-panel')).to_have_class(re.compile(r"open"))
    expect(page.locator('#diff-backdrop')).to_be_visible()

    # 4. Body scroll locked
    body_overflow = page.evaluate("() => document.body.style.overflow")
    assert body_overflow == "hidden"

    # 5. Content loaded
    expect(page.locator('#diff-content')).to_contain_text("BERTScore")
    expect(page.locator('#diff-content')).to_contain_text("Classification")

    # 6. Close via X button
    page.locator('[data-action="close-slideout"]').first.click()
    # Wait for animation (300ms)
    page.wait_for_timeout(400)
    expect(page.locator('#diff-panel')).to_be_hidden()
    expect(page.locator('#diff-backdrop')).to_be_hidden()

    # 7. Body scroll restored
    body_overflow = page.evaluate("() => document.body.style.overflow")
    assert body_overflow == ""
```

### 15.2 Close via Backdrop Click

```python
def test_slideout_close_on_backdrop(page, server):
    page.goto(server + "/ui/migrations/1")
    page.locator('button:has-text("View diff")').first.click()
    page.wait_for_load_state("networkidle")
    expect(page.locator('#diff-panel')).to_be_visible()

    # Click backdrop
    page.locator('#diff-backdrop').click(force=True)
    page.wait_for_timeout(400)
    expect(page.locator('#diff-panel')).to_be_hidden()
```

### 15.3 Close via Escape Key

```python
def test_slideout_close_on_escape(page, server):
    page.goto(server + "/ui/migrations/1")
    page.locator('button:has-text("View diff")').first.click()
    page.wait_for_load_state("networkidle")
    expect(page.locator('#diff-panel')).to_be_visible()

    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    expect(page.locator('#diff-panel')).to_be_hidden()
```

### 15.4 Close via Footer "CLOSE" Button

```python
def test_slideout_close_via_footer_button(page, server):
    page.goto(server + "/ui/migrations/1")
    page.locator('button:has-text("View diff")').first.click()
    page.wait_for_load_state("networkidle")

    # Click the footer CLOSE button (second close-slideout button)
    close_btns = page.locator('[data-action="close-slideout"]')
    close_btns.last.click()
    page.wait_for_timeout(400)
    expect(page.locator('#diff-panel')).to_be_hidden()
```

---

## 16. Slideout Panel from "Do Not Ship" Detail (Migration 3)

### 16.1 Hover-to-Reveal + Click Pattern

The "Do not ship" layout has "View diff" links that are hidden (`opacity-0`) until hover:

```python
def test_do_not_ship_slideout(page, server):
    page.goto(server + "/ui/migrations/3")

    # Find first regression row
    first_reg = page.locator('.group').filter(has_text="Schema Violation").first

    # Before hover, "View diff" is invisible (opacity: 0)
    diff_link = first_reg.locator('a:has-text("View diff")')

    # Hover to reveal
    first_reg.hover()

    # Click the diff link
    diff_link.click(force=True)  # force=True because opacity animation
    page.wait_for_load_state("networkidle")

    # Panel should open
    expect(page.locator('#diff-panel')).to_be_visible()
    expect(page.locator('#diff-content')).to_contain_text("BERTScore")
```

---

## 17. Complete Test List Summary

Below is the definitive list of individual test functions to implement:

### Page Load & Content Tests
1. `test_models_dashboard_loads` -- `/ui/` returns 200, "Your models" heading
2. `test_models_dashboard_shows_all_models` -- All 4 DUMMY_MODELS rendered
3. `test_models_dashboard_active_count` -- "3 ACTIVE INSTANCES" text
4. `test_models_dashboard_deprecated_card` -- Deprecated card with retirement date, replacement
5. `test_models_dashboard_alerts_banner` -- "3 things need your attention", alert messages
6. `test_models_dashboard_add_model_button` -- "Add model" card present
7. `test_models_dashboard_explore_table` -- 5 rows with correct model data
8. `test_models_empty_state_loads` -- `/ui/?empty=true` returns 200, "Welcome to RosettaStone"
9. `test_models_empty_state_onboarding_card` -- Input fields, buttons, links
10. `test_models_empty_state_input_accepts_text` -- Type into input field
11. `test_empty_param_false_shows_dashboard` -- `/ui/?empty=false` shows normal dashboard
12. `test_migrations_list_loads` -- `/ui/migrations` returns 200, heading, 3 cards
13. `test_migrations_list_card_data` -- All 3 migration cards with correct data
14. `test_migrations_list_card_links` -- Cards link to `/ui/migrations/{id}`
15. `test_migrations_list_pagination_footer` -- "Showing 3 migrations" text
16. `test_migration_detail_safe_to_ship` -- `/ui/migrations/1` full content
17. `test_migration_detail_safe_recommendation_card` -- Recommendation text, icon, confidence
18. `test_migration_detail_safe_kpi_grid` -- Parity, Baseline, Improvement values
19. `test_migration_detail_safe_per_type_results` -- 4 type cards with correct badges
20. `test_migration_detail_safe_regressions` -- 3 regressions with titles, expected/got
21. `test_migration_detail_safe_config_collapsible` -- Toggle open/close
22. `test_migration_detail_safe_model_metadata` -- Static metadata values
23. `test_migration_detail_safe_export_link` -- Links to executive report
24. `test_migration_detail_safe_header_buttons` -- Version History, Deploy to Prod
25. `test_migration_detail_needs_review` -- `/ui/migrations/2` full content
26. `test_migration_detail_needs_review_no_per_type` -- "Results by output type" absent
27. `test_migration_detail_needs_review_no_regressions` -- "Regressions to review" absent
28. `test_migration_detail_needs_review_recommendation` -- Gold card, warning icon
29. `test_migration_detail_do_not_ship` -- `/ui/migrations/3` full content
30. `test_migration_detail_do_not_ship_header` -- "FAILED" badge, trending_flat icon
31. `test_migration_detail_do_not_ship_per_type` -- 4 type cards with FAIL/WARN badges
32. `test_migration_detail_do_not_ship_regressions` -- 3 regressions, "Showing 3 of 3"
33. `test_costs_page_loads` -- `/ui/costs` returns 200, heading
34. `test_costs_page_kpi_values` -- $1,247, $312, $935
35. `test_costs_page_model_breakdown` -- 3 model bars with correct data
36. `test_costs_page_optimization_opportunities` -- 2 opportunity cards
37. `test_alerts_page_loads` -- `/ui/alerts` returns 200, heading
38. `test_alerts_page_hero_section` -- "3 Critical Alerts", "ACTION REQUIRED."
39. `test_alerts_page_alert_cards` -- 3 alert cards with correct type-specific rendering
40. `test_alerts_page_notification_settings` -- 3 checkboxes with correct default states
41. `test_alerts_page_checkbox_toggle` -- Toggle unchecked checkbox
42. `test_executive_report_safe` -- `/ui/migrations/1/executive` full content
43. `test_executive_report_standalone_no_nav` -- No nav bar or base footer
44. `test_executive_report_metric_boxes` -- Quality match, Cost, Risk, Deployment status
45. `test_executive_report_what_improves` -- Improvement bullets
46. `test_executive_report_what_to_watch` -- Regression bullets
47. `test_executive_report_per_type` -- 4 per-type rows
48. `test_executive_report_do_not_ship` -- `/ui/migrations/3/executive` variant
49. `test_executive_report_needs_review` -- `/ui/migrations/2/executive` variant
50. `test_diff_fragment_loads` -- `/ui/fragments/diff/1/42` returns 200
51. `test_diff_fragment_content` -- All score values, expected/actual content, metadata

### Navigation Tests
52. `test_nav_links_present_on_all_pages` -- Verify all nav links on each page
53. `test_nav_active_state_models` -- "Models" active on `/ui/`
54. `test_nav_active_state_migrations` -- "Migrations" active on `/ui/migrations`
55. `test_nav_active_state_costs` -- "Costs" active on `/ui/costs`
56. `test_nav_active_state_alerts` -- "Alerts" active on `/ui/alerts`
57. `test_nav_active_state_migration_detail` -- "Migrations" active on `/ui/migrations/1`
58. `test_nav_click_navigation` -- Click each nav link, verify URL change
59. `test_migration_card_click_navigates` -- Click migration card, verify detail page loads
60. `test_breadcrumb_navigation` -- Click breadcrumb on detail page, verify return to list

### Interactive Behavior Tests
61. `test_theme_toggle_dark_to_light` -- Toggle theme, verify data-theme attribute
62. `test_theme_toggle_light_to_dark` -- Toggle back
63. `test_theme_persists_across_navigation` -- Toggle, navigate, verify persistence
64. `test_theme_toggle_updates_localstorage` -- Verify localStorage value
65. `test_slideout_opens_on_view_diff` -- Click "View diff", panel opens
66. `test_slideout_content_loads` -- Diff content present after open
67. `test_slideout_close_via_x_button` -- Close via X
68. `test_slideout_close_via_backdrop` -- Close via backdrop click
69. `test_slideout_close_via_escape` -- Close via Escape key
70. `test_slideout_close_via_footer_button` -- Close via footer CLOSE button
71. `test_slideout_body_scroll_locked` -- Body overflow hidden when open
72. `test_slideout_body_scroll_restored` -- Body overflow restored when closed
73. `test_collapsible_config_toggle` -- Open/close config section on migration detail
74. `test_do_not_ship_hover_reveal_diff_link` -- Hover to reveal "View diff"

### Visual/Styling Tests
75. `test_safe_to_ship_badge_color` -- Sage green badge
76. `test_needs_review_badge_color` -- Gold badge
77. `test_do_not_ship_badge_color` -- Red badge
78. `test_migration_card_left_border_colors` -- Correct border colors per recommendation
79. `test_alerts_card_border_colors` -- Deprecation=red, price=gold, new=sage
80. `test_executive_report_light_background` -- Light bg on standalone report

### Error Handling Tests
81. `test_migration_detail_404` -- `/ui/migrations/999` returns 404
82. `test_executive_report_404` -- `/ui/migrations/999/executive` returns 404
83. `test_migration_detail_zero_id_404` -- `/ui/migrations/0` returns 404

### Footer Tests
84. `test_footer_present_on_all_pages` -- Footer visible on dashboard, migrations, costs, alerts
85. `test_footer_copyright_text` -- "2024 RosettaStone Intelligence"
86. `test_footer_links` -- Documentation, API Reference, Status, Privacy links present
87. `test_footer_absent_on_executive_report` -- No base footer on executive report

---

## 18. Implementation Notes for the Sonnet Subagent

### File Structure
```
tests/test_server/test_playwright_ui.py
```

### Dependencies
- `playwright` (install via `pip install playwright && playwright install chromium`)
- `pytest-playwright` or manual fixture management
- The `conftest.py` in `tests/test_server/` may need a server fixture

### Server Fixture
- Use `subprocess.Popen` to start the uvicorn server
- Use a unique port (e.g., 8765) to avoid conflicts
- Poll the `/api/v1/health` endpoint until it returns 200
- Kill the process in fixture teardown
- Use `scope="session"` for efficiency

### Important: HTMX Wait Pattern
After clicking any element that triggers an HTMX request:
```python
page.wait_for_load_state("networkidle")
```

### Important: Slideout Animation Wait
After triggering slideout close, wait for the 300ms CSS transition:
```python
page.wait_for_timeout(400)  # slightly more than 300ms transition
```

### Important: Force Clicks on Hidden Elements
The "View diff" links on the "Do not ship" page have `opacity: 0` until hover. Use:
```python
element.hover()  # first make visible
element.click()  # then click
# OR
element.click(force=True)  # bypass visibility check
```

### Important: Regex for URL Matching
```python
import re
expect(page).to_have_url(re.compile(r"/ui/migrations/1$"))
```

### Important: Evaluating JavaScript
For checking localStorage, computed styles, etc.:
```python
value = page.evaluate("() => localStorage.getItem('rosettastone-theme')")
bg_color = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
```

### Grouping
Organize tests into classes for logical grouping:
```python
class TestModelsDashboard: ...
class TestModelsEmptyState: ...
class TestMigrationsList: ...
class TestMigrationDetailSafeToShip: ...
class TestMigrationDetailNeedsReview: ...
class TestMigrationDetailDoNotShip: ...
class TestCostsPage: ...
class TestAlertsPage: ...
class TestExecutiveReport: ...
class TestDiffFragment: ...
class TestNavigation: ...
class TestThemeToggle: ...
class TestFooter: ...
class TestSlideoutPanel: ...
class TestErrorHandling: ...
```
