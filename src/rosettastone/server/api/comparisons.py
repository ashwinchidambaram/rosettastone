"""Score distribution and diff comparison endpoints."""

from __future__ import annotations

import difflib
import html
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select

from rosettastone.server.api.migrations import _test_case_to_diff_dict
from rosettastone.server.api.utils import _get_migration_with_owner_check
from rosettastone.server.database import get_session
from rosettastone.server.models import TestCaseRecord
from rosettastone.server.schemas import DiffData, ScoreDistribution, TypeScoreStats

router = APIRouter()


# ---------------------------------------------------------------------------
# Word-level diff helper
# ---------------------------------------------------------------------------


def _word_diff_html(expected: str, actual: str) -> tuple[str, str]:
    """Return (expected_html, actual_html) with changed words wrapped in diff spans.

    Uses difflib.SequenceMatcher on content tokens only (non-whitespace), while
    preserving the original whitespace separators (including newlines, indentation)
    between tokens.  This means multi-line content (JSON, code) renders correctly
    when the container uses ``white-space: pre-wrap``.

    Unchanged tokens are HTML-escaped and emitted as-is; deleted tokens (in expected
    but not actual) are wrapped in ``<span class="diff-del">``; inserted tokens (in
    actual but not expected) are wrapped in ``<span class="diff-add">``.
    """
    # re.split with a capturing group keeps the separators in the result list:
    # "a\n b" -> ["a", "\n ", "b"]
    # Empty strings can appear at the start/end; we handle them harmlessly below.
    exp_parts = re.split(r"(\s+)", expected)  # [word, ws, word, ws, ...]
    act_parts = re.split(r"(\s+)", actual)

    # Even indices → content tokens; odd indices → whitespace separators.
    exp_words: list[str] = exp_parts[0::2]
    act_words: list[str] = act_parts[0::2]
    exp_ws: list[str] = exp_parts[1::2]
    act_ws: list[str] = act_parts[1::2]

    matcher = difflib.SequenceMatcher(None, exp_words, act_words, autojunk=False)
    opcodes = matcher.get_opcodes()

    def _build(words: list[str], ws_list: list[str], side: str) -> str:
        parts: list[str] = []
        for tag, i1, i2, j1, j2 in opcodes:
            rng = range(i1, i2) if side == "exp" else range(j1, j2)
            for i in rng:
                word = words[i]
                esc = html.escape(word)
                if word:  # skip empty strings produced by leading/trailing split
                    if (side == "exp" and tag in ("replace", "delete")) or (
                        side == "act" and tag in ("replace", "insert")
                    ):
                        cls = "diff-del" if side == "exp" else "diff-add"
                        parts.append(f'<span class="{cls}">{esc}</span>')
                    else:
                        parts.append(esc)
                # Restore the original whitespace that followed this word
                if i < len(ws_list):
                    parts.append(html.escape(ws_list[i]))
        return "".join(parts)

    return _build(exp_words, exp_ws, "exp"), _build(act_words, act_ws, "act")


# ---------------------------------------------------------------------------
# Dummy data for UI diff fragment
# ---------------------------------------------------------------------------

_DUMMY_EXPECTED = '{\n  "priority": "urgent",\n  "category": "billing",\n  "confidence": 0.94\n}'
_DUMMY_ACTUAL = (
    '{\n  "priority": "high_priority",\n  "category": "billing",\n  "confidence": 0.91\n}'
)
_dummy_expected_html, _dummy_actual_html = _word_diff_html(_DUMMY_EXPECTED, _DUMMY_ACTUAL)

DUMMY_DIFF = {
    "tc_id": 42,
    "is_win": False,
    "composite_score": 0.72,
    "output_type": "Classification",
    "scores": {"bertscore": 0.85, "embedding": 0.79, "composite": 0.72},
    "source_model": "gpt-4o",
    "target_model": "claude-sonnet-4",
    "expected": _DUMMY_EXPECTED,
    "actual": _DUMMY_ACTUAL,
    "expected_html": _dummy_expected_html,
    "actual_html": _dummy_actual_html,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_distributions(migration_id: int, session: Session) -> list[ScoreDistribution]:
    """Compute score distributions per output type from TestCaseRecords."""
    stmt = select(TestCaseRecord).where(TestCaseRecord.migration_id == migration_id)
    test_cases = list(session.exec(stmt).all())

    # Group test cases by output type
    cases_by_type: dict[str, list[TestCaseRecord]] = {}
    for tc in test_cases:
        output_type = tc.output_type
        if output_type not in cases_by_type:
            cases_by_type[output_type] = []
        cases_by_type[output_type].append(tc)

    distributions: list[ScoreDistribution] = []
    for output_type, cases in sorted(cases_by_type.items()):
        scores = sorted(tc.composite_score for tc in cases)
        n = len(scores)
        if n == 0:
            continue

        mean_score = sum(scores) / n
        median_score = scores[n // 2] if n % 2 == 1 else (scores[n // 2 - 1] + scores[n // 2]) / 2

        def _percentile(sorted_vals: list[float], p: float) -> float:
            """Compute percentile from sorted values."""
            if len(sorted_vals) == 1:
                return sorted_vals[0]
            idx = p * (len(sorted_vals) - 1)
            lower = int(idx)
            upper = lower + 1
            if upper >= len(sorted_vals):
                return sorted_vals[-1]
            frac = idx - lower
            return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])

        win_count = sum(1 for tc in cases if tc.is_win)

        # Build histogram: 10 buckets from 0.0 to 1.0
        histogram = [0] * 10
        for s in scores:
            bucket = min(int(s * 10), 9)
            histogram[bucket] += 1

        stats = TypeScoreStats(
            win_rate=win_count / n,
            mean=mean_score,
            median=median_score,
            p10=_percentile(scores, 0.10),
            p50=_percentile(scores, 0.50),
            p90=_percentile(scores, 0.90),
            min_score=scores[0],
            max_score=scores[-1],
            sample_count=n,
            confidence_interval=(
                _percentile(scores, 0.025),
                _percentile(scores, 0.975),
            ),
        )

        distributions.append(
            ScoreDistribution(
                output_type=output_type,
                stats=stats,
                histogram=histogram,
            )
        )

    return distributions


def _build_diff_data(tc: TestCaseRecord) -> DiffData:
    """Build diff data from a test case record."""
    available = bool(tc.prompt_text or tc.response_text or tc.new_response_text)
    return DiffData(
        prompt=tc.prompt_text,
        source_response=tc.response_text,
        target_response=tc.new_response_text,
        available=available,
    )


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/migrations/{migration_id}/distributions",
    response_model=list[ScoreDistribution],
)
async def get_distributions(
    migration_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> list[ScoreDistribution]:
    """Get score distributions per output type."""
    _get_migration_with_owner_check(migration_id, session, request)
    return _compute_distributions(migration_id, session)


@router.get(
    "/api/v1/migrations/{migration_id}/test-cases/{tc_id}/diff",
    response_model=DiffData,
)
async def get_diff(
    migration_id: int,
    tc_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> DiffData:
    """Get diff data for a test case."""
    _get_migration_with_owner_check(migration_id, session, request)

    tc = session.get(TestCaseRecord, tc_id)
    if tc is None or tc.migration_id != migration_id:
        raise HTTPException(status_code=404, detail="Test case not found")

    return _build_diff_data(tc)


# ---------------------------------------------------------------------------
# HTMX / UI endpoints
# ---------------------------------------------------------------------------


@router.get("/ui/fragments/diff/{migration_id}/{tc_id}", response_class=HTMLResponse)
async def diff_fragment(
    migration_id: int,
    tc_id: int,
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """HTMX partial for diff view."""
    try:
        migration = _get_migration_with_owner_check(migration_id, session, request)
    except HTTPException as exc:
        if exc.status_code == 403:
            raise  # IDOR: deny cross-user access
        migration = None  # 404: fall back to dummy diff
    tc = session.get(TestCaseRecord, tc_id) if migration else None

    if tc and tc.migration_id == migration_id:
        diff = _test_case_to_diff_dict(tc, migration)  # type: ignore[arg-type]
        # Augment with word-level diff HTML when actual content is stored in the DB.
        # Check the ORM field directly rather than inspecting the rendered placeholder text.
        content_stored = tc.response_text is not None
        expected_text: str = diff.get("expected", "") or ""
        actual_text: str = diff.get("actual", "") or ""
        if content_stored and expected_text and actual_text:
            diff["expected_html"], diff["actual_html"] = _word_diff_html(expected_text, actual_text)
    else:
        diff = DUMMY_DIFF  # fallback

    return request.app.state.templates.TemplateResponse(  # type: ignore[no-any-return]
        request,
        "fragments/diff_slideout.html",
        {"diff": diff},
    )


@router.get("/ui/fragments/charts/{migration_id}", response_class=HTMLResponse)
async def charts_fragment(migration_id: int, request: Request) -> HTMLResponse:
    """HTMX partial for charts."""
    return HTMLResponse("<div>Template pending integration</div>")
