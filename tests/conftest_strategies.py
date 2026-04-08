"""Shared Hypothesis strategies for property-based testing."""

from __future__ import annotations

from hypothesis import strategies as st

from rosettastone.core.types import OutputType, PromptPair


@st.composite
def st_prompt_pair(draw):
    """Generate a valid PromptPair."""
    prompt = draw(st.text(min_size=1, max_size=200))
    response = draw(st.text(min_size=1, max_size=200))
    source_model = draw(
        st.sampled_from(["openai/gpt-4o", "anthropic/claude-sonnet-4", "ollama/llama3"])
    )
    return PromptPair(prompt=prompt, response=response, source_model=source_model)


@st.composite
def st_output_type(draw):
    """Generate a valid OutputType."""
    return draw(st.sampled_from(list(OutputType)))


@st.composite
def st_score_dict(draw):
    """Generate a valid evaluation score dictionary.

    Produces a dict with 1–5 keys drawn from the known metric names, each
    mapped to a float in [0.0, 1.0].  The ``json_valid`` key is intentionally
    included so callers can exercise the JSON-gate code path.
    """
    score_keys = [
        "bertscore_f1",
        "embedding_sim",
        "exact_match",
        "json_valid",
        "json_structural_sim",
    ]
    num_keys = draw(st.integers(min_value=1, max_value=len(score_keys)))
    selected = draw(st.permutations(score_keys).map(lambda x: x[:num_keys]))
    return {k: draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)) for k in selected}
