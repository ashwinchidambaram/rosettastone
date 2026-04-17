"""Property-based tests for splitter deduplication and splitting."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from rosettastone.ingest.splitter import deduplicate, split_data
from tests.conftest_strategies import st_prompt_pair


def _pair_key(pair) -> str:
    """Stable identity key: prompt text + response."""
    prompt = pair.prompt if isinstance(pair.prompt, str) else str(pair.prompt)
    return prompt + "\x00" + pair.response


@settings(max_examples=200)
@given(st.lists(st_prompt_pair(), min_size=0, max_size=50))
def test_deduplicate_output_leq_input(pairs):
    """Deduplicated list is never longer than the input."""
    result = deduplicate(pairs)
    assert len(result) <= len(pairs)


@settings(max_examples=200)
@given(st.lists(st_prompt_pair(), min_size=0, max_size=50))
def test_deduplicate_idempotent(pairs):
    """Deduplicating twice produces the same result as deduplicating once."""
    once = deduplicate(pairs)
    twice = deduplicate(once)
    assert [_pair_key(p) for p in twice] == [_pair_key(p) for p in once]


@settings(max_examples=200)
@given(st.lists(st_prompt_pair(), min_size=0, max_size=50))
def test_deduplicate_all_output_from_input(pairs):
    """Every pair returned by deduplicate was present in the input."""
    input_keys = {_pair_key(p) for p in pairs}
    for pair in deduplicate(pairs):
        assert _pair_key(pair) in input_keys


@settings(max_examples=200)
@given(st.lists(st_prompt_pair(), min_size=3, max_size=50))
def test_split_union_equals_deduped(pairs):
    """Union of all splits equals the deduplicated input (no pairs lost or added)."""
    deduped_keys = {_pair_key(p) for p in deduplicate(pairs)}
    train, val, test = split_data(pairs, seed=42)
    split_keys = {_pair_key(p) for p in train + val + test}
    assert split_keys == deduped_keys


@settings(max_examples=200)
@given(st.lists(st_prompt_pair(), min_size=3, max_size=50))
def test_split_no_overlap(pairs):
    """No pair object appears in more than one split (checked by object identity)."""
    train, val, test = split_data(pairs, seed=42)
    train_ids = {id(p) for p in train}
    val_ids = {id(p) for p in val}
    test_ids = {id(p) for p in test}
    assert train_ids.isdisjoint(val_ids), "train and val overlap"
    assert train_ids.isdisjoint(test_ids), "train and test overlap"
    assert val_ids.isdisjoint(test_ids), "val and test overlap"


@settings(max_examples=200)
@given(st.lists(st_prompt_pair(), min_size=3, max_size=50))
def test_split_test_never_empty(pairs):
    """Test set is never empty when enough deduplicated pairs exist for all three splits.

    split_data requires at least 3 unique pairs to guarantee a non-empty test set:
    one each for train, val, and test.  With fewer deduplicated items the train
    slice can consume all data before a test set can be formed.
    """
    deduped = deduplicate(pairs)
    if len(deduped) < 3:
        return  # not enough unique pairs to make the guarantee; skip
    _train, _val, test = split_data(pairs, seed=42)
    assert len(test) >= 1


@settings(max_examples=200)
@given(st.lists(st_prompt_pair(), min_size=3, max_size=50), st.integers(min_value=0))
def test_split_deterministic_with_seed(pairs, seed):
    """Two calls with the same seed produce identical splits."""
    train1, val1, test1 = split_data(pairs, seed=seed)
    train2, val2, test2 = split_data(pairs, seed=seed)
    assert [_pair_key(p) for p in train1] == [_pair_key(p) for p in train2]
    assert [_pair_key(p) for p in val1] == [_pair_key(p) for p in val2]
    assert [_pair_key(p) for p in test1] == [_pair_key(p) for p in test2]
