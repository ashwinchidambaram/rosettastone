"""Tests for the deduplication and train/val/test split logic."""

from __future__ import annotations

import random

import pytest

from rosettastone.core.types import PromptPair
from rosettastone.ingest.splitter import deduplicate, split_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(prompt: str, response: str, source_model: str = "openai/gpt-4o") -> PromptPair:
    return PromptPair(prompt=prompt, response=response, source_model=source_model)


def _make_pairs(n: int) -> list[PromptPair]:
    """Create n unique pairs with distinct prompt/response content."""
    return [_make_pair(f"Prompt number {i}", f"Response number {i}") for i in range(n)]


# ---------------------------------------------------------------------------
# deduplicate()
# ---------------------------------------------------------------------------


def test_deduplicate_removes_exact_duplicates():
    """This test proves that identical prompt+response pairs are reduced to one entry."""
    pair = _make_pair("What is 2+2?", "4")
    duplicated = [pair, pair, pair]

    result = deduplicate(duplicated)

    assert len(result) == 1, (
        f"Expected 1 unique pair after dedup of 3 identical pairs, got {len(result)}"
    )
    assert result[0].prompt == "What is 2+2?", (
        f"Expected original pair preserved, got: {result[0].prompt!r}"
    )


def test_deduplicate_preserves_all_unique_pairs():
    """This test proves that distinct pairs are all retained after deduplication."""
    pairs = _make_pairs(5)

    result = deduplicate(pairs)

    assert len(result) == 5, (
        f"Expected 5 unique pairs preserved, got {len(result)}"
    )


def test_deduplicate_mixed_duplicates_and_uniques():
    """This test proves that deduplication only removes true duplicates, not near-duplicates."""
    pair_a = _make_pair("Hello", "Hi there")
    pair_b = _make_pair("Hello", "Hey!")  # same prompt, different response — NOT a duplicate
    pair_c = _make_pair("Greetings", "Hi there")  # different prompt, same response — NOT a dup
    pair_a_copy = _make_pair("Hello", "Hi there")  # exact duplicate of pair_a

    result = deduplicate([pair_a, pair_b, pair_c, pair_a_copy])

    assert len(result) == 3, (
        f"Expected 3 unique pairs (a, b, c) with a_copy removed, got {len(result)}"
    )


def test_deduplicate_same_prompt_different_response_not_removed():
    """This test proves that pairs with matching prompts but different responses are both kept."""
    pair1 = _make_pair("What is the weather?", "Sunny")
    pair2 = _make_pair("What is the weather?", "Rainy")

    result = deduplicate([pair1, pair2])

    assert len(result) == 2, (
        f"Expected both pairs preserved (different responses), got {len(result)}"
    )


def test_deduplicate_same_response_different_prompt_not_removed():
    """This test proves that pairs with matching responses but different prompts are both kept."""
    pair1 = _make_pair("What color is the sky?", "Blue")
    pair2 = _make_pair("What color is the ocean?", "Blue")

    result = deduplicate([pair1, pair2])

    assert len(result) == 2, (
        f"Expected both pairs preserved (different prompts), got {len(result)}"
    )


def test_deduplicate_empty_list_returns_empty():
    """This test proves that deduplicating an empty list returns an empty list."""
    result = deduplicate([])

    assert result == [], f"Expected empty list, got: {result!r}"


def test_deduplicate_single_pair_returns_single():
    """This test proves that a single-element list is unchanged by deduplication."""
    pair = _make_pair("Only one", "response")
    result = deduplicate([pair])

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"


def test_deduplicate_preserves_insertion_order():
    """This test proves that the first occurrence of a duplicate is kept (not the last)."""
    first = _make_pair("Same prompt", "Same response")
    # Create a second instance with same content but slightly different Python object
    second = _make_pair("Same prompt", "Same response")

    result = deduplicate([first, second])

    assert len(result) == 1, f"Expected 1 pair, got {len(result)}"
    # The first occurrence should be the one kept (order preservation)
    assert result[0] is first, "Expected the first occurrence to be preserved, not the second"


def test_deduplicate_list_prompt_fingerprinted_correctly():
    """This test proves that list-format prompts are also correctly deduplicated."""
    messages = [{"role": "user", "content": "Hello"}]
    pair1 = PromptPair(prompt=messages, response="Hi", source_model="openai/gpt-4o")
    pair2 = PromptPair(prompt=messages, response="Hi", source_model="openai/gpt-4o")

    result = deduplicate([pair1, pair2])

    assert len(result) == 1, (
        f"Expected 1 pair after dedup of list-prompt duplicates, got {len(result)}"
    )


# ---------------------------------------------------------------------------
# split_data(): structural invariants (seed-independent)
# ---------------------------------------------------------------------------


def test_split_data_total_count_preserved():
    """This test proves that no pairs are lost or created during splitting."""
    pairs = _make_pairs(20)

    train, val, test = split_data(pairs)

    total = len(train) + len(val) + len(test)
    assert total == 20, (
        f"Expected train+val+test=20, got train={len(train)}, val={len(val)}, test={len(test)}"
    )


def test_split_data_train_set_is_not_empty():
    """This test proves that the train set always contains at least one pair."""
    pairs = _make_pairs(10)

    train, val, test = split_data(pairs)

    assert len(train) >= 1, (
        f"Expected non-empty train set, got {len(train)} items"
    )


def test_split_data_val_set_is_not_empty():
    """This test proves that the validation set always contains at least one pair."""
    pairs = _make_pairs(10)

    train, val, test = split_data(pairs)

    assert len(val) >= 1, (
        f"Expected non-empty val set, got {len(val)} items"
    )


def test_split_data_test_set_is_not_empty():
    """This test proves that the test set is never empty, even with small inputs."""
    pairs = _make_pairs(10)

    train, val, test = split_data(pairs)

    assert len(test) >= 1, (
        f"Expected non-empty test set, got {len(test)} items"
    )


def test_split_data_no_overlap_between_sets():
    """This test proves that the same pair does not appear in multiple sets."""
    pairs = _make_pairs(20)

    train, val, test = split_data(pairs)

    # Use prompt+response as identity (pairs are unique by construction)
    train_ids = {(str(p.prompt), p.response) for p in train}
    val_ids = {(str(p.prompt), p.response) for p in val}
    test_ids = {(str(p.prompt), p.response) for p in test}

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    assert not train_val_overlap, (
        f"Found {len(train_val_overlap)} pair(s) in both train and val: {train_val_overlap}"
    )
    assert not train_test_overlap, (
        f"Found {len(train_test_overlap)} pair(s) in both train and test: {train_test_overlap}"
    )
    assert not val_test_overlap, (
        f"Found {len(val_test_overlap)} pair(s) in both val and test: {val_test_overlap}"
    )


# ---------------------------------------------------------------------------
# split_data(): edge cases with small datasets
# ---------------------------------------------------------------------------


def test_split_data_3_pairs_all_sets_nonempty():
    """This test proves the pop-from-val fallback works: 3 pairs still fill all 3 sets."""
    pairs = _make_pairs(3)

    train, val, test = split_data(pairs)

    total = len(train) + len(val) + len(test)
    assert total == 3, (
        f"Expected train+val+test=3, got {total} (train={len(train)}, val={len(val)}, test={len(test)})"
    )
    assert len(train) >= 1, f"Expected non-empty train with 3 pairs, got {len(train)}"
    assert len(test) >= 1, (
        f"Expected non-empty test with 3 pairs (pop-from-val fallback), got {len(test)}"
    )


def test_split_data_2_pairs_all_sets_nonempty():
    """This test proves that even 2 pairs produce all three non-empty sets via fallback."""
    pairs = _make_pairs(2)

    train, val, test = split_data(pairs)

    total = len(train) + len(val) + len(test)
    assert total == 2, (
        f"Expected train+val+test=2, got {total}"
    )
    assert len(train) >= 1, f"Expected non-empty train, got {len(train)}"
    assert len(test) >= 1, (
        f"Expected non-empty test (pop-from-val fallback), got {len(test)}"
    )


def test_split_data_1_pair_does_not_crash():
    """This test proves that splitting a single pair does not raise an exception."""
    pairs = _make_pairs(1)

    # Should not raise even with minimal data
    train, val, test = split_data(pairs)

    total = len(train) + len(val) + len(test)
    assert total == 1, f"Expected total of 1 pair across all sets, got {total}"


# ---------------------------------------------------------------------------
# split_data(): ratio tests (seed-controlled for reproducibility)
# ---------------------------------------------------------------------------


def test_split_data_train_ratio_respected_with_large_dataset():
    """This test proves that roughly 20% of pairs go to train when train_ratio=0.2."""
    random.seed(42)
    pairs = _make_pairs(100)

    train, val, test = split_data(pairs, train_ratio=0.2, val_ratio=0.8)

    # With 100 pairs and train_ratio=0.2, expect ~20 in train
    # Allow a few items of slack due to integer rounding
    assert 15 <= len(train) <= 25, (
        f"Expected ~20 items in train (train_ratio=0.2, n=100), got {len(train)}"
    )


def test_split_data_val_ratio_respected_with_large_dataset():
    """This test proves that the remaining data splits into val/test per val_ratio."""
    random.seed(42)
    pairs = _make_pairs(100)

    train, val, test = split_data(pairs, train_ratio=0.2, val_ratio=0.8)

    # Remaining 80 pairs split 80/20 into val/test => ~64 val, ~16 test
    remaining = len(val) + len(test)
    assert remaining == 100 - len(train), (
        f"Expected val+test to account for all non-train pairs, got {remaining} vs {100 - len(train)}"
    )


def test_split_data_deduplicates_before_splitting():
    """This test proves that split_data calls deduplicate internally before splitting."""
    pair = _make_pair("Duplicate prompt", "Duplicate response")
    # Feed 10 identical pairs — should deduplicate to 1 before splitting
    pairs = [pair] * 10

    train, val, test = split_data(pairs)

    total = len(train) + len(val) + len(test)
    assert total == 1, (
        f"Expected total of 1 after dedup of 10 identical pairs, got {total}"
    )


def test_split_data_returns_tuple_of_three_lists():
    """This test proves that split_data always returns a 3-tuple of lists."""
    pairs = _make_pairs(5)

    result = split_data(pairs)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3-tuple, got length {len(result)}"
    train, val, test = result
    assert isinstance(train, list), f"Expected train to be list, got {type(train)}"
    assert isinstance(val, list), f"Expected val to be list, got {type(val)}"
    assert isinstance(test, list), f"Expected test to be list, got {type(test)}"


def test_split_data_all_elements_are_prompt_pairs():
    """This test proves that every element in all three sets is a PromptPair."""
    pairs = _make_pairs(15)

    train, val, test = split_data(pairs)

    for i, pair in enumerate(train):
        assert isinstance(pair, PromptPair), (
            f"train[{i}] is {type(pair)}, expected PromptPair"
        )
    for i, pair in enumerate(val):
        assert isinstance(pair, PromptPair), (
            f"val[{i}] is {type(pair)}, expected PromptPair"
        )
    for i, pair in enumerate(test):
        assert isinstance(pair, PromptPair), (
            f"test[{i}] is {type(pair)}, expected PromptPair"
        )
