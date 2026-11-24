import pytest

from nulis.splits import CONDITIONS, EVOLVE, HELDOUT, split_of


def test_split_is_a_pure_function_of_seed_and_key():
    assert split_of("task-1") == split_of("task-1")
    assert {split_of(f"t{i}") for i in range(50)} == {EVOLVE, HELDOUT}
    moved = sum(split_of(f"t{i}", seed="a") != split_of(f"t{i}", seed="b") for i in range(200))
    assert moved > 50  # a different seed is a different split


def test_fraction_is_respected():
    keys = [f"task-{i}" for i in range(2000)]
    share = sum(split_of(key, heldout=0.25) == HELDOUT for key in keys) / len(keys)
    assert 0.21 < share < 0.29
    assert all(split_of(key, heldout=0.0) == EVOLVE for key in keys[:50])
    assert all(split_of(key, heldout=1.0) == HELDOUT for key in keys[:50])
    with pytest.raises(ValueError):
        split_of("x", heldout=1.5)


def test_conditions_only_add_capabilities():
    order = [CONDITIONS[name] for name in ("closed", "docs", "open", "memory")]
    flags = [(c.docs, c.library, c.memory) for c in order]
    assert flags == [(False, False, False), (True, False, False), (True, True, False),
                     (True, True, True)]
