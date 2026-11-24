import pytest

from nulis.records import ERROR, FAILED, PASSED, RunRecord
from nulis.report import gate, render, summarize, wilson


def rec(task: str, status: str, *, condition: str = "docs", split: str = "evolve",
        oracle: bool = False, started_at: float = 1.0) -> RunRecord:
    return RunRecord(task_id=task, library="lib", split=split, condition=condition, model="m",
                     adapter=None, harness="kilpa@abc", oracle=oracle, status=status,
                     started_at=started_at, elapsed_s=1.0)


def test_wilson_matches_known_values():
    low, high = wilson(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    low, high = wilson(0, 20)
    assert low == 0.0 and high == pytest.approx(0.1611, abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def test_summarize_leaves_rig_errors_out_of_the_rate_and_counts_the_latest_rerun():
    rows = summarize([
        rec("a", FAILED, started_at=1.0), rec("a", PASSED, started_at=2.0),
        rec("b", FAILED), rec("c", ERROR), rec("d", PASSED, condition="closed"),
        rec("e", PASSED, split="heldout"),
    ])
    docs = next(r for r in rows if r.condition == "docs" and r.split == "evolve")
    assert (docs.n, docs.passed, docs.errors, docs.rate) == (2, 1, 1, 0.5)
    assert len(rows) == 3 and "1/2" in render(rows)
    assert render([]) == "no records"


def test_gate_fires_on_a_ceiling_and_on_a_dirty_oracle():
    high = [rec(f"t{i}", PASSED) for i in range(38)] + [rec("t38", FAILED), rec("t39", FAILED)]
    (finding,) = gate(summarize(high))
    assert "little headroom" in finding and "95%" in finding
    assert gate(summarize(high), ceiling=0.99) == []
    assert gate(summarize(high[:20])) == []  # too few tasks to say anything about headroom
    assert len(gate(summarize(high[:20]), min_n=20)) == 1
    closed_only = [rec(f"t{i}", PASSED, condition="closed") for i in range(40)]
    assert gate(summarize(closed_only)) == []  # closed-book at ceiling is success, not a stop
    oracle = [rec("a", PASSED, oracle=True), rec("b", FAILED, oracle=True)]
    (finding,) = gate(summarize(oracle))
    assert "oracle is not clean" in finding
