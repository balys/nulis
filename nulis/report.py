"""Pass rates from run records, with intervals, and the step 1 gate.

The gate is the plan's stop condition for the rig step: if the base model with documentation
to hand already solves nearly everything, there is little left for self-study to add on that
benchmark, and the benchmark (or the plan) needs rethinking before any training is paid for.
"""

from __future__ import annotations

import math

import msgspec

from nulis.records import ERROR, PASSED, RunRecord

OPEN_BOOK = ("docs", "open")


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a pass rate. Benchmarks here are a few hundred tasks, where
    the plain +-1.96*se interval is wrong near 0 and 1."""
    if n == 0:
        return (0.0, 1.0)
    p = passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class Row(msgspec.Struct, kw_only=True):
    model: str
    adapter: str | None
    harness: str
    split: str
    condition: str
    oracle: bool
    n: int
    passed: int
    errors: int
    rate: float
    low: float
    high: float


def summarize(records: list[RunRecord]) -> list[Row]:
    """One row per model, adapter, harness, split and condition. When a rollout was repeated
    (--rerun), the latest record is the one counted. Rig errors are counted apart and are in
    neither `n` nor the rate."""
    latest: dict[tuple, RunRecord] = {}
    for record in records:
        key = (*record.key, record.split)
        if key not in latest or record.started_at >= latest[key].started_at:
            latest[key] = record
    groups: dict[tuple, list[RunRecord]] = {}
    for record in latest.values():
        key = (record.model, record.adapter or "", record.harness, record.split,
               record.condition, record.oracle)
        groups.setdefault(key, []).append(record)
    rows = []
    for key in sorted(groups):
        group = groups[key]
        scored = [r for r in group if r.status != ERROR]
        passed = sum(r.status == PASSED for r in scored)
        low, high = wilson(passed, len(scored))
        rows.append(Row(model=key[0], adapter=key[1] or None, harness=key[2], split=key[3],
                        condition=key[4], oracle=key[5], n=len(scored), passed=passed,
                        errors=len(group) - len(scored),
                        rate=passed / len(scored) if scored else 0.0, low=low, high=high))
    return rows


def gate(rows: list[Row], ceiling: float = 0.9, min_n: int = 30) -> list[str]:
    """Findings that should stop the run of the plan: an oracle that does not pass everything
    (the pipeline is broken, so no other number means anything) and open-book results already
    at the ceiling. The ceiling is only judged on `min_n` or more tasks: three passes out of
    three say nothing about headroom."""
    findings = []
    for row in rows:
        if row.oracle and (row.passed < row.n or row.errors):
            findings.append(f"oracle is not clean on {row.split}: {row.passed}/{row.n} passed, "
                            f"{row.errors} errors. Fix the tasks or the sandbox before any "
                            "model run.")
        elif (not row.oracle and row.condition in OPEN_BOOK and row.n >= max(min_n, 1)
              and row.rate >= ceiling):
            findings.append(f"{row.model} already passes {row.rate:.0%} with condition "
                            f"{row.condition!r} on {row.split}: little headroom for study "
                            "on this benchmark.")
    return findings


def render(rows: list[Row]) -> str:
    if not rows:
        return "no records"
    head = ("model", "adapter", "harness", "split", "condition", "pass", "rate", "95% interval",
            "errors")
    table = [head]
    for row in rows:
        table.append((row.model, row.adapter or "-", row.harness, row.split,
                      row.condition + (" (oracle)" if row.oracle else ""),
                      f"{row.passed}/{row.n}", f"{row.rate:.1%}",
                      f"{row.low:.1%}-{row.high:.1%}", str(row.errors)))
    widths = [max(len(line[i]) for line in table) for i in range(len(head))]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip()
                     for line in table)
