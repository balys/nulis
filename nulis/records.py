"""Run records: one JSON line per rollout, in a file per split.

Held-out records live in their own file on purpose. Whatever proposes harness edits or
training recipes is handed `records.evolve.jsonl` and nothing else; keeping the two apart on
disk makes that a matter of which file is passed, not of filtering done correctly every time.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from nulis.splits import SPLITS

PASSED = "passed"
FAILED = "failed"
ERROR = "error"
"""passed/failed are the model's result. error means the rig could not run the rollout (no
image, docker down); it is never counted as a failure and is retried on the next run."""


class RunRecord(msgspec.Struct, kw_only=True):
    task_id: str
    library: str
    split: str
    condition: str
    model: str
    adapter: str | None
    harness: str
    oracle: bool
    status: str
    started_at: float
    elapsed_s: float
    agent_exit: int | None = None
    agent_timed_out: bool = False
    grade_exit: int | None = None
    grade_timed_out: bool = False
    detail: str | None = None
    session: str | None = None

    @property
    def key(self) -> tuple:
        """What makes two rollouts the same measurement."""
        return (self.task_id, self.condition, self.model, self.adapter, self.harness,
                self.oracle)


def records_path(run_dir: Path, split: str) -> Path:
    return run_dir / f"records.{split}.jsonl"


def append(run_dir: Path, record: RunRecord) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with records_path(run_dir, record.split).open("ab") as handle:
        handle.write(msgspec.json.encode(record) + b"\n")


def load(run_dir: Path, split: str | None = None) -> list[RunRecord]:
    """Records for one split, or for all. A torn last line (a killed run) is skipped."""
    out: list[RunRecord] = []
    for name in ([split] if split else SPLITS):
        path = records_path(run_dir, name)
        if not path.is_file():
            continue
        for line in path.read_bytes().splitlines():
            if not line.strip():
                continue
            try:
                out.append(msgspec.json.decode(line, type=RunRecord))
            except msgspec.DecodeError:
                continue
    return out
