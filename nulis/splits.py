"""Evolve and held-out splits, and the conditions a task can be run under.

The split is a pure function of a seed and a key, so it needs no lock file and adding tasks
never moves an existing one. The key is the task id, or the library when whole libraries must
be held out (the library stream: studying one task of a library would leak into its others).
"""

from __future__ import annotations

import hashlib

import msgspec

EVOLVE = "evolve"
HELDOUT = "heldout"
SPLITS = (EVOLVE, HELDOUT)


def split_of(key: str, *, seed: str = "nulis-v1", heldout: float = 0.5) -> str:
    """`heldout` is the fraction of keys that land in the held-out split."""
    if not 0.0 <= heldout <= 1.0:
        raise ValueError(f"heldout fraction must be within 0..1, got {heldout}")
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    return HELDOUT if point < heldout else EVOLVE


class Condition(msgspec.Struct, frozen=True):
    """What the agent phase is given. Grading is identical under every condition."""

    name: str
    docs: bool
    """The task's docs/ directory is readable in the workspace."""
    library: bool
    """The library is installed where the agent runs, so it can execute code against it."""
    memory: bool
    """A notes directory persists across the tasks of one library."""


CONDITIONS = {
    "closed": Condition("closed", docs=False, library=False, memory=False),
    "docs": Condition("docs", docs=True, library=False, memory=False),
    "open": Condition("open", docs=True, library=True, memory=False),
    "memory": Condition("memory", docs=True, library=True, memory=True),
}
"""closed: what the weights know. docs: lookup only. open: lookup plus execution feedback.
memory: open, plus notes carried from task to task. "Closed-book after study" is `closed`
again, run with the adapter that the study produced."""
