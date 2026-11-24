"""Library tasks: one directory per task, `task.toml` as the contract.

    <task>/task.toml            the [task] table below
    <task>/prompt.md            what the agent is asked to write
    <task>/tests/               the hidden tests; mounted only into the grading container
    <task>/docs/                optional: library documentation for the open-book conditions
    <task>/solution/<entry>     optional: the reference answer, used only by --oracle

This is the rig's own schema, so it is strict: an unknown field is refused, because a silently
ignored `test_cmnd` is a task graded by the default command.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import msgspec


class TaskError(Exception):
    """A task directory that breaks its contract. The message names the path and the fix."""


_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class TaskSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    id: str
    library: str
    image: str
    """Docker image with the library and the test runner installed. Grading always runs in
    it; the agent phase uses it only in conditions that allow execution."""
    entry: str = "solution.py"
    library_version: str | None = None
    released: str | None = None
    """ISO date the library version came out. The library stream orders and filters on it."""
    test_cmd: str = "python -m pytest -q -p no:cacheprovider tests"
    agent_timeout_sec: float = 1800.0
    grade_timeout_sec: float = 300.0


@dataclass(frozen=True)
class Task:
    spec: TaskSpec
    dir: Path

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def prompt(self) -> str:
        return (self.dir / "prompt.md").read_text(encoding="utf-8")

    @property
    def tests_dir(self) -> Path:
        return self.dir / "tests"

    @property
    def docs_dir(self) -> Path | None:
        docs = self.dir / "docs"
        return docs if docs.is_dir() and any(docs.iterdir()) else None

    @property
    def solution(self) -> Path | None:
        path = self.dir / "solution" / self.spec.entry
        return path if path.is_file() else None


def load_task(task_dir: Path) -> Task:
    path = task_dir / "task.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaskError(f"{path}: no task.toml") from exc
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise TaskError(f"bad {path}: {exc}") from exc
    table = data.get("task")
    if not isinstance(table, dict) or set(data) != {"task"}:
        raise TaskError(f"{path}: expected exactly one [task] table")
    try:
        spec = msgspec.convert(table, TaskSpec)
    except msgspec.ValidationError as exc:
        raise TaskError(f"bad {path}: {exc}") from exc
    if not _ID.match(spec.id):
        raise TaskError(f"{path}: id {spec.id!r} must match {_ID.pattern}")
    if not spec.library or not spec.image:
        raise TaskError(f"{path}: library and image must not be empty")
    if Path(spec.entry).name != spec.entry or spec.entry in ("", ".", ".."):
        # Only this one file leaves the agent's workspace for grading, so it must be a bare
        # name: a path could reach outside the workspace or onto the tests.
        raise TaskError(f"{path}: entry {spec.entry!r} must be a bare file name")
    if spec.agent_timeout_sec <= 0 or spec.grade_timeout_sec <= 0:
        raise TaskError(f"{path}: timeouts must be positive")
    if not (task_dir / "prompt.md").is_file():
        raise TaskError(f"{task_dir}: no prompt.md")
    tests = task_dir / "tests"
    if not tests.is_dir() or not any(tests.iterdir()):
        raise TaskError(f"{task_dir}: tests/ is missing or empty")
    return Task(spec=spec, dir=task_dir.resolve())


def load_tasks(root: Path) -> list[Task]:
    """Every task under `root`, sorted by id. Duplicate ids are refused: records are keyed by
    id, and two tasks sharing one would overwrite each other's results."""
    if not root.is_dir():
        raise TaskError(f"{root}: no such tasks directory")
    tasks = [load_task(path.parent) for path in sorted(root.rglob("task.toml"))]
    seen: dict[str, Path] = {}
    for task in tasks:
        if task.id in seen:
            raise TaskError(f"duplicate task id {task.id!r}: {seen[task.id]} and {task.dir}")
        seen[task.id] = task.dir
    return sorted(tasks, key=lambda task: task.id)
