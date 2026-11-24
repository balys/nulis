"""The rollout loop: tasks x one condition -> records.

One rollout is: a fresh workspace, the agent phase (or the reference solution, with --oracle),
then grading of the single entry file in a networkless sandbox. Rig failures are recorded as
`error` and retried next time; they never count against the model.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import msgspec

from nulis import records
from nulis.records import ERROR, FAILED, PASSED, RunRecord
from nulis.sandbox import AgentRun, CmdResult, DockerSandbox, DockerSettings, LocalSandbox
from nulis.splits import EVOLVE, HELDOUT, Condition, split_of
from nulis.task import Task

DEFAULT_AGENT_CMD = ("kilpa", "-p", "{prompt}", "--mode", "auto", "--model", "{model}")


class ConfigError(Exception):
    """A rig.toml or a run request that cannot be honoured. The message names the fix."""


class RigConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True, kw_only=True):
    model: str
    """The name handed to the harness, as the model server knows it."""
    adapter: str | None = None
    """Label for the weights on top of the base model (a LoRA's id). Recorded, not acted on:
    the server decides what `model` resolves to."""
    sandbox: Literal["docker", "local"] = "docker"
    agent_cmd: list[str] = msgspec.field(default_factory=lambda: list(DEFAULT_AGENT_CMD))
    harness_src: str | None = None
    harness_config: str | None = None
    harness_config_mount: str = "/root/.config/kilpa"
    bare_image: str = "python:3.12-slim"
    agent_network: str = "nulis-rig"
    jobs: int = 1
    split_seed: str = "nulis-v1"
    heldout_fraction: float = 0.5
    split_unit: Literal["task", "library"] = "task"


def load_config(path: Path) -> RigConfig:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: no rig config (docs/RIG.md has a template)") from exc
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"bad {path}: {exc}") from exc
    if set(data) != {"rig"} or not isinstance(data["rig"], dict):
        raise ConfigError(f"{path}: expected exactly one [rig] table")
    try:
        cfg = msgspec.convert(data["rig"], RigConfig)
    except msgspec.ValidationError as exc:
        raise ConfigError(f"bad {path}: {exc}") from exc
    return check_config(cfg, where=str(path))


def check_config(cfg: RigConfig, *, where: str = "rig config") -> RigConfig:
    if not cfg.model:
        raise ConfigError(f"{where}: model must not be empty")
    if not 1 <= cfg.jobs <= 64:
        raise ConfigError(f"{where}: jobs must be 1..64, got {cfg.jobs}")
    if not 0.0 <= cfg.heldout_fraction <= 1.0:
        raise ConfigError(f"{where}: heldout_fraction must be 0..1, got {cfg.heldout_fraction}")
    if not cfg.agent_cmd:
        raise ConfigError(f"{where}: agent_cmd must not be empty")
    if cfg.sandbox == "docker" and not (cfg.harness_src and cfg.harness_config):
        raise ConfigError(f"{where}: the docker sandbox needs harness_src and harness_config")
    return cfg


class Sandbox(Protocol):
    harness: str

    async def prepare(self) -> None: ...
    async def agent(self, run: AgentRun) -> CmdResult: ...
    async def grade(self, task: Task, entry_file: Path) -> CmdResult: ...


def make_sandbox(cfg: RigConfig) -> Sandbox:
    src = Path(cfg.harness_src).expanduser() if cfg.harness_src else None
    if cfg.sandbox == "local":
        return LocalSandbox(src)
    assert src is not None and cfg.harness_config is not None  # check_config
    return DockerSandbox(DockerSettings(
        harness_src=src, harness_config=Path(cfg.harness_config).expanduser().resolve(),
        harness_config_mount=cfg.harness_config_mount, bare_image=cfg.bare_image,
        agent_network=cfg.agent_network))


def task_split(task: Task, cfg: RigConfig) -> str:
    key = task.spec.library if cfg.split_unit == "library" else task.id
    return split_of(key, seed=cfg.split_seed, heldout=cfg.heldout_fraction)


def compose_prompt(task: Task, condition: Condition, split: str) -> str:
    """The task statement plus the facts of the environment. Facts only: how to work is the
    harness's business, and the harness is what gets evolved."""
    spec = task.spec
    lines = [(f"Write your answer to `{spec.entry}` in the current directory. Only that file "
              "is graded, by tests you cannot see.")]
    if condition.docs:
        lines.append(f"Documentation for {spec.library} is in `./docs` (read-only).")
    if condition.library:
        lines.append(f"{spec.library} is installed here, so you can run code against it.")
    else:
        lines.append(f"{spec.library} is not installed here, so you cannot run code "
                     "against it.")
    if condition.memory:
        if split == HELDOUT:
            lines.append(f"Notes from earlier tasks on {spec.library} are in `./memory` "
                         "(read-only).")
        else:
            lines.append(f"Notes from earlier tasks on {spec.library} are in `./memory`. "
                         "Before you finish, add or correct notes there on what running "
                         "code taught you about the library.")
    return "\n".join(lines) + "\n\n" + task.prompt


def _keep_session(workspace: Path, logs: Path, run_dir: Path) -> str | None:
    """The harness's own trace of the rollout, copied next to the logs: the newest session
    file kilpa wrote in the workspace. The three loops all read these. Returned relative to
    the run directory, so a run can be moved or copied whole."""
    sessions = workspace / ".agent" / "sessions"
    if not sessions.is_dir():
        return None
    files = [p for p in sessions.glob("*.jsonl") if p.is_file()]
    if not files:
        return None
    kept = logs / "session.jsonl"
    shutil.copyfile(max(files, key=lambda p: p.stat().st_mtime), kept)
    return str(kept.relative_to(run_dir))


async def _rollout(task: Task, *, condition: Condition, split: str, cfg: RigConfig,
                   run_dir: Path, sandbox: Sandbox, oracle: bool, keep: bool) -> RunRecord:
    started = time.time()
    spec = task.spec
    workspace = run_dir / "work" / condition.name / task.id
    logs = run_dir / "logs" / condition.name / task.id
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    logs.mkdir(parents=True, exist_ok=True)
    fields: dict = {}
    frozen: Path | None = None
    try:
        if oracle:
            assert task.solution is not None  # run_tasks checked
            shutil.copyfile(task.solution, workspace / spec.entry)
        else:
            prompt_file = logs / "prompt.md"
            prompt_file.write_text(compose_prompt(task, condition, split), encoding="utf-8")
            memory_dir = None
            if condition.memory:
                memory_dir = run_dir / "memory" / spec.library
                memory_dir.mkdir(parents=True, exist_ok=True)
                if split == HELDOUT:
                    # Held-out tasks read a frozen copy: nothing they do may reach the notes.
                    frozen = workspace.parent / f"{task.id}.memory"
                    shutil.rmtree(frozen, ignore_errors=True)
                    shutil.copytree(memory_dir, frozen)
                    memory_dir = frozen
            agent = await sandbox.agent(AgentRun(
                task=task, condition=condition, workspace=workspace, prompt_file=prompt_file,
                argv=cfg.agent_cmd, model=cfg.model, adapter=cfg.adapter,
                memory_dir=memory_dir, memory_readonly=split == HELDOUT))
            (logs / "agent.log").write_text(agent.output, encoding="utf-8")
            fields.update(agent_exit=agent.code, agent_timed_out=agent.timed_out)
            fields["session"] = _keep_session(workspace, logs, run_dir)

        entry = workspace / spec.entry
        if entry.is_symlink() or not entry.is_file():
            fields.update(status=FAILED, detail=f"no {spec.entry} written")
        else:
            kept = logs / spec.entry
            shutil.copyfile(entry, kept)
            grade = await sandbox.grade(task, kept)
            (logs / "grade.log").write_text(grade.output, encoding="utf-8")
            fields.update(grade_exit=grade.code, grade_timed_out=grade.timed_out,
                          status=PASSED if grade.code == 0 else FAILED)
    except Exception as exc:  # noqa: BLE001 -- a run is unattended: one rollout's crash is
        # that rollout's error line, never a reason to lose the others.
        fields.update(status=ERROR, detail=f"{type(exc).__name__}: {exc}"[:2000])
    finally:
        if frozen is not None:
            shutil.rmtree(frozen, ignore_errors=True)
        if not keep:
            shutil.rmtree(workspace, ignore_errors=True)
    return RunRecord(task_id=task.id, library=spec.library, split=split,
                     condition=condition.name, model=cfg.model, adapter=cfg.adapter,
                     harness=sandbox.harness, oracle=oracle, started_at=started,
                     elapsed_s=time.time() - started, **fields)


async def run_tasks(tasks: list[Task], *, condition: Condition, cfg: RigConfig, run_dir: Path,
                    sandbox: Sandbox, split: str = "all", oracle: bool = False,
                    rerun: bool = False, keep: bool = False, limit: int | None = None,
                    on_record: Callable[[RunRecord], None] | None = None) -> list[RunRecord]:
    """Run every selected task once and append a record for each. Returns the new records.

    A rollout already recorded for the same task, condition, model, adapter and harness is
    skipped unless `rerun`; an `error` record does not count as done.
    """
    selected = [(task, task_split(task, cfg)) for task in tasks]
    if split != "all":
        selected = [(task, s) for task, s in selected if s == split]
    if limit is not None:
        selected = selected[:limit]
    for task, _ in selected:
        if oracle and task.solution is None:
            raise ConfigError(f"{task.dir}: --oracle needs solution/{task.spec.entry}")
        if not oracle and condition.docs and task.docs_dir is None:
            raise ConfigError(f"{task.dir}: condition {condition.name!r} needs docs/, and "
                              "running it without would measure a different condition")

    await sandbox.prepare()
    if not rerun:
        done = {r.key for r in records.load(run_dir) if r.status != ERROR}
        selected = [(task, s) for task, s in selected
                    if (task.id, condition.name, cfg.model, cfg.adapter, sandbox.harness,
                        oracle) not in done]

    # Notes carry from task to task within a library, so under the memory condition a library
    # is one sequential group: its evolve tasks first (they write), then its held-out tasks
    # (they read a frozen copy). Otherwise every task is its own group.
    groups: dict[str, list[tuple[Task, str]]] = {}
    for task, s in selected:
        key = task.spec.library if condition.memory and not oracle else task.id
        groups.setdefault(key, []).append((task, s))
    for group in groups.values():
        group.sort(key=lambda item: (item[1] != EVOLVE, item[0].id))

    gate = asyncio.Semaphore(cfg.jobs)
    out: list[RunRecord] = []

    async def run_group(group: list[tuple[Task, str]]) -> None:
        for task, s in group:
            async with gate:
                record = await _rollout(task, condition=condition, split=s, cfg=cfg,
                                        run_dir=run_dir, sandbox=sandbox, oracle=oracle,
                                        keep=keep)
            records.append(run_dir, record)
            out.append(record)
            if on_record is not None:
                on_record(record)

    await asyncio.gather(*(run_group(group) for group in groups.values()))
    return out


def progress_line(record: RunRecord) -> str:
    extra = f"  {record.detail.splitlines()[0][:120]}" if record.detail else ""
    return (f"  {record.status:<6} {record.task_id}  [{record.condition}/{record.split}]  "
            f"({record.elapsed_s:.0f}s){extra}")


def print_progress(record: RunRecord) -> None:
    print(progress_line(record), file=sys.stderr, flush=True)
