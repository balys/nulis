from pathlib import Path

import msgspec
import pytest

from nulis import records
from nulis.records import ERROR, FAILED, PASSED
from nulis.runner import ConfigError, check_config, compose_prompt, load_config, run_tasks
from nulis.sandbox import LocalSandbox, SandboxError
from nulis.splits import CONDITIONS, EVOLVE, HELDOUT
from nulis.task import load_task


async def _run(tasks, cfg, run_dir, condition="closed", **kw):
    return await run_tasks(tasks, condition=CONDITIONS[condition], cfg=cfg, run_dir=run_dir,
                           sandbox=LocalSandbox(), **kw)


async def test_docs_condition_passes_where_closed_fails(make_task, cfg, tmp_path: Path):
    tasks = [load_task(make_task("lib-0001"))]
    run_dir = tmp_path / "run"
    (closed,) = await _run(tasks, cfg, run_dir, "closed")
    (docs,) = await _run(tasks, cfg, run_dir, "docs")
    assert (closed.status, docs.status) == (FAILED, PASSED)
    assert closed.grade_exit not in (None, 0) and docs.grade_exit == 0
    assert docs.model == "write" and docs.harness == "unknown" and docs.adapter is None
    logs = run_dir / "logs" / "docs" / "lib-0001"
    assert (logs / "solution.py").read_text() == "def answer():\n    return 42\n"
    assert "Documentation for lib is in `./docs`" in (logs / "prompt.md").read_text()
    assert docs.session == "logs/docs/lib-0001/session.jsonl"
    assert (run_dir / docs.session).is_file()
    assert not (run_dir / "work" / "docs" / "lib-0001").exists()  # workspaces are cleaned up


async def test_oracle_grades_the_reference_solution_without_an_agent(make_task, cfg, tmp_path):
    tasks = [load_task(make_task("lib-0001"))]
    (record,) = await _run(tasks, cfg, tmp_path / "run", oracle=True)
    assert record.status == PASSED and record.oracle and record.agent_exit is None
    with pytest.raises(ConfigError, match="--oracle needs solution"):
        await _run([load_task(make_task("lib-0002", solution=False))], cfg, tmp_path / "run",
                   oracle=True)


async def test_only_a_real_entry_file_is_graded(make_task, cfg, tmp_path):
    tasks = [load_task(make_task("lib-0001"))]

    async def run(model: str):
        (record,) = await _run(tasks, msgspec.structs.replace(cfg, model=model),
                               tmp_path / model, "docs")
        return record

    silent = await run("silent")
    assert (silent.status, silent.detail, silent.agent_exit) == (
        FAILED, "no solution.py written", 0)
    assert silent.grade_exit is None
    crash = await run("crash")
    assert (crash.status, crash.agent_exit) == (FAILED, 3)
    # A symlink where the answer should be is no answer: it could point at anything.
    assert (await run("symlink")).detail == "no solution.py written"
    # The answer is what is graded, not the exit code: work written before a late crash counts.
    late = await run("late-crash")
    assert (late.status, late.agent_exit) == (PASSED, 3)


async def test_resume_skips_what_is_recorded_and_retries_rig_errors(make_task, cfg, tmp_path):
    tasks = [load_task(make_task("lib-0001")), load_task(make_task("lib-0002"))]
    run_dir = tmp_path / "run"

    class Broken(LocalSandbox):
        async def grade(self, task, entry_file):
            if task.id == "lib-0002":
                raise SandboxError("image build failed")
            return await super().grade(task, entry_file)

    first = await run_tasks(tasks, condition=CONDITIONS["docs"], cfg=cfg, run_dir=run_dir,
                            sandbox=Broken())
    assert {r.task_id: r.status for r in first} == {"lib-0001": PASSED, "lib-0002": ERROR}
    assert "image build failed" in next(r for r in first if r.status == ERROR).detail
    second = await _run(tasks, cfg, run_dir, "docs")
    assert [(r.task_id, r.status) for r in second] == [("lib-0002", PASSED)]
    assert await _run(tasks, cfg, run_dir, "docs") == []
    assert len(await _run(tasks, cfg, run_dir, "docs", rerun=True)) == 2
    # Another model, or another adapter on the same model, is a different measurement.
    assert len(await _run(tasks, msgspec.structs.replace(cfg, adapter="lora-1"), run_dir,
                          "docs")) == 2


async def test_records_are_filed_by_split(make_task, cfg, tmp_path):
    tasks = [load_task(make_task(f"lib-{i:04d}")) for i in range(12)]
    run_dir = tmp_path / "run"
    held = await _run(tasks, cfg, run_dir, "docs", split=HELDOUT)
    assert held and all(r.split == HELDOUT for r in held) and len(held) < len(tasks)
    assert not records.records_path(run_dir, EVOLVE).exists()
    await _run(tasks, cfg, run_dir, "docs")
    assert len(records.load(run_dir, EVOLVE)) + len(held) == len(tasks)
    assert {r.task_id for r in records.load(run_dir, HELDOUT)} == {r.task_id for r in held}
    by_library = msgspec.structs.replace(cfg, split_unit="library")
    one_split = await _run(tasks, by_library, tmp_path / "run2", "docs")
    assert len({r.split for r in one_split}) == 1  # a library is held out whole


async def test_memory_carries_notes_forward_and_held_out_cannot_write(make_task, cfg, tmp_path):
    # No docs for these tasks would be refused, so the notes are what is under test here:
    # every task has docs, and the log shows who was allowed to write.
    tasks = [load_task(make_task(f"lib-{i:04d}")) for i in range(12)]
    run_dir = tmp_path / "run"
    out = await _run(tasks, msgspec.structs.replace(cfg, jobs=4), run_dir, "memory")
    assert all(r.status == PASSED for r in out)
    order = [r.split for r in out]
    assert order == sorted(order)  # every evolve task of the library ran before any held-out
    log = (run_dir / "memory" / "lib" / "log.txt").read_text().split()
    assert sorted(log) == sorted(r.task_id for r in out if r.split == EVOLVE)
    assert "(read-only)" in (run_dir / "logs" / "memory" / next(
        r.task_id for r in out if r.split == HELDOUT) / "prompt.md").read_text()


async def test_docs_conditions_refuse_tasks_without_docs(make_task, cfg, tmp_path):
    tasks = [load_task(make_task("lib-0001", docs=False))]
    with pytest.raises(ConfigError, match="needs docs/"):
        await _run(tasks, cfg, tmp_path / "run", "docs")
    (record,) = await _run(tasks, cfg, tmp_path / "run", "closed")
    assert record.status == FAILED


def test_prompt_states_the_environment(make_task):
    task = load_task(make_task("lib-0001"))
    closed = compose_prompt(task, CONDITIONS["closed"], EVOLVE)
    assert "lib is not installed here" in closed and "./docs" not in closed
    opened = compose_prompt(task, CONDITIONS["open"], EVOLVE)
    assert "lib is installed here" in opened and "./memory" not in opened
    assert closed.endswith(task.prompt)


def test_config_is_strict(tmp_path: Path):
    path = tmp_path / "rig.toml"
    path.write_text('[rig]\nmodel = "qwen"\nsandbox = "local"\n')
    assert load_config(path).agent_cmd[0] == "kilpa"
    path.write_text('[rig]\nmodel = "qwen"\nsandbox = "local"\njob = 4\n')
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(path)
    path.write_text('[rig]\nmodel = "qwen"\n')
    with pytest.raises(ConfigError, match="needs harness_src and harness_config"):
        load_config(path)
    with pytest.raises(ConfigError, match="jobs must be"):
        check_config(msgspec.structs.replace(load_config_ok(path), jobs=0))
    with pytest.raises(ConfigError, match="no rig config"):
        load_config(tmp_path / "missing.toml")


def load_config_ok(path: Path):
    path.write_text('[rig]\nmodel = "qwen"\nsandbox = "local"\n')
    return load_config(path)
