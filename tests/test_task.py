from pathlib import Path

import pytest

from nulis.task import TaskError, load_task, load_tasks


def test_loads_a_valid_task(make_task):
    task = load_task(make_task("lib-0001"))
    assert task.id == "lib-0001"
    assert task.spec.entry == "solution.py"
    assert task.docs_dir is not None and task.solution is not None
    assert "magic number" in task.prompt


def test_optional_parts_may_be_absent(make_task):
    task = load_task(make_task("lib-0002", docs=False, solution=False))
    assert task.docs_dir is None and task.solution is None


@pytest.mark.parametrize("extra, message", [
    ('test_cmnd = "x"\n', "unknown field"),
    ('entry = "../tests/test_solution.py"\n', "bare file name"),
    ('entry = "sub/solution.py"\n', "bare file name"),
])
def test_refuses_a_broken_contract(make_task, extra, message):
    with pytest.raises(TaskError, match=message):
        load_task(make_task("lib-0003", extra=extra))


def test_refuses_a_bad_id_and_missing_parts(make_task, tmp_path: Path):
    root = make_task("lib-0004")
    (root / "task.toml").write_text('[task]\nid = "Bad Id"\nlibrary = "l"\nimage = "i"\n')
    with pytest.raises(TaskError, match="must match"):
        load_task(root)
    (root / "task.toml").write_text(
        '[task]\nid = "lib-0004"\nlibrary = "l"\nimage = "i"\ngrade_timeout_sec = 0\n')
    with pytest.raises(TaskError, match="timeouts must be positive"):
        load_task(root)
    root = make_task("lib-0005")
    (root / "tests" / "test_solution.py").unlink()
    with pytest.raises(TaskError, match="tests/ is missing or empty"):
        load_task(root)
    with pytest.raises(TaskError, match="no task.toml"):
        load_task(tmp_path / "nowhere")


def test_load_tasks_sorts_and_refuses_duplicates(make_task, tmp_path: Path):
    make_task("lib-b")
    make_task("lib-a")
    assert [task.id for task in load_tasks(tmp_path / "tasks")] == ["lib-a", "lib-b"]
    clash = make_task("lib-c")
    (clash / "task.toml").write_text('[task]\nid = "lib-a"\nlibrary = "l"\nimage = "i"\n')
    with pytest.raises(TaskError, match="duplicate task id"):
        load_tasks(tmp_path / "tasks")


def test_the_example_task_is_valid():
    tasks = load_tasks(Path(__file__).resolve().parents[1] / "tasks")
    assert [task.id for task in tasks] == ["example-slug-0001"]
