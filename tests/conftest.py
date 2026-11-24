"""Fixtures: throwaway task directories and a scripted stand-in for the harness."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from nulis.runner import RigConfig

FAKE_AGENT = '''
"""A scripted agent. It "knows" the library's magic number only if it can read it: from
./docs, or from notes an earlier task left in ./memory. Modes are chosen by the model name."""
import pathlib
import re
import sys

mode = sys.argv[1]
cwd = pathlib.Path.cwd()
memory = cwd / "memory"
number = 0
for source in (cwd / "docs" / "api.md", memory / "notes.md"):
    if source.is_file():
        number = int(re.search(r"\\d+", source.read_text()).group())
        break
if memory.is_dir():
    try:
        (memory / "notes.md").write_text(f"magic number is {number}")
        with (memory / "log.txt").open("a") as handle:
            handle.write(cwd.name + "\\n")
    except OSError:
        pass
sessions = cwd / ".agent" / "sessions"
sessions.mkdir(parents=True)
(sessions / "0001.jsonl").write_text('{"role": "user"}\\n')
if mode == "silent":
    sys.exit(0)
if mode == "crash":
    sys.exit(3)
if mode == "symlink":
    (cwd / "solution.py").symlink_to(cwd / "docs" / "api.md")
    sys.exit(0)
(cwd / "solution.py").write_text(f"def answer():\\n    return {number}\\n")
if mode == "late-crash":
    sys.exit(3)
'''


@pytest.fixture
def fake_agent(tmp_path: Path) -> Path:
    path = tmp_path / "fake_agent.py"
    path.write_text(textwrap.dedent(FAKE_AGENT))
    return path


@pytest.fixture
def cfg(fake_agent: Path) -> RigConfig:
    return RigConfig(model="write", sandbox="local",
                     agent_cmd=[sys.executable, str(fake_agent), "{model}"])


@pytest.fixture
def make_task(tmp_path: Path):
    """Build a task whose answer is `magic`. `docs` and `solution` can be left out."""

    def make(task_id: str, *, library: str = "lib", magic: int = 42, docs: bool = True,
             solution: bool = True, extra: str = "") -> Path:
        root = tmp_path / "tasks" / task_id
        (root / "tests").mkdir(parents=True)
        (root / "task.toml").write_text(
            f'[task]\nid = "{task_id}"\nlibrary = "{library}"\nimage = "img:1"\n'
            'test_cmd = "python -m unittest discover -s tests -q"\n'
            f'agent_timeout_sec = 30\ngrade_timeout_sec = 30\n{extra}')
        (root / "prompt.md").write_text("Write answer() returning the library's magic number.\n")
        (root / "tests" / "test_solution.py").write_text(textwrap.dedent(f"""
            import unittest

            from solution import answer


            class T(unittest.TestCase):
                def test_answer(self):
                    self.assertEqual(answer(), {magic})
            """))
        if docs:
            (root / "docs").mkdir()
            (root / "docs" / "api.md").write_text(f"The magic number is {magic}.\n")
        if solution:
            (root / "solution").mkdir()
            (root / "solution" / "solution.py").write_text(f"def answer():\n    return {magic}\n")
        return root

    return make
