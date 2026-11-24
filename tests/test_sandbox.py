import subprocess
import sys
import time
from pathlib import Path

from nulis.sandbox import (
    AgentRun,
    DockerSandbox,
    DockerSettings,
    harness_id,
    render_argv,
    run_cmd,
)
from nulis.splits import CONDITIONS
from nulis.task import load_task


def test_render_argv_keeps_braces_in_the_prompt():
    argv = render_argv(["h", "-p", "{prompt}", "--model", "{model}", "--f={prompt_file}"],
                       prompt="use {x} and {model}", prompt_file="/t/p.md", model="m1",
                       adapter=None)
    assert argv == ["h", "-p", "use {x} and {model}", "--model", "m1", "--f=/t/p.md"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


def test_harness_id_names_the_commit_and_any_dirt(tmp_path: Path):
    assert harness_id(None) == "unknown"
    assert harness_id(tmp_path) == "unknown"  # not a checkout
    repo = tmp_path / "kilpa"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "one")
    clean = harness_id(repo)
    assert clean.startswith("kilpa@") and "dirty" not in clean
    (repo / "a.py").write_text("x = 2\n")
    dirty = harness_id(repo)
    assert dirty.startswith(clean + "+dirty.")
    (repo / "a.py").write_text("x = 3\n")
    assert harness_id(repo) != dirty  # a different edit is a different harness


async def test_run_cmd_reports_output_exit_and_timeout():
    done = await run_cmd([sys.executable, "-c", "import sys; print('hi'); sys.exit(4)"],
                         timeout=30)
    assert (done.output.strip(), done.code, done.timed_out) == ("hi", 4, False)
    started = time.monotonic()
    done = await run_cmd([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.5)
    assert done.timed_out and done.code is None
    assert time.monotonic() - started < 10


def _docker(tmp_path: Path) -> DockerSandbox:
    return DockerSandbox(DockerSettings(harness_src=tmp_path / "kilpa",
                                        harness_config=tmp_path / "hc"))


def test_docker_agent_argv_mounts_only_what_the_condition_allows(make_task, tmp_path: Path):
    task = load_task(make_task("lib-0001"))
    prompt = tmp_path / "prompt.md"
    prompt.write_text("do it")
    sandbox = _docker(tmp_path)

    def argv(condition: str, **kw) -> list[str]:
        run = AgentRun(task=task, condition=CONDITIONS[condition], workspace=tmp_path / "w",
                       prompt_file=prompt, argv=["kilpa", "-p", "{prompt}", "--model", "{model}"],
                       model="qwen", **kw)
        return sandbox.agent_argv(run, "agent-image", "c1")

    closed = argv("closed")
    joined = " ".join(closed)
    assert closed[:7] == ["docker", "run", "--rm", "--name", "c1", "--network", "nulis-rig"]
    assert f"{tmp_path / 'hc'}:/root/.config/kilpa:ro" in closed
    assert "/work/docs" not in joined and "/work/memory" not in joined
    assert closed[-6:] == ["agent-image", "kilpa", "-p", "do it", "--model", "qwen"]
    # Whatever the condition, the agent phase never sees the tests or the reference answer.
    for name in CONDITIONS:
        text = " ".join(argv(name, memory_dir=tmp_path / "mem"))
        assert str(task.tests_dir) not in text and "solution/" not in text
    assert f"{task.docs_dir}:/work/docs:ro" in argv("docs")
    assert f"{tmp_path / 'mem'}:/work/memory" in argv("memory", memory_dir=tmp_path / "mem")
    assert f"{tmp_path / 'mem'}:/work/memory:ro" in argv(
        "memory", memory_dir=tmp_path / "mem", memory_readonly=True)


def test_docker_grade_argv_has_no_network_and_read_only_inputs(make_task, tmp_path: Path):
    task = load_task(make_task("lib-0001"))
    argv = _docker(tmp_path).grade_argv(task, tmp_path / "kept" / "solution.py")
    assert argv[argv.index("--network") + 1] == "none"
    assert f"{tmp_path / 'kept' / 'solution.py'}:/grade/solution.py:ro" in argv
    assert f"{task.tests_dir}:/grade/tests:ro" in argv
    assert argv[-4:] == ["img:1", "sh", "-c", task.spec.test_cmd]


def test_agent_image_tag_changes_with_the_harness(tmp_path: Path):
    sandbox = _docker(tmp_path)
    sandbox.harness = "kilpa@abc123"
    one = sandbox.agent_image_tag("python:3.12-slim")
    sandbox.harness = "kilpa@abc123+dirty.9f"
    assert one == "nulis-agent-python-3-12-slim:kilpa-abc123"
    assert sandbox.agent_image_tag("python:3.12-slim") != one
