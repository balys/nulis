"""Where rollouts run: Docker for real runs, a plain subprocess for tests and smoke checks.

Two phases, and the split between them is the anti-cheat:

  agent  the harness works in a scratch workspace. It sees the prompt and, by condition, the
         docs, the library and a notes directory. It never sees tests/ or solution/.
  grade  a fresh container with no network. Exactly one file leaves the workspace -- the
         task's `entry` -- and is mounted read-only beside the read-only tests.

The Docker backend drives the `docker` CLI as a subprocess, the same posture as kilpa's
DeepSWE runner: no SDK, and every argv is built by a pure function that tests can read.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from nulis.splits import Condition
from nulis.task import Task

OUTPUT_LIMIT = 100_000
"""Per-command output kept, head and tail. A log is for diagnosis, not an archive."""


class SandboxError(Exception):
    """The sandbox itself could not do its job (no docker, a failed image build)."""


@dataclass(frozen=True)
class CmdResult:
    output: str
    code: int | None
    timed_out: bool = False


def _cap(text: str, limit: int = OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters dropped] ...\n{text[-half:]}"


async def run_cmd(argv: list[str], *, timeout: float, cwd: Path | None = None,
                  env: dict[str, str] | None = None, stdin: bytes | None = None) -> CmdResult:
    """One command, stdout and stderr merged. On timeout the whole process group is killed:
    an agent or a test run that spawned children must not leave them behind."""
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env=env, start_new_session=True,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        raw, _ = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    except TimeoutError:
        _kill_group(proc)
        await proc.wait()
        return CmdResult(output=f"timed out after {timeout:.0f}s", code=None, timed_out=True)
    except asyncio.CancelledError:
        _kill_group(proc)
        raise
    return CmdResult(output=_cap(raw.decode("utf-8", "replace")), code=proc.returncode)


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@dataclass(frozen=True)
class AgentRun:
    """Everything one agent phase needs. `argv` is the command template from the rig config."""

    task: Task
    condition: Condition
    workspace: Path
    prompt_file: Path
    argv: list[str]
    model: str
    adapter: str | None = None
    memory_dir: Path | None = None
    memory_readonly: bool = False


def render_argv(template: list[str], *, prompt: str, prompt_file: str, model: str,
                adapter: str | None) -> list[str]:
    """Fill the command template. A token that is exactly `{prompt}` becomes the prompt text
    whole; the other placeholders are substituted inside tokens. No str.format: prompt text
    is full of braces."""
    out = []
    for token in template:
        if token == "{prompt}":
            out.append(prompt)
            continue
        out.append(token.replace("{prompt_file}", prompt_file).replace("{model}", model)
                   .replace("{adapter}", adapter or ""))
    return out


def harness_id(src: Path | None) -> str:
    """The identity of the harness a rollout ran on: commit, plus a digest of any uncommitted
    change. Every record carries it, so a result can always be tied to the code that made it."""
    if src is None:
        return "unknown"

    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True,
                              timeout=30, check=False)
        return done.stdout if done.returncode == 0 else ""

    head = git("rev-parse", "--short=12", "HEAD").strip()
    if not head:
        return "unknown"
    dirty = git("status", "--porcelain")
    if not dirty.strip():
        return f"{src.name}@{head}"
    digest = hashlib.sha256((dirty + git("diff", "HEAD")).encode()).hexdigest()[:8]
    return f"{src.name}@{head}+dirty.{digest}"


class LocalSandbox:
    """No isolation at all: the agent command and the tests run as you, on this machine.
    For the rig's own tests and for smoke-checking a task directory with --oracle. Never for
    a model's rollouts."""

    def __init__(self, harness_src: Path | None = None) -> None:
        self.harness = harness_id(harness_src)

    async def prepare(self) -> None:
        return None

    async def agent(self, run: AgentRun) -> CmdResult:
        if run.condition.docs and run.task.docs_dir is not None:
            shutil.copytree(run.task.docs_dir, run.workspace / "docs", dirs_exist_ok=True)
        if run.memory_dir is not None:
            link = run.workspace / "memory"
            if not link.exists():
                link.symlink_to(run.memory_dir, target_is_directory=True)
        argv = render_argv(run.argv, prompt=run.prompt_file.read_text(encoding="utf-8"),
                           prompt_file=str(run.prompt_file), model=run.model,
                           adapter=run.adapter)
        return await run_cmd(argv, timeout=run.task.spec.agent_timeout_sec, cwd=run.workspace)

    async def grade(self, task: Task, entry_file: Path) -> CmdResult:
        with tempfile.TemporaryDirectory(prefix="nulis-grade-") as tmp:
            root = Path(tmp)
            shutil.copyfile(entry_file, root / task.spec.entry)
            shutil.copytree(task.tests_dir, root / "tests")
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            # The task's own command names `python`; locally that must be this interpreter.
            env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
            return await run_cmd(["sh", "-c", task.spec.test_cmd], cwd=root, env=env,
                                 timeout=task.spec.grade_timeout_sec)


@dataclass(frozen=True)
class DockerSettings:
    harness_src: Path
    """A source checkout of the harness; installed into the agent image with pip."""
    harness_config: Path
    """Mounted read-only as the harness's config directory. It must hold only what a rollout
    may know: the local model's row and a dummy key. Never your real credentials."""
    harness_config_mount: str = "/root/.config/kilpa"
    bare_image: str = "python:3.12-slim"
    """The agent's base image in conditions without the library."""
    agent_network: str = "nulis-rig"
    """Docker network for the agent phase. It should be an --internal network whose only
    other member forwards to the model server (docs/RIG.md); `bridge` gives the agent the
    whole internet, which makes every closed-book number meaningless."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class DockerSandbox:
    def __init__(self, settings: DockerSettings) -> None:
        self.settings = settings
        self.harness = harness_id(settings.harness_src)
        self._images: dict[str, asyncio.Task[str]] = {}

    async def prepare(self) -> None:
        if self.harness == "unknown":
            raise SandboxError(f"{self.settings.harness_src} is not a git checkout: the rig "
                               "records the harness commit with every result")
        if not self.settings.harness_config.is_dir():
            raise SandboxError(f"{self.settings.harness_config}: no harness config directory")
        done = await run_cmd(["docker", "version", "--format", "{{.Server.Version}}"], timeout=30)
        if done.code != 0:
            raise SandboxError(f"docker is not available: {done.output.strip()[-500:]}")
        done = await run_cmd(["docker", "network", "inspect", self.settings.agent_network],
                             timeout=30)
        if done.code != 0:
            raise SandboxError(f"docker network {self.settings.agent_network!r} does not exist "
                               "(docs/RIG.md, 'The agent network')")

    def agent_image_tag(self, base: str) -> str:
        return f"nulis-agent-{_slug(base)}:{_slug(self.harness)}"

    async def _build(self, base: str) -> str:
        tag = self.agent_image_tag(base)
        if (await run_cmd(["docker", "image", "inspect", tag], timeout=30)).code == 0:
            return tag
        dockerfile = (f"FROM {base}\nCOPY . /opt/harness-src\n"
                      "RUN pip install --no-cache-dir --quiet /opt/harness-src\nWORKDIR /work\n")
        done = await run_cmd(["docker", "build", "-f", "-", "-t", tag,
                              str(self.settings.harness_src)],
                             timeout=1800, stdin=dockerfile.encode())
        if done.code != 0:
            raise SandboxError(f"building {tag} failed:\n{done.output[-2000:]}")
        return tag

    async def agent_image(self, base: str) -> str:
        """Base image plus the harness, built once per base and harness version. Concurrent
        rollouts share the one build."""
        if base not in self._images:
            self._images[base] = asyncio.ensure_future(self._build(base))
        return await self._images[base]

    def agent_argv(self, run: AgentRun, image: str, name: str | None = None) -> list[str]:
        s = self.settings
        argv = ["docker", "run", "--rm", *(["--name", name] if name else []),
                "--network", s.agent_network,
                "-v", f"{run.workspace}:/work",
                "-v", f"{run.prompt_file}:/task/prompt.md:ro",
                "-v", f"{s.harness_config}:{s.harness_config_mount}:ro"]
        if run.condition.docs and run.task.docs_dir is not None:
            argv += ["-v", f"{run.task.docs_dir}:/work/docs:ro"]
        if run.memory_dir is not None:
            mode = ":ro" if run.memory_readonly else ""
            argv += ["-v", f"{run.memory_dir}:/work/memory{mode}"]
        argv += ["-w", "/work", image]
        return argv + render_argv(run.argv, prompt=run.prompt_file.read_text(encoding="utf-8"),
                                  prompt_file="/task/prompt.md", model=run.model,
                                  adapter=run.adapter)

    def grade_argv(self, task: Task, entry_file: Path, name: str | None = None) -> list[str]:
        return ["docker", "run", "--rm", *(["--name", name] if name else []),
                "--network", "none",
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-v", f"{entry_file}:/grade/{task.spec.entry}:ro",
                "-v", f"{task.tests_dir}:/grade/tests:ro",
                "-w", "/grade", task.spec.image, "sh", "-c", task.spec.test_cmd]

    async def agent(self, run: AgentRun) -> CmdResult:
        base = run.task.spec.image if run.condition.library else self.settings.bare_image
        image = await self.agent_image(base)
        name = _container_name()
        return await self._run(self.agent_argv(run, image, name), name,
                               run.task.spec.agent_timeout_sec)

    async def grade(self, task: Task, entry_file: Path) -> CmdResult:
        name = _container_name()
        return await self._run(self.grade_argv(task, entry_file, name), name,
                               task.spec.grade_timeout_sec)

    async def _run(self, argv: list[str], name: str, timeout: float) -> CmdResult:
        """Killing the `docker run` client does not stop the container, so a timed-out or
        cancelled phase is killed by name as well."""
        try:
            done = await run_cmd(argv, timeout=timeout)
        except asyncio.CancelledError:
            _kill_container(name)
            raise
        if done.timed_out:
            await run_cmd(["docker", "kill", name], timeout=30)
        return done


def _container_name() -> str:
    return f"nulis-{os.urandom(6).hex()}"


def _kill_container(name: str) -> None:
    """Fire and forget: called while a cancellation is unwinding, where nothing may await."""
    try:
        subprocess.Popen(["docker", "kill", name], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except OSError:
        pass
