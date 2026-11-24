"""`nulis validate | run | report` (docs/RIG.md).

Exit codes: 0 done; 1 done, but something needs attention (a rig error, an oracle that did
not pass, a gate finding); 2 could not start.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

import msgspec

from nulis import records as records_mod
from nulis.records import ERROR, PASSED
from nulis.report import gate, render, summarize
from nulis.runner import (
    ConfigError,
    check_config,
    load_config,
    make_sandbox,
    print_progress,
    run_tasks,
    task_split,
)
from nulis.sandbox import SandboxError
from nulis.splits import CONDITIONS, SPLITS
from nulis.task import TaskError, load_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nulis", description="The Nulis benchmark rig")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="check every task directory and show the splits")
    validate.add_argument("tasks", metavar="TASKS_DIR")
    validate.add_argument("--config", default="rig.toml", metavar="PATH")

    run = sub.add_parser("run", help="run tasks under one condition and record the results")
    run.add_argument("tasks", metavar="TASKS_DIR")
    run.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    run.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    run.add_argument("--config", default="rig.toml", metavar="PATH")
    run.add_argument("--run-dir", default=None, metavar="DIR",
                     help="where records, logs and notes go (default: runs/<timestamp>); "
                          "pass an existing one to resume it")
    run.add_argument("--model", default=None, help="override the config's model")
    run.add_argument("--adapter", default=None, help="override the config's adapter label")
    run.add_argument("--jobs", type=int, default=None, metavar="N")
    run.add_argument("--limit", type=int, default=None, metavar="N",
                     help="run only the first N selected tasks")
    run.add_argument("--oracle", action="store_true",
                     help="grade the reference solutions instead of running the agent")
    run.add_argument("--rerun", action="store_true",
                     help="run tasks again even when a record exists")
    run.add_argument("--keep", action="store_true", help="keep the rollout workspaces")
    run.add_argument("--json", action="store_true", help="one JSON line per record on stdout")

    report = sub.add_parser("report", help="pass rates and the step 1 gate for a run directory")
    report.add_argument("run_dir", metavar="RUN_DIR")
    report.add_argument("--split", choices=SPLITS, default=None,
                        help="report one split only (a proposer is shown evolve only)")
    report.add_argument("--ceiling", type=float, default=0.9, metavar="RATE",
                        help="open-book pass rate at which the gate fires (default 0.9)")
    report.add_argument("--min-n", type=int, default=30, metavar="N",
                        help="fewest tasks on which the ceiling is judged (default 30)")
    report.add_argument("--json", action="store_true")
    return parser


def _config(args: argparse.Namespace):
    cfg = load_config(Path(args.config))
    updates = {name: value for name in ("model", "adapter", "jobs")
               if (value := getattr(args, name, None)) is not None}
    return check_config(msgspec.structs.replace(cfg, **updates) if updates else cfg)


def _validate(args: argparse.Namespace) -> int:
    tasks = load_tasks(Path(args.tasks))
    cfg = load_config(Path(args.config)) if Path(args.config).is_file() else None
    print(f"{len(tasks)} task(s) under {args.tasks}")
    by_library = Counter(task.spec.library for task in tasks)
    for library, count in sorted(by_library.items()):
        subset = [task for task in tasks if task.spec.library == library]
        parts = [f"{count} tasks", f"{sum(t.docs_dir is not None for t in subset)} with docs",
                 f"{sum(t.solution is not None for t in subset)} with a solution"]
        if cfg is not None:
            splits = Counter(task_split(task, cfg) for task in subset)
            parts += [f"{splits[name]} {name}" for name in SPLITS]
        print(f"  {library}: " + ", ".join(parts))
    if cfg is None:
        print(f"(no {args.config}: splits not shown)")
    return 0


async def _run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    tasks = load_tasks(Path(args.tasks))
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    sandbox = make_sandbox(cfg)
    if cfg.sandbox == "docker" and cfg.agent_network == "bridge" and not args.oracle:
        print("nulis: agent_network is 'bridge': rollouts can reach the internet, so "
              "closed-book results are not trustworthy (docs/RIG.md)", file=sys.stderr)
    print(f"nulis: {len(tasks)} task(s), condition {args.condition}, split {args.split}, "
          f"model {cfg.model}, harness {sandbox.harness}, records in {run_dir}", file=sys.stderr)

    def on_record(record) -> None:
        print_progress(record)
        if args.json:
            print(msgspec.json.encode(record).decode(), flush=True)

    new = await run_tasks(tasks, condition=CONDITIONS[args.condition], cfg=cfg, run_dir=run_dir,
                          sandbox=sandbox, split=args.split, oracle=args.oracle,
                          rerun=args.rerun, keep=args.keep, limit=args.limit,
                          on_record=on_record)
    passed = sum(r.status == PASSED for r in new)
    errors = sum(r.status == ERROR for r in new)
    print(f"nulis: {len(new)} rollout(s): {passed} passed, {len(new) - passed - errors} failed, "
          f"{errors} rig errors", file=sys.stderr)
    return 1 if errors or (args.oracle and passed < len(new)) else 0


def _report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise ConfigError(f"{run_dir}: no such run directory")
    rows = summarize(records_mod.load(run_dir, args.split))
    findings = gate(rows, ceiling=args.ceiling, min_n=args.min_n)
    if args.json:
        print(msgspec.json.encode({"rows": rows, "gate": findings}).decode())
    else:
        print(render(rows))
        for finding in findings:
            print(f"GATE: {finding}")
    return 1 if findings else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "report":
            return _report(args)
        return asyncio.run(_run(args))
    except (ConfigError, TaskError, SandboxError) as exc:
        print(f"nulis: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
