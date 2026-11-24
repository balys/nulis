# nulis

Nulis is a coding model that keeps learning after release: Qwen3.8-27B, a harness that
evolves ([kilpa](https://github.com/balys/kilpa)), and nightly weight updates from its own
practice. "Nulis" is Lithuanian for zero: it learns a new library from zero.

This repository is the **rig**, the evaluator side of the project. It defines the tasks, runs
rollouts in a sandbox, grades them, and records each result against the model, adapter and
harness version that produced it. The harness lives in its own repository on purpose: the
thing being improved must never be able to edit its own tests.

```bash
uv sync --dev
uv run nulis validate tasks
uv run nulis run tasks --condition closed --oracle   # the pipeline check, before any model run
uv run nulis run tasks --condition closed
uv run nulis run tasks --condition docs --run-dir runs/<the same run>
uv run nulis report runs/<the same run>
```

How to set it up and what each part guarantees: [docs/RIG.md](docs/RIG.md).

## Status

Step 1 of the plan, in progress. Built and tested: the task format, evolve and held-out
splits, the four conditions, the rollout loop, records and the report with its gate. Not yet
done: importing the real benchmarks, and a first run on a GPU machine. The Docker path has
unit tests on the commands it builds, and has not yet been run against a Docker daemon.
