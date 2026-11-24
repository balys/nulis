# The rig

The rig answers one question, repeatedly: on these tasks, under this condition, with these
weights and this harness, what fraction passes? Every number the project reports comes from
it, so its rules are stricter than the harness's.

## A task

One directory per task. `task.toml` is the whole contract; unknown fields are refused.

```
tasks/<suite>/<task>/
  task.toml
  prompt.md             what the agent is asked to write
  tests/                hidden tests; only the grading sandbox sees them
  docs/                 optional: library documentation, for the open-book conditions
  solution/<entry>      optional: the reference answer, used only by --oracle
```

```toml
[task]
id = "ndonnx-0001"                # lowercase letters, digits, ., _ and -
library = "ndonnx"
image = "nulis/ndonnx:0.9"        # the library and the test runner, installed
entry = "solution.py"             # the one file that is graded (default)
library_version = "0.9.0"         # optional
released = "2024-06-01"           # optional: orders the library stream
test_cmd = "python -m pytest -q -p no:cacheprovider tests"   # default
agent_timeout_sec = 1800          # default
grade_timeout_sec = 300           # default
```

`tasks/example/slug-0001` is a smoke task that needs no library. It checks the pipeline, not
the model.

## Conditions

Grading is the same under every condition. What changes is what the agent phase is given.

| condition | docs readable | library installed | notes across tasks | what it measures |
|---|---|---|---|---|
| `closed` | no | no | no | what the weights know |
| `docs` | yes | no | no | lookup |
| `open` | yes | yes | no | lookup plus execution feedback |
| `memory` | yes | yes | yes | open, plus notes carried forward |

"Closed-book after study" is `closed` again, run with the adapter the study produced: pass
`--adapter <label>` and a `--model` name that the server resolves to those weights.

Under `memory` the tasks of one library run in sequence. Evolve tasks run first and may write
`./memory`. Held-out tasks run afterwards and get a frozen, read-only copy.

The prompt the agent receives is `prompt.md` plus the facts of the environment: where to
write, whether docs and the library are present, whether notes exist. It says nothing about
how to work. That is the harness's business, and the harness is what gets evolved.

## Splits

A task's split is a hash of `split_seed` and its id, so adding tasks never moves old ones.
Set `split_unit = "library"` to hold out whole libraries instead.

Records go to `records.evolve.jsonl` and `records.heldout.jsonl`, in separate files on
purpose. Anything that proposes harness edits or training recipes is given the evolve file
and nothing else. `runs/` is in `.gitignore` for the same reason.

## What the sandbox guarantees

- **Agent phase.** The workspace holds the prompt and, by condition, `docs/` and `memory/`.
  `tests/` and `solution/` are never mounted.
- **Grade phase.** A fresh container with `--network none`. One file leaves the workspace,
  the task's `entry`, and it is mounted read-only beside the read-only tests. A symlink in
  its place counts as no answer.
- **Identity.** Every record carries the model name, the adapter label and the harness
  commit, with a digest of any uncommitted change. The agent image is tagged by that
  identity, so an edited harness can never run from a stale image.
- **Failures.** `failed` is the model's result. `error` means the rig could not run the
  rollout; it is left out of every rate and retried on the next run.

`sandbox = "local"` gives none of this. It runs the agent command and the tests as you, and
exists for the rig's own tests and for `--oracle` checks of a task directory.

## Setting up a GPU machine

Nothing below has been run by the author of this file yet. Run the oracle check first; it
costs no model time and fails loudly when mounts or images are wrong.

1. **Serve the model** with an OpenAI-compatible server (vLLM, LM Studio).

2. **The agent network.** Rollouts must reach the model server and nothing else. Otherwise a
   closed-book run can `pip install` the library or fetch its docs.

   ```bash
   docker network create --internal nulis-rig
   docker run -d --name nulis-llm --restart unless-stopped --network nulis-rig \
     alpine/socat tcp-listen:8000,fork,reuseaddr tcp-connect:<MODEL_HOST>:<MODEL_PORT>
   docker network connect bridge nulis-llm
   ```

   The internal network has no route out. `nulis-llm` sits on both networks and forwards one
   port, so inside a rollout the model is `http://nulis-llm:8000/v1`.

3. **The harness config.** A directory that is mounted read-only as kilpa's config. Keep it
   apart from your own `~/.config/kilpa`: a rollout must hold no real credentials.

   ```toml
   # harness-config/models.toml
   [models."qwen3.8-27b"]
   provider = "openai"
   base_url = "http://nulis-llm:8000/v1"
   window = 65536
   thinking = "none"
   ```

   ```toml
   # harness-config/credentials.toml   (chmod 600; any non-empty value)
   [keys]
   openai = "local"
   ```

4. **`rig.toml`**, next to where you run `nulis`:

   ```toml
   [rig]
   model = "qwen3.8-27b"
   sandbox = "docker"
   harness_src = "~/src/kilpa"            # a git checkout; its commit is recorded
   harness_config = "./harness-config"
   agent_network = "nulis-rig"
   jobs = 4                               # rollouts in flight
   # adapter = "lora-ndonnx-001"          # label for the weights on top of the base
   # agent_cmd = ["kilpa", "-p", "{prompt}", "--mode", "auto", "--model", "{model}"]
   # bare_image = "python:3.12-slim"      # the agent's image when the library is withheld
   # split_seed = "nulis-v1"
   # heldout_fraction = 0.5
   # split_unit = "task"                  # or "library"
   ```

5. **Run.**

   ```bash
   nulis validate tasks
   nulis run tasks --condition closed --oracle --run-dir runs/base   # must pass everything
   nulis run tasks --condition closed --run-dir runs/base
   nulis run tasks --condition docs   --run-dir runs/base
   nulis run tasks --condition open   --run-dir runs/base
   nulis report runs/base
   ```

   A run directory can be resumed: rollouts already recorded for the same task, condition,
   model, adapter and harness are skipped. `--rerun` repeats them.

## The report and its gate

`nulis report` prints pass rates with 95% Wilson intervals, and exits 1 when a gate fires:

- an `--oracle` run that did not pass everything, which means the tasks or the sandbox are
  broken and no other number can be trusted;
- an open-book condition (`docs` or `open`) at or above `--ceiling` (default 0.9), which is
  step 1's stop condition: the base model with documentation already solves the benchmark,
  so self-study has little to add there.

## Checking the rig without a GPU

`tools/fake_model.py` is a stand-in model server. Its first reply calls the harness's `write`
tool with a file you name, and its second reply ends the turn. With it, one command exercises
the whole chain: rig, harness, tool call, grading, record.

```bash
python tools/fake_model.py 18080 tasks/example/slug-0001/solution/solution.py &
# models.toml: a row "fake" with provider = "openai", base_url = "http://127.0.0.1:18080/v1"
# rig.toml:    model = "fake", sandbox = "local", harness_src = "<kilpa checkout>"
OPENAI_API_KEY=local nulis run tasks/example --condition docs
```

This is how the rig was first checked against kilpa: the rollout passed, and the record
carried kilpa's commit and its session file.

## Known limits

- Containers run as root, so files a rollout leaves in a workspace are root-owned on the
  host. Workspaces are removed after each rollout on a best-effort basis.
- The agent image is the base image plus `pip install` of the harness. A base image without
  Python 3.12 and pip cannot host kilpa.
- `memory` gives the agent a notes directory and one sentence about it. Proper execution
  memory belongs in kilpa, and will replace that sentence.
