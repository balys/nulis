import json
import sys
from pathlib import Path

from nulis.cli import main


def _config(tmp_path: Path, fake_agent: Path, model: str = "write") -> str:
    path = tmp_path / "rig.toml"
    path.write_text(f'[rig]\nmodel = "{model}"\nsandbox = "local"\n'
                    f'agent_cmd = ["{sys.executable}", "{fake_agent}", "{{model}}"]\n')
    return str(path)


def test_validate_run_and_report(make_task, fake_agent, tmp_path: Path, capsys):
    for i in range(6):
        make_task(f"lib-{i:04d}")
    tasks, config, run_dir = str(tmp_path / "tasks"), _config(tmp_path, fake_agent), str(
        tmp_path / "run")
    assert main(["validate", tasks, "--config", config]) == 0
    assert "lib: 6 tasks, 6 with docs, 6 with a solution" in capsys.readouterr().out

    assert main(["run", tasks, "--condition", "closed", "--oracle", "--config", config,
                 "--run-dir", run_dir]) == 0
    assert main(["run", tasks, "--condition", "closed", "--config", config,
                 "--run-dir", run_dir]) == 0
    assert main(["run", tasks, "--condition", "docs", "--config", config, "--run-dir", run_dir,
                 "--json"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 6 and all(line["status"] == "passed" for line in lines)

    # Six tasks are too few to judge headroom on, so the gate stays quiet by default.
    assert main(["report", run_dir]) == 0
    assert "closed (oracle)" in capsys.readouterr().out
    # Told to judge anyway: docs solve everything here, and the exit code says so.
    assert main(["report", run_dir, "--min-n", "1"]) == 1
    out = capsys.readouterr().out
    assert "GATE:" in out and "little headroom" in out
    assert main(["report", run_dir, "--split", "evolve", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert {row["split"] for row in payload["rows"]} == {"evolve"}


def test_exit_codes_for_runs_that_cannot_start(make_task, fake_agent, tmp_path: Path, capsys):
    make_task("lib-0001", docs=False)
    tasks, config = str(tmp_path / "tasks"), _config(tmp_path, fake_agent)
    assert main(["run", tasks, "--condition", "docs", "--config", config]) == 2
    assert "needs docs/" in capsys.readouterr().err
    assert main(["run", tasks, "--condition", "closed", "--config", "nope.toml"]) == 2
    assert main(["report", str(tmp_path / "no-run")]) == 2
    assert main(["validate", str(tmp_path / "no-tasks")]) == 2
