"""Pure-function tests for the driver. Run: python3 test/run-tests.py"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("run", Path(__file__).resolve().parents[1] / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

failures = 0


def check(name, cond, detail=""):
    global failures
    print(("ok   " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures += 1


# model mapping
m, e = run.model_args("openai/gpt-4.1-mini", "https://openrouter.ai/api/v1", "sk-or-x")
check("openrouter base → openrouter/ id", m == "openrouter/openai/gpt-4.1-mini" and e == {"OPENROUTER_API_KEY": "sk-or-x"})
m, e = run.model_args("gpt-4.1-mini", "", "sk-or-x")
check("sk-or key, no base → openrouter with inferred vendor", m == "openrouter/openai/gpt-4.1-mini")
m, e = run.model_args("openrouter/anthropic/claude-haiku-4-5", "https://openrouter.ai/api/v1", "k")
check("openrouter/ prefix not doubled", m == "openrouter/anthropic/claude-haiku-4-5")
m, e = run.model_args("my-ft-model", "https://api.together.xyz/v1", "tk")
check("explicit base → openai/ compatible", m == "openai/my-ft-model" and e == {"OPENAI_API_KEY": "tk", "OPENAI_API_BASE": "https://api.together.xyz/v1", "OPENAI_BASE_URL": "https://api.together.xyz/v1"})
m, e = run.model_args("openai/gpt-4.1-mini", "https://openrouter.ai/api/v1", "sk-or-x", agent="openclaw")
check("openclaw reaches OpenRouter as the openai provider", m == "openai/openai/gpt-4.1-mini" and e["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1" and e["OPENAI_API_KEY"] == "sk-or-x")
m, e = run.model_args("claude-sonnet-4-5", "", "ak")
check("no base, claude → anthropic", m == "anthropic/claude-sonnet-4-5" and e == {"ANTHROPIC_API_KEY": "ak"})
m, e = run.model_args("gpt-5", "", "ok")
check("no base, gpt → openai", m == "openai/gpt-5" and e == {"OPENAI_API_KEY": "ok"})
try:
    run.model_args("mistral-large", "", "k")
    check("unknown vendor refuses", False)
except ValueError as ex:
    check("unknown vendor refuses", "MODEL_API_BASE" in str(ex))
m, e = run.model_args("", "", "k", raw_override="gemini/gemini-2.5-pro")
check("HARBOR_MODEL override passes through", m == "gemini/gemini-2.5-pro" and "OPENAI_API_KEY" in e)

# command
cmd = run.build_command({"dataset": "terminal-bench/terminal-bench-2-1", "env": "modal", "agent": "terminus-2",
                         "model": "openrouter/openai/gpt-5", "concurrency": 4, "jobs_dir": Path("jobs"),
                         "n_tasks": 10, "task_names": ["hello*", "fix-git"], "attempts": 3})
check("command shape", cmd == ["harbor", "run", "-d", "terminal-bench/terminal-bench-2-1", "-e", "modal", "-a", "terminus-2",
                                "-m", "openrouter/openai/gpt-5", "-n", "4", "-o", "jobs", "-y", "-q", "-l", "10",
                                "-i", "hello*", "-i", "fix-git", "-k", "3"], str(cmd))
cmd = run.build_command({"dataset": "d", "env": "docker", "agent": "oracle", "model": "m", "concurrency": 1, "jobs_dir": Path("j"), "n_tasks": None, "task_names": [], "attempts": 1})
check("no optional flags when unset", "-l" not in cmd and "-i" not in cmd and "-k" not in cmd)

# aggregation
with tempfile.TemporaryDirectory() as d:
    job = Path(d)
    def trial(name, reward=None, exc=None, rewards=None):
        (job / name).mkdir()
        r = {"task_name": name.split("__")[0], "trial_name": name, "verifier_result": {"rewards": rewards if rewards is not None else ({"reward": reward} if reward is not None else {})},
             "exception_info": exc, "agent_info": {"name": "terminus-2"}, "agent_result": {"n_input_tokens": 10, "n_output_tokens": 5}}
        (job / name / "result.json").write_text(json.dumps(r))
    trial("a__1", reward=1.0)
    trial("b__2", reward=0.0)
    trial("c__3", rewards={"reward": 0.5})
    trial("d__4", exc={"exception_type": "RuntimeError", "exception_message": "boom"})
    trial("e__5", rewards={"accuracy": 1.0, "speed": 0.0})
    agg = run.aggregate(job)
    check("counts", agg["total"] == 5 and agg["passed"] == 3 and agg["errored"] == 1, str(agg))
    check("score averages with errored as 0", abs(agg["score"] - (1 + 0 + 0.5 + 0 + 0.5) / 5) < 1e-9, str(agg["score"]))
    check("error recorded", agg["tasks"][3]["error"] == "RuntimeError: boom")
    check("multi-metric mean", agg["tasks"][4]["score"] == 0.5)
    check("error rate", abs(agg["error_rate"] - 0.2) < 1e-9)

# bundle materialization + agent mapping
check("belvedir agent maps to import path", run.agent_arg("belvedir") == run.BELVEDIR_AGENT_IMPORT and run.agent_arg("terminus-2") == "terminus-2")
cmd = run.build_command({"dataset": "belvedir:x", "dataset_path": Path("dataset"), "env": "modal", "agent": "belvedir", "model": "openai/gpt-4.1-mini", "concurrency": 2, "jobs_dir": Path("jobs"), "n_tasks": None, "task_names": [], "attempts": 1})
check("bundle run uses -p and the agent import path", cmd[2:4] == ["-p", "dataset"] and "-d" not in cmd and run.BELVEDIR_AGENT_IMPORT in cmd, str(cmd))
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "ds"
    n = run.materialize_bundle({"files": {"README.md": "r", "task-001/task.toml": "t", "task-001/solution/solve.sh": "#!/bin/bash\n"},
                                "shared": {"tests/test.sh": "#!/bin/bash\n", "tests/verify.py": "print(1)\n"},
                                "taskDirs": ["task-001"], "executable": ["tests/test.sh", "solution/solve.sh"]}, out)
    check("bundle file count", n == 5, str(n))
    check("shared files land in the task dir", (out / "task-001/tests/verify.py").read_text() == "print(1)\n")
    check("scripts executable", os.access(out / "task-001/tests/test.sh", os.X_OK) and os.access(out / "task-001/solution/solve.sh", os.X_OK) and not os.access(out / "task-001/task.toml", os.X_OK))
    try:
        run.materialize_bundle({"files": {"../escape": "x"}}, out)
        check("path escape refused", False)
    except RuntimeError:
        check("path escape refused", True)
check("elapsed parsing", run._elapsed_sec("2026-09-02T20:05:58.100000+00:00", "2026-09-02T20:07:03.600000+00:00") == 65.5 and run._elapsed_sec(None, "x") is None)

if failures:
    sys.exit(f"{failures} failing")
print("all tests passed")
