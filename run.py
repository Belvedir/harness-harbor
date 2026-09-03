#!/usr/bin/env python3
"""Belvedir driver for Harbor-format benchmark suites.

Runs a dataset from the Harbor hub (hub.harborframework.com — Terminal-Bench,
SWE-bench Verified, τ³-bench, GAIA, Aider polyglot, …) through `harbor run`
and reshapes the result into the Belvedir results contract. Complete suites
bring their own tasks AND grading; this driver only picks the agent scaffold,
wires the model under test, runs the suite, and writes results.json.

Platform contract (Belvedir benchmarks docs):
  env  MODEL                 model under test (required)
       MODEL_API_BASE        OpenAI-compatible base URL; OpenRouter by default
       MODEL_API_KEY         key for that endpoint
       HARBOR_DATASET        hub dataset, `org/name` or `org/name@version`
                             (the catalog preset fills it) — OR
       BELVEDIR_HARBOR_BUNDLE_URL  (set by the platform for the Harbor
                             runner) a signed URL to an exported Belvedir
                             environment (lib/harbor/export.ts manifest:
                             files/shared/taskDirs/executable), materialized
                             to ./dataset and run with `harbor run -p`
       HARBOR_AGENT          Harbor agent scaffold (default terminus-2);
                             `belvedir` = the Belvedir external agent
                             (belvedir_harbor.agent:BelvedirAgent — one model
                             call per task, MODEL_* contract, traced)
       HARBOR_ENV            Harbor environment backend (default modal — the
                             Belvedir sandbox has no Docker daemon; `docker`
                             for local runs)
       HARBOR_N_TASKS        cap on tasks (default 10; blank/0 = the whole
                             suite — mind the 1h sandbox cap)
       HARBOR_TASK_NAMES     comma-separated task names/globs to include
       HARBOR_CONCURRENCY    concurrent trials (default 4)
       HARBOR_MODEL          raw Harbor/LiteLLM model id override (skips the
                             MODEL_* mapping — for providers we don't infer)
       BELVEDIR_TASK_ATTEMPTS attempts per task (Harbor -k)
       MODAL_TOKEN_ID/SECRET  when HARBOR_ENV=modal: the Modal account that
                             hosts the per-task containers (bring your own —
                             platform credentials never enter a sandbox)
  writes results.json at the repo root: numeric top-level `score` in 0..1

Harbor's built-in agents resolve their model through LiteLLM-style ids and
provider env vars (harbor/agents/model_connection.py). MODEL/MODEL_API_BASE/
MODEL_API_KEY are mapped onto that: OpenRouter base (or an sk-or- key) →
`openrouter/<vendor/model>` + OPENROUTER_API_KEY; any other explicit base →
`openai/<model>` + OPENAI_API_BASE/OPENAI_API_KEY (OpenAI-compatible);
no base → the vendor inferred from the id (claude-* → anthropic, gpt-*/o* →
openai) with that vendor's key env.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_AGENT = "terminus-2"
BELVEDIR_AGENT_IMPORT = "belvedir_harbor.agent:BelvedirAgent"
DATASET_DIR = Path("dataset")
DEFAULT_ENV = "modal"
DEFAULT_N_TASKS = 10
DEFAULT_CONCURRENCY = 4
JOBS_DIR = Path("jobs")
RESULTS_PATH = Path("results.json")
RESULTS_SOFT_CAP_BYTES = 200 * 1024
# Above this share of errored trials the score says more about the
# infrastructure than the model — same threshold as the Belvedir Runner.
MAX_ERROR_RATE = 0.3
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def log(msg: str) -> None:
    print(f"belvedir-harbor: {msg}", flush=True)


def fail(msg: str) -> None:
    log(msg)
    sys.exit(1)


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


# --- model mapping (pure) ------------------------------------------------------


def infer_vendor(model: str) -> str | None:
    if re.match(r"^(anthropic/)?claude-", model):
        return "anthropic"
    if re.match(r"^(openai/)?(gpt-|o\d)", model):
        return "openai"
    return None


# Harbor agents with no native OpenRouter provider (they raise "Unsupported
# provider 'openrouter'"). For these, OpenRouter is reached as the OpenAI-
# compatible `openai` provider with an explicit base URL instead.
NO_OPENROUTER_PROVIDER = frozenset({"openclaw"})


def model_args(
    model: str, base: str, key: str, raw_override: str = "", agent: str = ""
) -> tuple[str, dict[str, str]]:
    """(harbor model id, env vars to set) for the Belvedir MODEL_* contract."""
    if raw_override:
        # Power users: pass Harbor/LiteLLM ids straight through; the key still
        # has to reach the right provider env, so hand it to every plausible one.
        extra = {}
        if key:
            extra = {"OPENROUTER_API_KEY": key, "OPENAI_API_KEY": key, "ANTHROPIC_API_KEY": key}
            if base:
                extra["OPENAI_API_BASE"] = base
        return raw_override, extra
    if not model:
        raise ValueError("MODEL is not set")
    bare = model[len("openrouter/"):] if model.startswith("openrouter/") else model
    is_openrouter = "openrouter.ai" in base or (not base and key.startswith("sk-or-"))
    if is_openrouter:
        if "/" not in bare:
            vendor = infer_vendor(bare)
            if not vendor:
                raise ValueError(
                    f"OpenRouter ids are vendor/model; can't infer the vendor for {bare!r}"
                )
            bare = f"{vendor}/{bare}"
        if agent in NO_OPENROUTER_PROVIDER:
            # OpenRouter is OpenAI-compatible: the agent's `openai` provider
            # pointed at OpenRouter's base, model id kept as vendor/model.
            or_base = base or OPENROUTER_BASE
            return f"openai/{bare}", {
                "OPENAI_API_KEY": key,
                "OPENAI_BASE_URL": or_base,
                "OPENAI_API_BASE": or_base,
            }
        return f"openrouter/{bare}", {"OPENROUTER_API_KEY": key}
    if base:
        # Any OpenAI-compatible endpoint (Together, vLLM, a Belvedir router
        # key, …): LiteLLM's openai/ provider with an explicit base. Both
        # base-URL spellings, since agents differ (LiteLLM: OPENAI_API_BASE;
        # OpenClaw & co: OPENAI_BASE_URL).
        stripped = bare.split("/", 1)[1] if bare.startswith("openai/") else bare
        return f"openai/{stripped}", {"OPENAI_API_KEY": key, "OPENAI_API_BASE": base, "OPENAI_BASE_URL": base}
    vendor = infer_vendor(bare)
    if vendor == "anthropic":
        return f"anthropic/{bare.split('/', 1)[-1]}", {"ANTHROPIC_API_KEY": key}
    if vendor == "openai":
        return f"openai/{bare.split('/', 1)[-1]}", {"OPENAI_API_KEY": key}
    raise ValueError(
        f"MODEL_API_BASE is not set and can't be inferred from {model!r}; "
        "set it to an OpenAI-compatible base URL (or use OpenRouter)"
    )


# --- command (pure) ------------------------------------------------------------


def agent_arg(agent: str) -> str:
    """`belvedir` (and the import path itself) → the Belvedir external agent;
    anything else is one of Harbor's built-in agent names."""
    if agent in ("belvedir", "belvedir-agent", BELVEDIR_AGENT_IMPORT):
        return BELVEDIR_AGENT_IMPORT
    return agent


def build_command(cfg: dict) -> list[str]:
    cmd = [
        "harbor", "run",
        *(["-p", str(cfg["dataset_path"])] if cfg.get("dataset_path") else ["-d", cfg["dataset"]]),
        "-e", cfg["env"],
        "-a", agent_arg(cfg["agent"]),
        "-m", cfg["model"],
        "-n", str(cfg["concurrency"]),
        "-o", str(cfg["jobs_dir"]),
        "-y", "-q",
    ]
    if cfg.get("n_tasks"):
        cmd += ["-l", str(cfg["n_tasks"])]
    for name in cfg.get("task_names") or []:
        cmd += ["-i", name]
    if cfg.get("attempts", 1) > 1:
        cmd += ["-k", str(cfg["attempts"])]
    return cmd


# --- exported Belvedir environment (bundle) -------------------------------------


def materialize_bundle(manifest: dict, out: Path) -> int:
    """Write an exported dataset (the platform's HarborExport JSON: per-task
    `files`, once-only `shared` files written into every `taskDirs` entry,
    `executable` suffixes) into `out`. Returns the file count. Paths are
    confined to `out`."""
    out = out.resolve()
    execs = manifest.get("executable") or []

    def is_exec(rel: str) -> bool:
        return any(rel == e or rel.endswith("/" + e) for e in execs)

    def write(rel: str, content: str) -> None:
        target = (out / rel).resolve()
        if out not in target.parents:
            raise RuntimeError(f"bundle path escapes the dataset dir: {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        if is_exec(rel):
            target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    count = 0
    for rel, content in (manifest.get("files") or {}).items():
        write(rel, content)
        count += 1
    for task_dir in manifest.get("taskDirs") or []:
        for rel, content in (manifest.get("shared") or {}).items():
            write(f"{task_dir}/{rel}", content)
            count += 1
    return count


def fetch_bundle(url: str) -> dict:
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as res:
                return json.loads(res.read().decode())
        except Exception as e:  # noqa: BLE001 - retried, then surfaced
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"bundle download failed: {last}")


# --- results (pure over the jobs dir) ------------------------------------------


def aggregate(job_dir: Path) -> dict:
    """Fold Harbor's per-trial result.json files into the Belvedir shape.
    An errored trial scores 0 and is counted; the caller decides whether the
    error rate invalidates the score."""
    trials = []
    for trial_result in sorted(job_dir.glob("*/result.json")):
        try:
            r = json.loads(trial_result.read_text())
        except (OSError, ValueError):
            continue
        rewards = ((r.get("verifier_result") or {}).get("rewards") or {})
        exc = r.get("exception_info")
        reward = rewards.get("reward")
        if not isinstance(reward, (int, float)):
            # Multi-metric verifiers: take the mean of numeric metrics.
            nums = [v for v in rewards.values() if isinstance(v, (int, float))]
            reward = sum(nums) / len(nums) if nums else None
        elapsed = _elapsed_sec(r.get("started_at"), r.get("finished_at"))
        trials.append(
            {
                "task": r.get("task_name") or trial_result.parent.name,
                "elapsed_sec": elapsed,
                "trial": r.get("trial_name") or trial_result.parent.name,
                "score": float(reward) if reward is not None and not exc else 0.0,
                "pass": bool(reward is not None and not exc and reward >= 0.5),
                "error": (
                    f"{exc.get('exception_type', 'error')}: {str(exc.get('exception_message', ''))[:200]}"
                    if isinstance(exc, dict)
                    else None
                ),
                "agent": ((r.get("agent_info") or {}).get("name")),
                "tokens_in": ((r.get("agent_result") or {}).get("n_input_tokens")),
                "tokens_out": ((r.get("agent_result") or {}).get("n_output_tokens")),
            }
        )
    total = len(trials)
    errored = sum(1 for t in trials if t["error"])
    score = sum(t["score"] for t in trials) / total if total else 0.0
    return {
        "score": score,
        "total": total,
        "passed": sum(1 for t in trials if t["pass"]),
        "errored": errored,
        "error_rate": (errored / total) if total else 0.0,
        # Summed per-task container wall time — what the platform meters
        # when the containers ran on its managed Modal workspace.
        "container_sec": round(sum(t["elapsed_sec"] or 0.0 for t in trials), 1),
        "tasks": trials,
    }


def _elapsed_sec(started, finished):
    """Seconds between two ISO-8601 stamps (Harbor writes them with a UTC
    offset); None when either is missing or unparseable."""
    from datetime import datetime

    try:
        a = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    sec = (b - a).total_seconds()
    return round(sec, 1) if sec >= 0 else None


# --- main ----------------------------------------------------------------------


def harbor_bin() -> str | None:
    """The harbor CLI: on PATH, or next to this interpreter (a venv or a
    --user/--break-system-packages install whose bin dir isn't on PATH)."""
    found = shutil.which("harbor")
    if found:
        return found
    for candidate in (
        Path(sys.executable).parent / "harbor",
        Path.home() / ".local" / "bin" / "harbor",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def main() -> None:
    harbor = harbor_bin()
    if harbor is None:
        fail(
            "harbor is not installed (the install step should `pip install --break-system-packages -r requirements.txt`, "
            "or `python -m pip install -r requirements.txt` inside a venv)"
        )
    dataset = env("HARBOR_DATASET")
    bundle_url = env("BELVEDIR_HARBOR_BUNDLE_URL")
    dataset_path = None
    if bundle_url:
        # The Harbor runner: the platform exported the environment's task set
        # as a Harbor dataset; run that instead of a hub dataset.
        manifest = fetch_bundle(bundle_url)
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        n = materialize_bundle(manifest, DATASET_DIR)
        dataset = f"belvedir:{manifest.get('slug') or 'environment'}"
        dataset_path = DATASET_DIR
        log(f"materialized exported environment {dataset} ({n} files, {len(manifest.get('taskDirs') or [])} tasks)")
    elif not dataset:
        fail("HARBOR_DATASET is not set — this driver runs a Harbor hub dataset (org/name[@version]) or an exported Belvedir environment")
    elif env("BELVEDIR_TASKS_FILE") or env("FRACTAL_TASKS_FILE"):
        log(
            "note: a task file was provided but this driver runs a complete Harbor suite; "
            "export the environment with `belvedir environments export` and run it with `harbor run` instead"
        )
    harbor_env = env("HARBOR_ENV", DEFAULT_ENV)
    if harbor_env == "modal" and not (env("MODAL_TOKEN_ID") and env("MODAL_TOKEN_SECRET")):
        fail(
            "HARBOR_ENV=modal needs MODAL_TOKEN_ID and MODAL_TOKEN_SECRET (the Modal account that "
            "hosts the per-task containers); set them on the harness, or HARBOR_ENV=docker for a local run"
        )
    agent = env("HARBOR_AGENT", DEFAULT_AGENT)
    if agent_arg(agent) == BELVEDIR_AGENT_IMPORT:
        # The Belvedir agent reads MODEL/MODEL_API_BASE/MODEL_API_KEY itself.
        model_id, model_env = env("MODEL") or env("HARBOR_MODEL"), {}
        if not model_id:
            fail("MODEL is not set")
    else:
        try:
            model_id, model_env = model_args(
                env("MODEL"), env("MODEL_API_BASE"), env("MODEL_API_KEY"), env("HARBOR_MODEL"), agent
            )
        except ValueError as e:
            fail(str(e))
    n_tasks_raw = env("HARBOR_N_TASKS", str(DEFAULT_N_TASKS))
    try:
        n_tasks = int(n_tasks_raw) if n_tasks_raw else 0
    except ValueError:
        fail(f"HARBOR_N_TASKS must be an integer, got {n_tasks_raw!r}")
    cfg = {
        "dataset": dataset,
        "dataset_path": dataset_path,
        "env": harbor_env,
        "agent": agent,
        "model": model_id,
        "concurrency": int(env("HARBOR_CONCURRENCY", str(DEFAULT_CONCURRENCY)) or DEFAULT_CONCURRENCY),
        "n_tasks": n_tasks if n_tasks > 0 else None,
        "task_names": [s.strip() for s in env("HARBOR_TASK_NAMES").split(",") if s.strip()],
        "attempts": max(1, int(env("BELVEDIR_TASK_ATTEMPTS", "1") or 1)),
        "jobs_dir": JOBS_DIR,
    }
    cmd = build_command(cfg)
    cmd[0] = harbor
    scope = "all tasks" if not cfg["n_tasks"] else f"up to {cfg['n_tasks']} tasks"
    log(
        f"dataset {dataset} · agent {cfg['agent']} · model {model_id} · env {harbor_env} · "
        f"{scope} · {cfg['attempts']} attempt(s)"
    )
    # The vendored belvedir_harbor package (this repo's dir) must be importable
    # by Harbor's interpreter for `-a belvedir_harbor.agent:BelvedirAgent`.
    here = str(Path(__file__).resolve().parent)
    run_env = {**os.environ, **model_env}
    run_env["PYTHONPATH"] = here + (os.pathsep + run_env["PYTHONPATH"] if run_env.get("PYTHONPATH") else "")
    started = time.time()
    proc = subprocess.run(cmd, env=run_env)
    elapsed = time.time() - started
    if proc.returncode != 0:
        log(f"harbor run exited {proc.returncode} after {elapsed:.0f}s")
    jobs = sorted((p for p in JOBS_DIR.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not jobs:
        fail("harbor produced no job directory — see the log above")
    results = aggregate(jobs[-1])
    if results["total"] == 0:
        fail("harbor ran no trials (dataset empty after filters, or every task failed validation)")
    if results["error_rate"] > MAX_ERROR_RATE:
        fail(
            f"{results['errored']}/{results['total']} trials errored — the score would measure the "
            "infrastructure, not the model; no score reported"
        )
    out = {
        **results,
        "model": env("MODEL_LABEL") or model_id,
        "dataset": dataset,
        "agent": cfg["agent"],
        "harbor_env": harbor_env,
        "attempts_per_task": cfg["attempts"],
        "elapsed_sec": round(elapsed),
        "harbor_version": harbor_version(),
    }
    text = json.dumps(out, indent=2)
    if len(text.encode()) > RESULTS_SOFT_CAP_BYTES:
        out.pop("tasks", None)
        text = json.dumps(out, indent=2)
        log("per-task detail dropped from results.json (size cap)")
    RESULTS_PATH.write_text(text)
    log(
        f"done: score {out['score']:.3f} ({out['passed']}/{out['total']} passed, "
        f"{out['errored']} errored) in {elapsed:.0f}s"
    )


def harbor_version() -> str | None:
    try:
        return subprocess.run([harbor_bin() or "harbor", "--version"], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    main()
