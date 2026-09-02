"""Belvedir external agent for Harbor.

Harbor's external agents run on the host and act on the task container
through the environment handle. This one is the Harbor counterpart of the
first-party runner's minimal harness (Belvedir/harness-belvedir harness.js):
task -> ONE chat call to the model under test -> the output is written to
/app/answer.md, which the task's verifier grades. Deliberately nearly
nothing: a Belvedir environment measures the MODEL on a training set's
tasks; scaffold cleverness here would be a confound nobody chose.

Bring your own agent per task, either way the runner supports:
  BELVEDIR_HARNESS_CMD  a shell command run on the host once per task with the
                        instruction on stdin and in $TASK; stdout is the answer.
                        (No TASK_INDEX here, unlike the runner — Harbor gives
                        the agent only the instruction.)
  BELVEDIR_AGENT_URL    an HTTP agent: POST {"task": ..., "run_id": ...} as
                        JSON, optional bearer BELVEDIR_AGENT_TOKEN; the reply
                        is the answer — a JSON object's "answer"/"output"/
                        "content"/"text" string, a JSON string, or the raw
                        body. This is how a production agent that is a
                        service gets scored on its own environment.

Tracing: when BELVEDIR_API_KEY is set (the Belvedir sandbox always sets it,
scoped to the run) and the `belvedir` SDK is importable, the SDK is
initialized once per process and every model call made here — via httpx, which
the SDK instruments — lands in the project's traces linked to the run, like
any other benchmark run. Each task runs in its own SDK session.

Run:
  harbor run -p <exported dataset dir> \
    -a belvedir_harbor.agent:BelvedirAgent -m openai/gpt-4.1-mini -y

Model endpoint (the Belvedir harness contract): MODEL_API_BASE (OpenAI-
compatible base URL) + MODEL_API_KEY. Unset base = OpenRouter when
OPENROUTER_API_KEY is set or the key is OpenRouter-shaped (any "vendor/model"
id; a bare id gets its vendor inferred), else inferred from the id
(claude-* -> Anthropic with ANTHROPIC_API_KEY, gpt-*/o* -> OpenAI with
OPENAI_API_KEY).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from . import __version__

try:  # the SDK instruments httpx; fall back to urllib without it
    import httpx  # type: ignore
except Exception:  # pragma: no cover - environment-dependent
    httpx = None  # type: ignore

ANSWER_PATH = "/app/answer.md"
SYSTEM_PROMPT = (
    "You are an AI agent completing a task for a user. Produce the actual "
    "final result of the task — not a plan, not a description of what you "
    "would do. Be direct and complete."
)
MODEL_TIMEOUT_SEC = 300
HTTP_RETRIES = 3
MAX_TOKENS = 4096
COMMAND_TIMEOUT_SEC = 300
COMMAND_OUTPUT_CAP = 1024 * 1024
# Whole-run wall budget, under the [agent] timeout_sec (600) the exporter
# pins, so the retry ladder ends with our error rather than Harbor's kill.
AGENT_BUDGET_SEC = 570
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def infer_base(model: str) -> str | None:
    if re.match(r"^(anthropic/)?claude-", model):
        return "https://api.anthropic.com/v1"
    if re.match(r"^(openai/)?(gpt-|o\d)", model):
        return "https://api.openai.com/v1"
    return None


def resolve_endpoint(model: str | None, get_env) -> dict:
    """Which endpoint serves the model under test. Explicit MODEL_API_BASE
    wins; then OpenRouter (the one-key, every-model default the platform's
    templates use); then the vendor inferred from the id."""
    model = (model or get_env("MODEL") or "").strip()
    if not model:
        raise RuntimeError("no model: pass -m <model> or set MODEL")
    base = (get_env("MODEL_API_BASE") or "").strip()
    key = (get_env("MODEL_API_KEY") or "").strip()
    if base:
        # Harbor-style "openrouter/vendor/model" ids drop the routing prefix
        # when the base is explicit; everything else is passed through.
        if model.startswith("openrouter/"):
            model = model[len("openrouter/"):]
        return {"model": model, "base": base.rstrip("/"), "key": key}
    openrouter_key = (get_env("OPENROUTER_API_KEY") or "").strip()
    # An OpenRouter-shaped MODEL_API_KEY with no base means OpenRouter too —
    # inferring the vendor from the model id would send the key to OpenAI
    # (the first live run did exactly that and got a 401).
    if openrouter_key or key.startswith("sk-or-"):
        if model.startswith("openrouter/"):
            model = model[len("openrouter/"):]
        # OpenRouter ids are vendor/model; a bare id gets the vendor the
        # runner would infer (gpt-* → openai, claude-* → anthropic).
        if "/" not in model:
            inferred = infer_base(model)
            if inferred and "anthropic" in inferred:
                model = "anthropic/" + model
            elif inferred:
                model = "openai/" + model
        return {"model": model, "base": OPENROUTER_BASE, "key": key or openrouter_key}
    inferred = infer_base(model)
    if not inferred:
        raise RuntimeError(
            f"MODEL_API_BASE is not set and can't be inferred from {model!r}; "
            "set it to an OpenAI-compatible base URL (or set OPENROUTER_API_KEY)"
        )
    vendor_key = (
        get_env("ANTHROPIC_API_KEY") if "anthropic" in inferred else get_env("OPENAI_API_KEY")
    ) or ""
    bare = model.split("/", 1)[1] if "/" in model else model
    return {"model": bare, "base": inferred, "key": key or vendor_key.strip()}


# --- HTTP (blocking; always called via asyncio.to_thread) ----------------------


def _post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, str]:
    """POST and return (status, text). httpx when available so the Belvedir
    SDK's HTTP instrumentation sees the call; urllib otherwise."""
    if httpx is not None:
        res = httpx.post(url, content=body, headers=headers, timeout=timeout)
        return res.status_code, res.text
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def _retrying_post(url: str, headers: dict[str, str], body: bytes, deadline: float, per_call_timeout: float) -> str:
    last_err: Exception | None = None
    for attempt in range(HTTP_RETRIES):
        left = deadline - time.monotonic()
        if left < 5:
            raise last_err or RuntimeError("agent time budget exhausted before the endpoint answered")
        try:
            status, text = _post(url, headers, body, min(per_call_timeout, left))
            if status == 429 or status >= 500:
                last_err = RuntimeError(f"HTTP {status}")
            elif status >= 400:
                raise RuntimeError(f"HTTP {status}: {text[:300]}")
            else:
                return text
        except RuntimeError:
            raise
        except Exception as e:  # timeouts, connection errors, bad TLS
            last_err = e
        if attempt < HTTP_RETRIES - 1:
            time.sleep(min(2 * (attempt + 1), max(0.0, deadline - time.monotonic() - 5)))
    raise last_err or RuntimeError("request failed")


def chat(endpoint: dict, messages: list[dict], deadline: float) -> tuple[str, dict]:
    """One chat completion against the model under test. Blocking."""
    body = json.dumps(
        {"model": endpoint["model"], "messages": messages, "max_tokens": MAX_TOKENS}
    ).encode()
    text = _retrying_post(
        endpoint["base"] + "/chat/completions",
        {"Content-Type": "application/json", "Authorization": "Bearer " + endpoint["key"]},
        body,
        deadline,
        MODEL_TIMEOUT_SEC,
    )
    data = json.loads(text)
    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError("no message content in response")
    return content, data.get("usage") or {}


def parse_agent_reply(text: str) -> str:
    """An HTTP agent's reply: a JSON object's answer-like string field, a JSON
    string, or the raw body."""
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith('"'):
        try:
            obj = json.loads(stripped)
        except ValueError:
            return text
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for key in ("answer", "output", "content", "text", "result", "response"):
                if isinstance(obj.get(key), str):
                    return obj[key]
            # OpenAI-shaped replies count too.
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                msg = (choices[0] or {}).get("message") or {}
                if isinstance(msg.get("content"), str):
                    return msg["content"]
            raise RuntimeError("agent reply is JSON without an answer-like string field")
    return text


def http_agent(url: str, token: str, instruction: str, run_id: str, deadline: float) -> str:
    body = json.dumps({"task": instruction, "instruction": instruction, "run_id": run_id or None}).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/plain"}
    if token:
        headers["Authorization"] = "Bearer " + token
    return parse_agent_reply(_retrying_post(url, headers, body, deadline, MODEL_TIMEOUT_SEC))


def command_harness(cmd: str, task: str, deadline: float) -> str:
    """Blocking (subprocess); call via asyncio.to_thread."""
    env = {**os.environ, "TASK": task}
    proc = subprocess.run(
        ["bash", "-c", cmd],
        input=task,
        capture_output=True,
        text=True,
        env=env,
        timeout=max(1.0, min(COMMAND_TIMEOUT_SEC, deadline - time.monotonic())),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"harness command exited {proc.returncode}: {proc.stderr[-300:]}")
    return proc.stdout[:COMMAND_OUTPUT_CAP]


# --- tracing -------------------------------------------------------------------

_tracing: dict = {"initialized": False, "sdk": None}


def init_tracing(get_env) -> object | None:
    """Initialize the Belvedir SDK once per process when a run-scoped key is
    present. Returns the SDK module (for sessions) or None."""
    if _tracing["initialized"]:
        return _tracing["sdk"]
    _tracing["initialized"] = True
    api_key = (get_env("BELVEDIR_API_KEY") or get_env("FRACTAL_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        import belvedir_loop as sdk  # the `belvedir` distribution
    except Exception:
        return None
    base_url = (get_env("BELVEDIR_BASE_URL") or get_env("FRACTAL_BASE_URL") or "").strip()
    kwargs = {"api_key": api_key, "app_name": "belvedir-harbor"}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        sdk.initialize(**kwargs)
    except Exception:
        return None
    _tracing["sdk"] = sdk
    return sdk


class BelvedirAgent(BaseAgent):
    """task -> one model call (or your command / HTTP agent) -> /app/answer.md."""

    @staticmethod
    def name() -> str:
        return "belvedir"

    def version(self) -> str | None:
        return __version__

    async def setup(self, environment: BaseEnvironment) -> None:
        # Nothing to install: the agent runs on the host. Only the answer's
        # directory has to exist in the container.
        await environment.exec(f"mkdir -p {Path(ANSWER_PATH).parent}")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        deadline = time.monotonic() + AGENT_BUDGET_SEC
        sdk = init_tracing(self._get_env)
        run_id = (self._get_env("BELVEDIR_RUN_ID") or self._get_env("FRACTAL_RUN_ID") or "").strip()
        session_id = f"harbor-{run_id or 'local'}-{self.logs_dir.parent.name}"
        span = sdk.session(session_id, metadata={"harbor_trial": self.logs_dir.parent.name}) if sdk else nullcontext()

        harness_cmd = (self._get_env("BELVEDIR_HARNESS_CMD") or "").strip()
        agent_url = (self._get_env("BELVEDIR_AGENT_URL") or "").strip()
        usage: dict = {}
        with span:
            if harness_cmd:
                answer = await asyncio.to_thread(command_harness, harness_cmd, instruction, deadline)
                model_label = "harness-command"
            elif agent_url:
                token = (self._get_env("BELVEDIR_AGENT_TOKEN") or "").strip()
                answer = await asyncio.to_thread(http_agent, agent_url, token, instruction, run_id, deadline)
                model_label = "http-agent"
            else:
                endpoint = resolve_endpoint(self.model_name, self._get_env)
                if not endpoint["key"]:
                    raise RuntimeError("no model key: set MODEL_API_KEY (or the vendor's key)")
                answer, usage = await asyncio.to_thread(
                    chat,
                    endpoint,
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": instruction},
                    ],
                    deadline,
                )
                model_label = endpoint["model"]

        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(answer)
            tmp = f.name
        try:
            await environment.upload_file(tmp, ANSWER_PATH)
        finally:
            os.unlink(tmp)

        (self.logs_dir / "answer.md").write_text(answer, encoding="utf-8")
        context.n_input_tokens = usage.get("prompt_tokens")
        context.n_output_tokens = usage.get("completion_tokens")
        context.metadata = {
            "model": model_label,
            "answer_path": ANSWER_PATH,
            "traced": bool(sdk),
        }
