#!/usr/bin/env python3
"""Belvedir verifier for Harbor tasks.

Runs as the task's `tests/test.sh` inside Harbor's verifier step. Grades the
agent's final answer (/app/answer.md) against tests/verify.json — the task's
verifier spec from the Belvedir environment — with an INDEPENDENT LLM judge,
and writes Harbor's reward file.

This is a line-for-line port of the grading half of the first-party runner
(Belvedir/harness-belvedir run.js): same judge prompts, same degenerate-
attempt guard, same rubric partial credit, same judge pin. Keep them in sync.

Verifier kinds: `judge` and `rubric` (LLM-graded, the runner's grading),
plus the deterministic `code` (the task's own `verify(answer, state)`) and
`state_diff` (golden files under /app/state) — Harbor-first: the Belvedir
runner still refuses these two, so today they run only under Harbor.

Standard library only: it must run in any python:3-slim image with nothing
installed. Env (resolved by Harbor from the host via [verifier.env]):
  JUDGE_MODEL      judge model id; defaults to a PINNED model, never the
                   model under test
  JUDGE_API_BASE   OpenAI-compatible base URL (default OpenRouter)
  JUDGE_API_KEY    key for that endpoint (falls back to MODEL_API_KEY, then
                   OPENROUTER_API_KEY)
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_JUDGE_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_JUDGE_BASE = "https://openrouter.ai/api/v1"
ANSWER_PATH = os.environ.get("BELVEDIR_ANSWER_PATH", "/app/answer.md")
VERIFY_PATH = os.environ.get("BELVEDIR_VERIFY_PATH", "/tests/verify.json")
REWARD_DIR = os.environ.get("BELVEDIR_REWARD_DIR", "/logs/verifier")
# Where a task's mutable state lives (state_diff verifier, code verifier's
# `state.files`): the exporter's Dockerfile COPYs setup.state here.
STATE_DIR = os.environ.get("BELVEDIR_STATE_DIR", "/app/state")
STATE_MAX_FILES = 200
STATE_MAX_BYTES = 2 * 1024 * 1024
JUDGE_TIMEOUT_SEC = 120
HTTP_RETRIES = 3
JUDGE_CLIP_CHARS = 8000
# Whole-verifier wall budget, under the [verifier] timeout_sec (300) the
# exporter pins: Harbor kills the verifier at the ceiling mid-retry, and then
# neither reward.json nor the explanatory verdict.json is written. Every
# request's timeout is clipped to what remains, so the ladder always ends
# inside the budget with a recorded error.
VERIFIER_BUDGET_SEC = 270

JUDGE_SYSTEM = (
    "You are a strict but fair grader of an AI agent's attempt at a task. "
    "Respond with ONLY a JSON object, no prose, no code fences."
)


def log(msg):
    print("belvedir-verify: " + msg, file=sys.stderr)


def clip(text, n):
    s = "" if text is None else str(text)
    return s[:n] + "\n[truncated]" if len(s) > n else s


def normalized(s):
    return re.sub(r"\s+", " ", "" if s is None else str(s)).strip().lower()


def degenerate_verdict(task, attempt):
    """Empty answers and task echoes fail without spending judge tokens."""
    if not normalized(attempt):
        return {"score": 0.0, "pass": False, "reason": "empty attempt"}
    if normalized(attempt) == normalized(task):
        return {"score": 0.0, "pass": False, "reason": "attempt echoes the task"}
    return None


def reference_match_verdict(expect, attempt):
    """An attempt identical to the reference passes without a judge call: the
    reference is known-good by definition. This is what `harbor run -a oracle`
    exercises — and judges reliably FAIL a verbatim reference as "copied, not
    composed" on originality-flavored tasks (observed Sept 2), which would
    make the oracle check useless for exactly the verifiers it should
    validate. Live attempts never hit this path unless they reproduce the
    reference exactly, which is a pass anyway."""
    if expect and normalized(attempt) == normalized(expect):
        return {"score": 1.0, "pass": True, "reason": "attempt matches the reference"}
    return None


def judge_prompt(task, expect, attempt):
    return "\n".join(
        [
            "Task the agent was given:",
            "<task>\n" + clip(task, JUDGE_CLIP_CHARS) + "\n</task>",
            "",
            "Reference outcome. This is one real answer known to have satisfied the",
            "task. Treat it as EVIDENCE of what success looks like — what facts,",
            "outcomes, and constraints matter — NOT as the only acceptable answer.",
            "<reference>\n" + clip(expect, JUDGE_CLIP_CHARS) + "\n</reference>",
            "",
            "Attempt to grade:",
            "<attempt>\n" + clip(attempt, JUDGE_CLIP_CHARS) + "\n</attempt>",
            "",
            "Does the attempt ACCOMPLISH THE TASK?",
            "- A different approach, format, wording, or level of detail than the",
            "  reference still PASSES. Never grade similarity to the reference.",
            "- FAIL an attempt that is empty, evasive, refuses, only restates or",
            "  plans the task, or contradicts load-bearing facts the reference",
            "  establishes.",
            'Respond with ONLY: {"pass": true or false, "reason": "<one short sentence>"}',
        ]
    )


def rubric_prompt(task, criteria, expect, attempt):
    numbered = "\n".join(
        "%d. %s" % (i + 1, clip(c, 500)) for i, c in enumerate(criteria)
    )
    lines = [
        "Task the agent was given:",
        "<task>\n" + clip(task, JUDGE_CLIP_CHARS) + "\n</task>",
        "",
        "Criteria. Grade EACH one independently against the attempt:",
        "<criteria>\n" + numbered + "\n</criteria>",
    ]
    if expect:
        lines += [
            "",
            "Reference outcome, as EVIDENCE of what success looks like (not the",
            "only acceptable answer):",
            "<reference>\n" + clip(expect, JUDGE_CLIP_CHARS) + "\n</reference>",
        ]
    lines += [
        "",
        "Attempt to grade:",
        "<attempt>\n" + clip(attempt, JUDGE_CLIP_CHARS) + "\n</attempt>",
        "",
        "For each criterion, decide whether the attempt satisfies it. A different",
        "approach, format, or wording still satisfies a criterion; grade outcomes,",
        "not style. An empty, evasive, or task-restating attempt satisfies nothing.",
        'Respond with ONLY: {"met": [%s], "reason": "<one short sentence on the first unmet criterion, or \'all met\'>"}'
        % ", ".join(["true or false"] * len(criteria)),
        'The "met" array must have exactly %d booleans, in criteria order.' % len(criteria),
    ]
    return "\n".join(lines)


def _extract_json(text):
    m = re.search(r"\{[\s\S]*\}", "" if text is None else str(text))
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def parse_verdict(text):
    obj = _extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("pass"), bool):
        return None
    return {
        "score": 1.0 if obj["pass"] else 0.0,
        "pass": obj["pass"],
        "reason": str(obj.get("reason", ""))[:200],
    }


def parse_rubric_verdict(text, criteria_count):
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        return None
    met = obj.get("met")
    if (
        not isinstance(met, list)
        or len(met) != criteria_count
        or any(not isinstance(m, bool) for m in met)
    ):
        return None
    n_met = sum(1 for m in met if m)
    score = n_met / float(criteria_count)
    return {
        "score": score,
        "pass": score >= 0.5,
        "reason": str(obj.get("reason", ""))[:200],
        "met": n_met,
    }


# --- deterministic verifiers (no judge) ---------------------------------------


def read_state(root=None):
    """{relative path: text} for the files under the state dir, bounded."""
    root = root or STATE_DIR
    files = {}
    total = 0
    if not os.path.isdir(root):
        return files
    for dirpath, _dirs, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if len(files) >= STATE_MAX_FILES or total + size > STATE_MAX_BYTES:
                return files
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    files[rel] = f.read()
            except OSError:
                continue
            total += size
    return files


def coerce_score(value):
    """A code verifier's return → score in 0..1, or None when unusable."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, dict):
        for key in ("score", "reward", "pass"):
            if key in value:
                return coerce_score(value[key])
    return None


def code_verdict(fn_src, attempt, state):
    """Run the task's own `verify(answer, state)` (spec: verifier kind
    `code`). The function is the environment author's code and runs inside
    the task container, so it gets plain exec — nothing to sandbox against
    here that the container isn't already. Errors surface as verifier
    errors (no reward), never as zeros."""
    namespace = {}
    exec(compile(fn_src, "<verify.fn>", "exec"), namespace)  # noqa: S102
    fn = namespace.get("verify")
    if not callable(fn):
        raise RuntimeError("code verifier defines no callable `verify(answer, state)`")
    score = coerce_score(fn(attempt, state))
    if score is None:
        raise RuntimeError("code verifier returned neither a bool, a number, nor {score}")
    return {"score": score, "pass": score >= 0.5, "reason": "code verifier"}


def state_diff_verdict(golden, files):
    """Fraction of golden files whose content the task left in place
    (whitespace-normalized). Extra files are ignored; a missing one counts
    against."""
    if not isinstance(golden, dict) or not golden:
        raise RuntimeError("state_diff verifier needs a non-empty `golden` {path: content} map")
    matched = 0
    first_miss = None
    for path, expected in golden.items():
        actual = files.get(path)
        if actual is not None and normalized(actual) == normalized(expected):
            matched += 1
        elif first_miss is None:
            first_miss = path if actual is not None else f"{path} (missing)"
    score = matched / float(len(golden))
    return {
        "score": score,
        "pass": score >= 0.5,
        "reason": "all golden files match" if first_miss is None else f"first mismatch: {first_miss}",
        "met": matched,
    }


def judge_endpoint():
    env = os.environ.get
    model = (env("JUDGE_MODEL") or "").strip() or DEFAULT_JUDGE_MODEL
    base = (
        (env("JUDGE_API_BASE") or "").strip()
        or (env("MODEL_API_BASE") or "").strip()
        or DEFAULT_JUDGE_BASE
    )
    key = (
        (env("JUDGE_API_KEY") or "").strip()
        or (env("MODEL_API_KEY") or "").strip()
        or (env("OPENROUTER_API_KEY") or "").strip()
    )
    return {"model": model, "base": base.rstrip("/"), "key": key}


def remaining(deadline):
    return deadline - time.monotonic()


def chat(endpoint, messages, deadline, timeout_sec=JUDGE_TIMEOUT_SEC, max_tokens=512):
    url = endpoint["base"] + "/chat/completions"
    body = json.dumps(
        {
            "model": endpoint["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    last_err = None
    for attempt in range(HTTP_RETRIES):
        left = remaining(deadline)
        if left < 5:
            raise last_err or RuntimeError("verifier time budget exhausted before the judge answered")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + endpoint["key"],
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=min(timeout_sec, left)) as res:
                data = json.loads(res.read().decode())
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            if not isinstance(content, str):
                raise RuntimeError("no message content in response")
            return content
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                last_err = RuntimeError("HTTP %d" % e.code)
            else:
                detail = e.read().decode(errors="replace")[:300]
                raise RuntimeError("HTTP %d: %s" % (e.code, detail))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            last_err = e
        if attempt < HTTP_RETRIES - 1:
            time.sleep(min(2 * (attempt + 1), max(0, remaining(deadline) - 5)))
    raise last_err or RuntimeError("request failed")


def ask_judge(endpoint, messages, parse, deadline):
    # One re-ask on an unparseable verdict before counting a judge error —
    # only when the budget still has room for a second call.
    verdict = parse(chat(endpoint, messages, deadline))
    if verdict:
        return verdict
    if remaining(deadline) > 10:
        verdict = parse(chat(endpoint, messages, deadline))
        if verdict:
            return verdict
    raise RuntimeError("judge returned no parseable verdict")


def verify(spec, attempt, endpoint, deadline=None):
    if deadline is None:
        deadline = time.monotonic() + VERIFIER_BUDGET_SEC
    task = spec["task"]
    v = spec["verify"]
    kind = v.get("kind")
    # The degenerate guard is about LLM-graded answers; deterministic
    # verifiers judge the state/answer themselves (an empty answer can still
    # leave a correct file tree).
    degenerate = None if kind in ("code", "state_diff") else degenerate_verdict(task, attempt)
    if degenerate:
        return degenerate
    if kind == "code":
        state = {"workdir": os.path.dirname(ANSWER_PATH), "state_dir": STATE_DIR, "files": read_state()}
        return code_verdict(v["fn"], attempt, state)
    if kind == "state_diff":
        return state_diff_verdict(v.get("golden"), read_state())
    matched = reference_match_verdict(v.get("expect"), attempt)
    if matched:
        return matched
    if kind == "rubric":
        criteria = v["criteria"]
        return ask_judge(
            endpoint,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": rubric_prompt(task, criteria, v.get("expect"), attempt)},
            ],
            lambda text: parse_rubric_verdict(text, len(criteria)),
            deadline,
        )
    if kind == "judge":
        return ask_judge(
            endpoint,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_prompt(task, v["expect"], attempt)},
            ],
            parse_verdict,
            deadline,
        )
    raise RuntimeError("unsupported verifier kind: %r" % kind)


def write_reward(verdict):
    os.makedirs(REWARD_DIR, exist_ok=True)
    with open(os.path.join(REWARD_DIR, "reward.json"), "w") as f:
        json.dump({"reward": float(verdict["score"])}, f)
    with open(os.path.join(REWARD_DIR, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)


def main():
    with open(VERIFY_PATH) as f:
        spec = json.load(f)
    try:
        with open(ANSWER_PATH, encoding="utf-8", errors="replace") as f:
            attempt = f.read()
    except OSError:
        attempt = ""
        log("no answer at %s — grading an empty attempt" % ANSWER_PATH)
    endpoint = judge_endpoint()
    needs_judge = spec.get("verify", {}).get("kind") in ("judge", "rubric")
    degenerate = degenerate_verdict(spec["task"], attempt) if needs_judge else None
    if degenerate is None and needs_judge and not endpoint["key"]:
        log("no judge key: set JUDGE_API_KEY (or MODEL_API_KEY / OPENROUTER_API_KEY)")
        sys.exit(2)
    deadline = time.monotonic() + VERIFIER_BUDGET_SEC
    try:
        verdict = verify(spec, attempt, endpoint, deadline)
    except Exception as e:  # a judge failure is NOT a zero — leave no reward
        os.makedirs(REWARD_DIR, exist_ok=True)
        with open(os.path.join(REWARD_DIR, "verdict.json"), "w") as f:
            json.dump({"error": "judge error: " + str(e)[:300], "judge_model": endpoint["model"]}, f)
        log("judge error: %s" % e)
        sys.exit(1)
    verdict["judge_model"] = endpoint["model"]
    write_reward(verdict)
    log(
        "%s (score %.3f%s)"
        % ("pass" if verdict["pass"] else "fail", verdict["score"], ": " + verdict["reason"] if verdict.get("reason") else "")
    )


if __name__ == "__main__":
    main()
