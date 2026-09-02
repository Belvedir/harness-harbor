# harness-harbor

Belvedir's driver for **Harbor-format benchmark suites**: Terminal-Bench,
SWE-bench Verified, τ³-bench, GAIA, Aider polyglot and any other dataset on
[hub.harborframework.com](https://hub.harborframework.com/datasets). The suite
brings its own tasks and grading; this driver picks the agent scaffold, wires
the model under test, runs `harbor run`, and writes Belvedir's `results.json`.

It is the catalog driver behind the complete-suite entries on Belvedir's Public
Benchmarks page. It also runs anywhere with Docker (`HARBOR_ENV=docker`).

## Contract

| Env | Meaning |
|---|---|
| `MODEL`, `MODEL_API_BASE`, `MODEL_API_KEY` | the model under test (Belvedir's standard harness contract; OpenRouter base by default) |
| `HARBOR_DATASET` | hub dataset `org/name[@version]` — the catalog preset fills it. Alternatively the platform sets `BELVEDIR_HARBOR_BUNDLE_URL`, an exported Belvedir environment the driver materializes and runs with `harbor run -p` |
| `HARBOR_AGENT` | Harbor agent scaffold, default `terminus-2` (`claude-code`, `codex`, `openhands`, … also work); `belvedir` = the Belvedir external agent (one model call per task via the `MODEL_*` contract, traced into the project) |
| `HARBOR_ENV` | Harbor backend, default `modal`; `docker` for local runs |
| `HARBOR_N_TASKS` | task cap, default 10 (blank/0 = whole suite; the Belvedir sandbox caps a run at 1h) |
| `HARBOR_TASK_NAMES` | comma-separated task names or globs to include |
| `HARBOR_CONCURRENCY` | concurrent trials, default 4 |
| `HARBOR_MODEL` | raw Harbor/LiteLLM model id, bypassing the `MODEL_*` mapping |
| `BELVEDIR_TASK_ATTEMPTS` | attempts per task |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | with `HARBOR_ENV=modal`: the Modal account hosting the per-task containers. Bring your own — Belvedir never puts platform credentials in a sandbox |

Output: `results.json` with `score` (mean reward over trials, errored trials
count as 0), `total`, `passed`, `errored`, `container_sec` (summed per-task
container wall time — what the platform meters for managed Modal), per-task
rewards and errors, the dataset/agent/model, and Harbor's version. When more than 30% of trials error
the driver exits scoreless rather than report a number that measures the
infrastructure.

## Model mapping

Harbor's agents resolve models through LiteLLM-style ids and provider env
vars. `MODEL_*` maps onto that: an OpenRouter base (or an `sk-or-` key) becomes
`openrouter/<vendor/model>` + `OPENROUTER_API_KEY`; any other explicit base
becomes `openai/<model>` + `OPENAI_API_BASE` (any OpenAI-compatible endpoint,
including a Belvedir router key); no base infers the vendor from the id.

## Local run

```bash
pip install -r requirements.txt            # in the Belvedir sandbox: pip install --break-system-packages -r requirements.txt (Debian, PEP 668)
export HARBOR_DATASET=harbor/hello-world HARBOR_ENV=docker HARBOR_AGENT=oracle MODEL=openai/gpt-4.1-mini MODEL_API_KEY=$OPENROUTER_API_KEY
python run.py && cat results.json
```

Tests: `python3 test/run-tests.py`.
