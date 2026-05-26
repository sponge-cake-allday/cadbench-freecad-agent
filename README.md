# CADBench-Style Local Harness

This project is a small scaffold for testing CAD generation attempts without using the benchmark evaluator inside the repair loop.

## License

This project is source-available under the PolyForm Noncommercial License 1.0.0. Noncommercial use is permitted; commercial use requires separate written permission from the copyright holder. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Public benchmark maintainers may also run and cite this agent under the narrow additional permission in [BENCHMARKING_PERMISSION.md](BENCHMARKING_PERMISSION.md).

This repository contains the agent harness code only. It does not include CADBench datasets, benchmark reference artifacts, private run logs, or generated submission artifacts.

## Provenance Note

This revision is a clean baseline that removes the deterministic CAD family templates developed after inspecting CADBench scorer results. The agent does not use benchmark scores, reference `.FCStd` files, or hidden verifier diagnostics during a trial; it uses only generated code, local FreeCAD execution, and artifact validation.

The intended flow is:

```text
task JSON
  -> self-validation rubric
  -> generated FreeCAD Python
  -> local FreeCAD execution
  -> local feedback report
  -> repair or final artifact
```

The feedback is deliberately limited to local execution signals such as FreeCAD logs, expected artifact existence, and geometry metrics measured from the generated artifact itself. Official CADBench/Harbor scoring should run once at the end when measuring `pass@1`.

The harness has a retry cap through `--max-attempts`. It stops early for non-repairable blockers, such as a missing FreeCAD executable.

## Run a Local Candidate

```bash
.venv/bin/python -m cad_harness.cli \
  --task path/to/task.json \
  --script path/to/answer.py \
  --max-attempts 3
```

If FreeCAD is not installed, the harness writes a clear `missing_freecad` feedback report instead of crashing.

## Generate a Candidate With a Model

When an OpenAI-compatible model server is available, such as vLLM, pass the model id exposed by that server:

```bash
.venv/bin/python -m cad_harness.generate_candidate \
  --task path/to/task.json \
  --output runs/generated/attempt.py \
  --base-url http://localhost:8000/v1 \
  --model model
```

Then run the generated candidate through the local harness:

```bash
.venv/bin/python -m cad_harness.cli \
  --task path/to/task.json \
  --script runs/generated/attempt.py \
  --run-dir runs/generated \
  --max-attempts 3
```

Each attempt gets its own directory under the run directory with:

- `attempt.py`: the candidate script that was tested.
- `feedback.json`: local execution and artifact feedback.
- `reflection.json`: a deterministic decision about whether to stop or repair.

Each run also gets `self_validation_rubric.json`. Right now this rubric is generated deterministically from the task description and key parameters. The next agentic step is to optionally call an LLM before attempt 1 to produce a richer rubric, still using only the prompt and key parameters.

The current repair loop classifies failures and records repair suggestions. It does not yet call an LLM patcher to rewrite `attempt.py`; that is the next integration point.

## Files

- `harbor_agents/freecad_cad_agent.py`: Harbor custom agent that calls an OpenAI-compatible model, executes candidates, and reflects on execution failures.
- `harbor_agents/qwen_cad_agent.py`: backward-compatible import shim for earlier model-specific runs.
- `cad_harness/cli.py`: command-line entrypoint.
- `cad_harness/freecad_runner.py`: detects and runs `freecadcmd`/`FreeCADCmd`.
- `cad_harness/feedback.py`: writes local, non-benchmark feedback, including geometry metrics when FreeCAD inspection succeeds.
- `cad_harness/generate_candidate.py`: calls an OpenAI-compatible model and writes a FreeCAD candidate script.
- `cad_harness/llm_client.py`: minimal OpenAI-compatible HTTP client.
## Clean pass@1 boundary

Allowed inside the agent run:

- FreeCAD execution logs.
- Artifact existence checks.
- Prompt-derived validation.
- Geometry inspection implemented from the generated artifact itself.

Not allowed inside the repair loop:

- CADBench official score.
- Ground-truth `.FCStd` comparison.
- Hidden evaluator diagnostics.

## Local Prompt-Grounded Scorer

The harness includes a first-party local scorer in `cad_harness/spec_validator.py`. It is not a clone of CADBench scoring and does not compare against reference CAD. It uses only:

- the task prompt and key parameters
- the generated `.FCStd`
- FreeCAD-derived document structure and shape metrics

The scorer reports:

- hard validity: opens as FCStd, one `PartDesign::Body`, one non-empty solid
- parametric structure: editable PartDesign features, no mesh-only output, no single opaque `Part::Feature`
- prompt-grounded hints: bounding-box dimensions near explicit outer/overall parameters, cut-feature hints for holes/bores, repeated-feature hints for counted patterns

The output is written into each `feedback.json` under `artifacts.spec_validation` and mirrored into the top-level `checks` list with `spec:*` check names. Critical and important failures mark the run as `needs_repair`; prompt/spec heuristics start as advisory to avoid overfitting a particular benchmark.

## Harbor Agent

The generic custom Harbor agent import path is:

```text
harbor_agents.freecad_cad_agent:CadBenchFreeCadAgent
```

After an OpenAI-compatible model endpoint is reachable at `http://localhost:8000/v1`, a small CADBench run can use:

```bash
harbor run \
  -d gnucleus-ai/cad-bench@latest \
  --agent-import-path harbor_agents.freecad_cad_agent:CadBenchFreeCadAgent \
  -m model \
  --ak base_url=http://localhost:8000/v1 \
  --ak max_attempts=3 \
  --n-tasks 1 \
  --n-concurrent 1 \
  --jobs-dir jobs \
  --yes
```

For OpenRouter or another hosted OpenAI-compatible endpoint, load the local secret file first and point the agent at that API:

```bash
set -a
source .env.openrouter
set +a

harbor run \
  -d gnucleus-ai/cad-bench@latest \
  --agent-import-path harbor_agents.freecad_cad_agent:CadBenchFreeCadAgent \
  -m provider/model-id \
  --ak base_url=https://openrouter.ai/api/v1 \
  --ak max_attempts=3 \
  --ak max_tokens=1536 \
  --ak context_window_tokens=32768 \
  --n-tasks 1 \
  --n-concurrent 1 \
  --jobs-dir jobs/openrouter-smoke \
  --yes
```

`context_window_tokens` is optional but recommended when a provider/model has a known context size. The agent uses it to cap per-request output tokens before sending a request, and it also retries with a smaller output budget if an OpenAI-compatible endpoint returns a context-window error.

For development runs where you want only the agent's FreeCAD execution and self-validation feedback, disable the official benchmark verifier:

```bash
harbor run \
  -d gnucleus-ai/cad-bench@latest \
  --agent-import-path harbor_agents.freecad_cad_agent:CadBenchFreeCadAgent \
  -m model \
  --ak base_url=http://localhost:8000/v1 \
  --ak max_attempts=3 \
  --disable-verification \
  --n-tasks 3 \
  --n-concurrent 1 \
  --jobs-dir jobs/local-validation \
  --yes
```

For OpenAI-compatible backends that return a standard `usage` object, such as vLLM, the agent reports `n_input_tokens` and `n_output_tokens` to Harbor. Self-hosted serving cost is not inferable from the API response, so `cost_usd` is only populated when `CAD_AGENT_HOURLY_RATE_USD` is set. In that case, cost is reported as agent wall time multiplied by the hourly rate:

```bash
CAD_AGENT_HOURLY_RATE_USD=1.99 harbor run ...
```

For the downloaded local dataset:

```bash
harbor run \
  --path harbor_datasets/cad-bench \
  --agent-import-path harbor_agents.freecad_cad_agent:CadBenchFreeCadAgent \
  -m model \
  --ak base_url=http://127.0.0.1:8000/v1 \
  --ak max_attempts=3 \
  --n-tasks 5 \
  --n-concurrent 1 \
  --jobs-dir jobs \
  --yes
```

For older runs that used the model-specific class name, the legacy import path still works:

```text
harbor_agents.qwen_cad_agent:QwenCadAgent
```

The agent is intentionally model-first in this clean baseline. It may repair generated scripts using local FreeCAD execution errors and artifact checks, but it does not use benchmark scorer feedback or benchmark-specific deterministic templates.
