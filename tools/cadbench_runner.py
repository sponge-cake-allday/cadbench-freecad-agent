#!/usr/bin/env python3
"""Run and summarize CADBench Harbor jobs for this agent.

Development runs should use ``--phase local`` so the official benchmark verifier is
disabled. Benchmark runs should use ``--phase benchmark`` after the agent is frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AGENT_IMPORT_PATH = "harbor_agents.freecad_cad_agent:CadBenchFreeCadAgent"
BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    context_window_tokens: int
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    category: str = "oss"


MODELS: dict[str, ModelSpec] = {
    "qwen25_coder_32b": ModelSpec(
        key="qwen25_coder_32b",
        model_id="qwen/qwen-2.5-coder-32b-instruct",
        context_window_tokens=128000,
        category="oss",
    ),
    "qwen3_coder": ModelSpec(
        key="qwen3_coder",
        model_id="qwen/qwen3-coder",
        context_window_tokens=1048576,
        category="oss",
    ),
    "deepseek_v32": ModelSpec(
        key="deepseek_v32",
        model_id="deepseek/deepseek-v3.2",
        context_window_tokens=131072,
        category="oss",
    ),
    "llama33_70b": ModelSpec(
        key="llama33_70b",
        model_id="meta-llama/llama-3.3-70b-instruct",
        context_window_tokens=131072,
        category="oss",
    ),
    "codestral_2508": ModelSpec(
        key="codestral_2508",
        model_id="mistralai/codestral-2508",
        context_window_tokens=256000,
        category="oss",
    ),
    "kimi_k26": ModelSpec(
        key="kimi_k26",
        model_id="moonshotai/kimi-k2.6",
        context_window_tokens=262144,
        category="oss",
    ),
    "glm46": ModelSpec(
        key="glm46",
        model_id="z-ai/glm-4.6",
        context_window_tokens=202752,
        category="oss",
    ),
    "devstral_2512": ModelSpec(
        key="devstral_2512",
        model_id="mistralai/devstral-2512",
        context_window_tokens=262144,
        category="oss",
    ),
    "gemini31_flash_lite_preview": ModelSpec(
        key="gemini31_flash_lite_preview",
        model_id="google/gemini-3.1-flash-lite-preview",
        context_window_tokens=1048576,
        category="closed",
    ),
    "claude_haiku_45": ModelSpec(
        key="claude_haiku_45",
        model_id="anthropic/claude-haiku-4.5",
        context_window_tokens=200000,
        category="closed",
    ),
    "gpt55_high": ModelSpec(
        key="gpt55_high",
        model_id="openai/gpt-5.5",
        context_window_tokens=1050000,
        max_tokens=64000,
        reasoning_effort="high",
        category="closed",
    ),
}


def load_env_file(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def selected_models(spec: str) -> list[ModelSpec]:
    if spec == "all":
        return list(MODELS.values())
    if spec == "oss":
        return [model for model in MODELS.values() if model.category == "oss"]
    if spec == "closed":
        return [model for model in MODELS.values() if model.category == "closed"]
    models = []
    for key in spec.split(","):
        key = key.strip()
        if not key:
            continue
        if key not in MODELS:
            raise SystemExit(f"Unknown model key {key!r}. Choices: {', '.join(MODELS)}")
        models.append(MODELS[key])
    return models


def dataset_args() -> list[str]:
    local_dataset = ROOT / "harbor_datasets" / "cad-bench"
    if local_dataset.exists():
        return ["--path", str(local_dataset)]
    return ["--dataset", "gnucleus-ai/cad-bench@latest"]


def run_harbor_job(
    model: ModelSpec,
    phase: str,
    n_tasks: int,
    n_concurrent: int,
    max_attempts: int,
    harbor_max_retries: int,
    jobs_root: Path,
    env: dict[str, str],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_dir = jobs_root / f"openrouter-{phase}-{model.key}-{timestamp}"
    job_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "harbor",
        "run",
        *dataset_args(),
        "--agent-import-path",
        AGENT_IMPORT_PATH,
        "-m",
        model.model_id,
        "--ak",
        f"base_url={BASE_URL}",
        "--ak",
        f"max_attempts={max_attempts}",
        "--ak",
        f"max_tokens={model.max_tokens}",
        "--ak",
        f"context_window_tokens={model.context_window_tokens}",
        "--ak",
        "compact_repair_context=true",
        "--n-tasks",
        str(n_tasks),
        "--n-concurrent",
        str(n_concurrent),
        "--max-retries",
        str(harbor_max_retries),
        "--artifact",
        "/app/answer.FCStd",
        "--artifact",
        "/app/answer.py",
        "--jobs-dir",
        str(job_dir),
        "--yes",
    ]
    if phase == "local":
        cmd.append("--disable-verification")
    if model.reasoning_effort:
        cmd.extend(["--ak", f"reasoning_effort={model.reasoning_effort}"])
        cmd.extend(["--ak", "reasoning_exclude=true"])

    manifest = {
        "phase": phase,
        "model": model.__dict__,
        "n_tasks": n_tasks,
        "n_concurrent": n_concurrent,
        "max_attempts": max_attempts,
        "harbor_max_retries": harbor_max_retries,
        "command": cmd,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "clean_development_boundary": phase == "local",
        "official_verifier_used": phase != "local",
    }
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "run_command.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_path = job_dir / "runner.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["return_code"] = result.returncode
    (job_dir / "run_command.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    actual_job_dir = resolve_job_dir(job_dir)
    summarize_job(actual_job_dir)
    if result.returncode != 0:
        raise SystemExit(f"Harbor job failed for {model.key}; see {log_path}")
    return actual_job_dir


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def resolve_job_dir(job_dir: Path) -> Path:
    job_dir = job_dir.resolve()
    if list(job_dir.glob("freecad-*")):
        return job_dir
    children = [path for path in job_dir.iterdir() if path.is_dir()] if job_dir.exists() else []
    children_with_trials = [path for path in children if list(path.glob("freecad-*"))]
    if len(children_with_trials) == 1:
        return children_with_trials[0]
    return job_dir


def last_attempt_payload(agent_dir: Path, prefix: str) -> dict[str, Any] | None:
    paths = sorted(agent_dir.glob(f"{prefix}-[0-9][0-9].json"))
    if not paths:
        return None
    return read_json(paths[-1])


def attempt_payload(agent_dir: Path, prefix: str, attempt: int) -> dict[str, Any] | None:
    path = agent_dir / f"{prefix}-{attempt:02d}.json"
    return read_json(path) if path.exists() else None


def local_success(execution: dict[str, Any] | None, artifact: dict[str, Any] | None) -> bool:
    return bool(artifact and artifact.get("return_code") == 0)


def summarize_job(job_dir: Path) -> dict[str, Any]:
    job_dir = resolve_job_dir(job_dir)
    trial_dirs = sorted(path for path in job_dir.glob("freecad-*") if path.is_dir())
    scores: dict[str, list[float]] = {
        "score": [],
        "combined": [],
        "geometry_similarity": [],
        "cad_spec_consistency": [],
    }
    local_spec_scores: list[float] = []
    first_success = fixed_by_repair = final_success = final_fail = 0
    attempts: list[int] = []
    input_tokens = output_tokens = 0
    cost_usd = 0.0
    context_errors = 0
    trial_summaries = []

    for trial_dir in trial_dirs:
        agent_dir = trial_dir / "agent"
        execution_paths = sorted(agent_dir.glob("execution-[0-9][0-9].json"))
        attempts.append(len(execution_paths))
        first_exec = attempt_payload(agent_dir, "execution", 1)
        first_artifact = attempt_payload(agent_dir, "artifact-check", 1)
        final_exec = last_attempt_payload(agent_dir, "execution")
        final_artifact = last_attempt_payload(agent_dir, "artifact-check")
        first_ok = local_success(first_exec, first_artifact)
        final_ok = local_success(final_exec, final_artifact)
        first_success += int(first_ok)
        final_success += int(final_ok)
        final_fail += int(not final_ok)
        fixed_by_repair += int((not first_ok) and final_ok)
        context_errors += len(list(agent_dir.glob("llm-error-*.json")))

        if final_artifact:
            spec = final_artifact.get("spec_validation") or {}
            if isinstance(spec.get("score"), (int, float)):
                local_spec_scores.append(float(spec["score"]))

        result_path = trial_dir / "result.json"
        agent_result = {}
        if result_path.exists():
            result = read_json(result_path)
            agent_result = result.get("agent_result") or {}
            input_tokens += int(agent_result.get("n_input_tokens") or 0)
            output_tokens += int(agent_result.get("n_output_tokens") or 0)
            cost_usd += float(agent_result.get("cost_usd") or 0.0)

        reward_path = trial_dir / "verifier" / "reward.json"
        reward = {}
        if reward_path.exists():
            reward = read_json(reward_path)
            for key in scores:
                value = reward.get(key)
                if isinstance(value, (int, float)):
                    scores[key].append(float(value))

        trial_summaries.append(
            {
                "trial": trial_dir.name,
                "attempts": len(execution_paths),
                "first_local_success": first_ok,
                "final_local_success": final_ok,
                "final_local_spec_score": (
                    (final_artifact or {}).get("spec_validation", {}).get("score")
                    if final_artifact
                    else None
                ),
                "official_score": reward.get("score"),
                "geometry_similarity": reward.get("geometry_similarity"),
                "cad_spec_consistency": reward.get("cad_spec_consistency"),
                "agent_cost_usd": agent_result.get("cost_usd"),
                "agent_input_tokens": agent_result.get("n_input_tokens"),
                "agent_output_tokens": agent_result.get("n_output_tokens"),
            }
        )

    summary = {
        "job_dir": str(job_dir),
        "n_trials": len(trial_dirs),
        "local": {
            "first_attempt_success": first_success,
            "first_attempt_failed": len(trial_dirs) - first_success,
            "fixed_by_repair": fixed_by_repair,
            "final_success": final_success,
            "final_failed": final_fail,
            "mean_attempts": safe_mean(attempts),
            "mean_final_local_spec_score": safe_mean(local_spec_scores),
            "llm_error_file_count": context_errors,
        },
        "official": {
            key: {"n": len(value), "mean": safe_mean(value)}
            for key, value in scores.items()
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        },
        "trials": trial_summaries,
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(job_dir, summary)
    return summary


def write_markdown_summary(job_dir: Path, summary: dict[str, Any]) -> None:
    official = summary["official"]
    local = summary["local"]
    usage = summary["usage"]
    lines = [
        f"# CADBench Run Summary",
        "",
        f"- Job: `{job_dir}`",
        f"- Trials: {summary['n_trials']}",
        f"- Local first-attempt success: {local['first_attempt_success']}/{summary['n_trials']}",
        f"- Local fixed by repair: {local['fixed_by_repair']}",
        f"- Local final success: {local['final_success']}/{summary['n_trials']}",
        f"- Mean local spec score: {local['mean_final_local_spec_score']}",
        f"- Official score mean: {official['score']['mean']} (n={official['score']['n']})",
        f"- Official geom mean: {official['geometry_similarity']['mean']} (n={official['geometry_similarity']['n']})",
        f"- Official spec mean: {official['cad_spec_consistency']['mean']} (n={official['cad_spec_consistency']['n']})",
        f"- Input tokens: {usage['input_tokens']}",
        f"- Output tokens: {usage['output_tokens']}",
        f"- Cost USD: {usage['cost_usd']}",
        "",
    ]
    (job_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--phase", choices=["local", "benchmark"], required=True)
    run_parser.add_argument("--models", default="qwen3_coder")
    run_parser.add_argument("--n-tasks", type=int, default=100)
    run_parser.add_argument("--n-concurrent", type=int, default=2)
    run_parser.add_argument("--max-attempts", type=int, default=3)
    run_parser.add_argument("--harbor-max-retries", type=int, default=1)
    run_parser.add_argument("--jobs-root", type=Path, default=ROOT / "jobs")
    run_parser.add_argument("--env-file", type=Path, default=ROOT / ".env.openrouter")

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("job_dirs", nargs="+", type=Path)

    list_parser = subparsers.add_parser("list-models")

    args = parser.parse_args()
    if args.command == "list-models":
        print(json.dumps({key: value.__dict__ for key, value in MODELS.items()}, indent=2))
        return 0
    if args.command == "summarize":
        for job_dir in args.job_dirs:
            summary = summarize_job(job_dir)
            print(json.dumps({k: summary[k] for k in ("job_dir", "n_trials", "local", "official", "usage")}, indent=2))
        return 0

    env = load_env_file(args.env_file)
    if not env.get("OPENROUTER_API_KEY"):
        raise SystemExit(f"OPENROUTER_API_KEY missing; expected it in {args.env_file}")
    args.jobs_root.mkdir(parents=True, exist_ok=True)
    run_dirs = []
    for model in selected_models(args.models):
        run_dirs.append(
            run_harbor_job(
                model=model,
                phase=args.phase,
                n_tasks=args.n_tasks,
                n_concurrent=args.n_concurrent,
                max_attempts=args.max_attempts,
                harbor_max_retries=args.harbor_max_retries,
                jobs_root=args.jobs_root,
                env=env,
            )
        )
    print(json.dumps({"run_dirs": [str(path) for path in run_dirs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
