import json
from pathlib import Path
from typing import Any, Dict, List

from .models import CadTask, ExecutionResult, FeedbackReport
from .spec_validator import validate_spec


def _tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _artifact_status(run_dir: Path, expected_outputs: List[str]) -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    for name in expected_outputs:
        path = run_dir / name
        files[name] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "path": str(path),
        }
    return files


def _rubric_checks(run_dir: Path) -> List[Dict[str, Any]]:
    rubric_path = run_dir.parent / "self_validation_rubric.json"
    metrics_path = run_dir / "geometry_metrics.json"
    if not rubric_path.exists():
        return []

    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    metrics = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    checks: List[Dict[str, Any]] = []
    for item in rubric.get("checks", []):
        item_type = item.get("type")
        if item_type == "geometry":
            passed = bool(metrics and metrics.get("solid_count", 0) > 0)
            detail = (
                f"solid_count={metrics.get('solid_count', 0)}"
                if metrics
                else "geometry metrics unavailable"
            )
        elif item_type == "prompt_constraint":
            passed = bool(metrics)
            detail = (
                "requires geometry-specific inspector; metrics captured"
                if metrics
                else "requires geometry metrics before parameter validation"
            )
        elif item_type == "topology":
            passed = bool(metrics)
            detail = (
                "requires topology-specific inspector; metrics captured"
                if metrics
                else "requires geometry metrics before topology validation"
            )
        else:
            continue

        checks.append(
            {
                "name": f"rubric:{item['id']}",
                "passed": passed,
                "detail": detail,
                "severity": item.get("severity", "important"),
            }
        )
    return checks


def build_feedback_report(task: CadTask, execution: ExecutionResult) -> FeedbackReport:
    artifacts = _artifact_status(execution.run_dir, task.expected_outputs)
    metrics_path = execution.run_dir / "geometry_metrics.json"
    geometry_metrics = None
    if metrics_path.exists():
        geometry_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    spec_validation = validate_spec(task, geometry_metrics)
    checks: List[Dict[str, Any]] = []

    checks.append(
        {
            "name": "freecad_executed",
            "passed": execution.ok,
            "detail": execution.message or f"status={execution.status}, returncode={execution.returncode}",
        }
    )

    for name, info in artifacts.items():
        checks.append(
            {
                "name": f"artifact_exists:{name}",
                "passed": bool(info["exists"] and info["bytes"] > 0),
                "detail": f"{info['bytes']} bytes at {info['path']}",
            }
        )
    checks.extend(_rubric_checks(execution.run_dir))
    checks.extend(spec_validation["checks"])

    report_status = (
        "ok"
        if all(check["passed"] or check.get("severity") == "advisory" for check in checks)
        else "needs_repair"
    )
    return FeedbackReport(
        task_id=task.task_id,
        status=report_status,
        execution={
            "status": execution.status,
            "returncode": execution.returncode,
            "command": execution.command,
            "stdout_tail": _tail(execution.stdout_path),
            "stderr_tail": _tail(execution.stderr_path),
        },
        artifacts={
            "files": artifacts,
            "geometry_metrics": geometry_metrics,
            "spec_validation": spec_validation,
        },
        checks=checks,
    )


def write_feedback_report(report: FeedbackReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
