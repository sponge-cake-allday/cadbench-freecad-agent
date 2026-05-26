import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .feedback import build_feedback_report, write_feedback_report
from .freecad_runner import run_freecad_script, run_geometry_inspection
from .models import CadTask
from .repair import decide_next_step
from .rubric import write_rubric


def _default_run_dir(task: CadTask) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"{task.task_id}-{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CADBench-style FreeCAD attempt locally.")
    parser.add_argument("--task", required=True, type=Path, help="Path to a task JSON file.")
    parser.add_argument("--script", required=True, type=Path, help="Path to generated FreeCAD Python.")
    parser.add_argument("--run-dir", type=Path, help="Directory for logs and artifacts.")
    parser.add_argument("--freecadcmd", help="Explicit path to FreeCADCmd/freecadcmd.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=1)
    args = parser.parse_args()

    task = CadTask.from_path(args.task)
    run_dir = args.run_dir or _default_run_dir(task)
    run_dir.mkdir(parents=True, exist_ok=True)

    task_copy = run_dir / "task.json"
    shutil.copyfile(args.task, task_copy)
    (run_dir / "prompt.txt").write_text(task.prompt_text(), encoding="utf-8")
    write_rubric(task, run_dir / "self_validation_rubric.json")

    output_name = task.expected_outputs[0] if task.expected_outputs else "candidate.FCStd"
    final_report = None
    final_decision = None
    attempts_completed = 0

    for attempt_index in range(1, args.max_attempts + 1):
        attempts_completed = attempt_index
        attempt_dir = run_dir / f"attempt-{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        script_copy = attempt_dir / "attempt.py"
        shutil.copyfile(args.script, script_copy)

        execution = run_freecad_script(
            script_path=script_copy,
            run_dir=attempt_dir,
            output_name=output_name,
            freecadcmd=args.freecadcmd,
            timeout_seconds=args.timeout,
        )
        if execution.ok and (attempt_dir / output_name).exists():
            run_geometry_inspection(
                run_dir=attempt_dir,
                artifact_name=output_name,
                freecadcmd=args.freecadcmd,
                timeout_seconds=args.timeout,
            )

        report = build_feedback_report(task, execution)
        decision = decide_next_step(report)
        write_feedback_report(report, attempt_dir / "feedback.json")
        (attempt_dir / "reflection.json").write_text(
            json.dumps(
                {
                    "action": decision.action,
                    "reason": decision.reason,
                    "repairable": decision.repairable,
                    "suggestions": decision.suggestions,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        final_report = report
        final_decision = decision

        print(f"attempt: {attempt_index}")
        print(f"attempt_dir: {attempt_dir}")
        print(f"status: {report.status}")
        for check in report.checks:
            marker = "PASS" if check["passed"] else "FAIL"
            print(f"{marker} {check['name']}: {check['detail']}")
        print(f"reflection: {decision.action} - {decision.reason}")

        if report.status == "ok" or decision.action == "stop":
            break

        if attempt_index < args.max_attempts:
            print("repair loop: no automatic patcher is wired yet; retrying the same script.")

    summary = {
        "task_id": task.task_id,
        "attempts_requested": args.max_attempts,
        "attempts_completed": attempts_completed,
        "final_status": final_report.status if final_report else "not_run",
        "final_action": final_decision.action if final_decision else "none",
        "final_reason": final_decision.reason if final_decision else "none",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"run_dir: {run_dir}")
    print(f"final_status: {summary['final_status']}")
    return 0 if final_report and final_report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
