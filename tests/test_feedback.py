import tempfile
import unittest
from pathlib import Path

from cad_harness.feedback import build_feedback_report
from cad_harness.models import CadTask, ExecutionResult


class FeedbackReportTests(unittest.TestCase):
    def test_missing_artifact_needs_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            stdout = run_dir / "stdout.log"
            stderr = run_dir / "stderr.log"
            stdout.write_text("", encoding="utf-8")
            stderr.write_text("FreeCAD missing", encoding="utf-8")
            task = CadTask(
                task_id="task-1",
                name="example",
                description="example",
                key_parameters={},
                expected_outputs=["candidate.FCStd"],
            )
            execution = ExecutionResult(
                status="missing_freecad",
                command=[],
                returncode=None,
                stdout_path=stdout,
                stderr_path=stderr,
                run_dir=run_dir,
            )

            report = build_feedback_report(task, execution)

            self.assertEqual(report.status, "needs_repair")
            self.assertFalse(report.artifacts["files"]["candidate.FCStd"]["exists"])


if __name__ == "__main__":
    unittest.main()
