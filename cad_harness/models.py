from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CadTask:
    task_id: str
    name: str
    description: str
    key_parameters: Dict[str, Any]
    expected_outputs: List[str] = field(default_factory=lambda: ["candidate.FCStd"])
    notes: Optional[str] = None

    @classmethod
    def from_path(cls, path: Path) -> "CadTask":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            task_id=payload["task_id"],
            name=payload["name"],
            description=payload["description"],
            key_parameters=payload.get("key_parameters", {}),
            expected_outputs=payload.get("expected_outputs", ["candidate.FCStd"]),
            notes=payload.get("notes"),
        )

    def prompt_text(self) -> str:
        params = "\n".join(f"- {key}: {value}" for key, value in self.key_parameters.items())
        return (
            f"Task: {self.name}\n\n"
            f"Description:\n{self.description}\n\n"
            f"Key parameters:\n{params}\n\n"
            "Generate a FreeCAD model that satisfies the description and parameters."
        )


@dataclass
class ExecutionResult:
    status: str
    command: List[str]
    returncode: Optional[int]
    stdout_path: Path
    stderr_path: Path
    run_dir: Path
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.returncode == 0


@dataclass
class FeedbackReport:
    task_id: str
    status: str
    execution: Dict[str, Any]
    artifacts: Dict[str, Any]
    checks: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "execution": self.execution,
            "artifacts": self.artifacts,
            "checks": self.checks,
        }

