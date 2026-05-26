import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .models import CadTask


def _humanize_parameter(name: str) -> str:
    cleaned = name
    cleaned = re.sub(r"_mm$", "", cleaned)
    cleaned = cleaned.replace("_", " ")
    return cleaned


def generate_deterministic_rubric(task: CadTask) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = [
        {
            "id": "valid_freecad_execution",
            "type": "execution",
            "severity": "required",
            "description": "The generated FreeCAD script runs without error.",
            "source": "harness",
        },
        {
            "id": "expected_artifact_exists",
            "type": "artifact",
            "severity": "required",
            "description": "The expected .FCStd artifact is written to the run directory.",
            "source": "harness",
        },
        {
            "id": "valid_solid_geometry",
            "type": "geometry",
            "severity": "required",
            "description": "The generated artifact can be opened and contains at least one valid solid.",
            "source": "artifact",
        },
    ]

    for name, value in task.key_parameters.items():
        if isinstance(value, (int, float)):
            checks.append(
                {
                    "id": f"parameter_{name}",
                    "type": "prompt_constraint",
                    "severity": "important",
                    "description": f"The CAD model should reflect {_humanize_parameter(name)} = {value}.",
                    "parameter": name,
                    "expected": value,
                    "source": "task.key_parameters",
                }
            )

    lower_description = task.description.lower()
    if "through" in lower_description and ("bore" in lower_description or "hole" in lower_description):
        checks.append(
            {
                "id": "through_cuts",
                "type": "topology",
                "severity": "important",
                "description": "All described through bores or holes should fully pass through the part.",
                "source": "task.description",
            }
        )

    if "equally spaced" in lower_description or "evenly" in lower_description:
        checks.append(
            {
                "id": "even_pattern_spacing",
                "type": "topology",
                "severity": "important",
                "description": "Patterned holes/features should be evenly spaced as described.",
                "source": "task.description",
            }
        )

    return {
        "task_id": task.task_id,
        "rubric_source": "deterministic_fallback",
        "notes": (
            "This rubric is generated only from the task prompt and key parameters. "
            "It does not use benchmark evaluator output or ground-truth CAD."
        ),
        "checks": checks,
    }


def write_rubric(task: CadTask, path: Path) -> Dict[str, Any]:
    rubric = generate_deterministic_rubric(task)
    path.write_text(json.dumps(rubric, indent=2), encoding="utf-8")
    return rubric

