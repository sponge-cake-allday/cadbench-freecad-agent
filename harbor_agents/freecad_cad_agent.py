from pathlib import Path
import asyncio
from datetime import datetime, timezone
import json
import os
import re

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from cad_harness.llm_client import (
    ChatCompletionError,
    cap_max_tokens_for_context,
    chat_completion_with_usage,
    extract_python_code,
    is_context_window_error,
    reduced_max_tokens_after_context_error,
)
from cad_harness.models import CadTask
from cad_harness.repair import build_repair_guidance
from cad_harness.spec_validator import validate_spec


SYSTEM_PROMPT = """You are a CADBench agent.
Write a single self-contained FreeCAD Python script to solve the task.

Rules:
- Use the task instruction as the source of truth.
- For CADBench FreeCAD tasks, write a parametric PartDesign model inside one Body.
- Prefer editable FreeCAD document objects/features over a single pre-computed Part::Feature.
- Run in headless FreeCADCmd: never import or use FreeCADGui, Gui, PartDesignGui, or other GUI modules.
- Do not use PartDesignGui. It fails in the benchmark image because GUI libraries are unavailable.
- If using PartDesign Pad/Pocket, create valid Sketcher sketches first. Do not assign to pad.Profile directly.
- Do not call nonexistent Sketcher constructors like Sketcher.Line, Sketcher.LineSegment, Sketcher.Geometry,
  Sketcher.GeometryData, or Sketcher.Circle. In FreeCAD 0.21, sketch geometry comes from Part, e.g.
  sketch.addGeometry(Part.LineSegment(App.Vector(...), App.Vector(...)), False).
- Do not use Part.Point, Part.GeomLine, or unconstrained construction points in Sketcher sketches.
- If adding Sketcher constraints, use only geometry indices returned by sketch.addGeometry; never guess constraint
  or geometry indices, and do not reference external geometry indices.
- For common blocks, cylinders, bores, slots, and shafts, prefer PartDesign primitive additive/subtractive features
  over fragile low-level Sketcher constraint graphs.
- Do not create document objects named PartDesign::SubtractivePad, PartDesign::SubtractivePocket,
  PartDesign::AdditivePad, PartDesign::AdditiveFeature, or PartDesign::AdditivePocket. Use
  PartDesign::SubtractiveBox/Cylinder/Cone/Sphere for primitive cuts, or a PartDesign::Pocket feature
  driven by a real sketch.
- Do not create document objects named PartDesign::AdditiveHexagon, PartDesign::AdditiveRegularPolygon,
  or PartDesign::AdditivePrism. These are not available in FreeCAD 0.21 PartDesign.
- Avoid PartDesign::LinearPattern and PartDesign::PolarPattern unless you know the exact FreeCAD 0.21 API.
  For repeated holes, rungs, slots, or bolts, create individual primitive features in a Python loop.
- Do not set unsupported properties such as Feature, Features, Source, Axis, Midplane, or mapMode on
  PartDesign primitive features. For sketch attachment use MapMode with the correct capitalization.
- Do not use `import Path`; use `from pathlib import Path`. Do not call the `Path` FreeCAD module.
- Do not reference `body.Origin.XY_Plane`, `doc.XYPlane`, or `App.Sketcher`. If Sketcher is required, import
  `Sketcher` directly and prefer unattached sketches with explicit Placement over fragile origin-plane references.
- For gears and toothed parts, prioritize a valid parametric approximation using cylinders, hubs, bores, and named
  dimensions over fragile per-tooth sketch/pocket code. Avoid additive pipe/sweep path features unless certain.
- For simple cylinder bodies and cylindrical holes, prefer this reliable headless PartDesign pattern:
  feature = doc.addObject("PartDesign::AdditiveCylinder" or "PartDesign::SubtractiveCylinder", name);
  set Radius, Height, Angle, optional Placement.Base; body.addObject(feature); doc.recompute().
- Use Python's math module for trigonometry. FreeCAD/App does not provide App.sin or App.cos.
- Use millimeters unless the instruction says otherwise.
- Convert key-parameter lines into valid Python assignments before modeling:
  `diameter = 20.0 mm` becomes `diameter = 20.0`; textual values like `drive_style = external_hex`
  become quoted strings such as `drive_style = "external_hex"`.
- Define every variable used in a formula from the key parameters first. Do not invent shorter aliases
  such as `outer_diameter` unless they are explicitly assigned before use.
- Do not access nonexistent shape properties such as feature.Solid. Use feature.Shape only after recompute,
  and use the generated FCStd artifact as the validation target.
- Never emit incomplete assignments. Every `=` in the returned code must have a valid right-hand side.
- Save the final CAD artifact exactly where the instruction requests, usually next to the script using
  Path(__file__).with_suffix(".FCStd").
- The verifier expects /app/answer.py to produce /app/answer.FCStd when run as FreeCADCmd /app/answer.py.
- Use CAD_HARNESS_OUTPUT_FCSTD as the save path if that environment variable is set.
- Prefer parametric solid geometry over mesh-only approximations.
- Return only complete Python code, no prose and no Markdown fences.
"""


GUI_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(?:FreeCADGui|Gui|PartDesignGui)\b.*$",
    flags=re.MULTILINE,
)


def sanitize_code(code: str) -> str:
    """Remove common GUI-only imports that break under FreeCADCmd."""
    cleaned = GUI_IMPORT_RE.sub("", code)
    cleaned = re.sub(r"^\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*import\s+Path\s*$", "from pathlib import Path", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("App.cos(", "math.cos(")
    cleaned = cleaned.replace("App.sin(", "math.sin(")
    cleaned = cleaned.replace("App.radians(", "math.radians(")
    cleaned = cleaned.replace("App.pi", "math.pi")
    cleaned = re.sub(r"\.Solid\b", ".Shape", cleaned)
    cleaned = cleaned.replace('"PartDesign::AdditivePad"', '"PartDesign::Pad"')
    cleaned = cleaned.replace("'PartDesign::AdditivePad'", "'PartDesign::Pad'")
    cleaned = cleaned.replace('"PartDesign::SubtractivePocket"', '"PartDesign::Pocket"')
    cleaned = cleaned.replace("'PartDesign::SubtractivePocket'", "'PartDesign::Pocket'")
    if (
        "FreeCAD." in cleaned
        and re.search(r"^\s*import\s+FreeCAD\s+as\s+App\b", cleaned, flags=re.MULTILINE)
        and not re.search(r"^\s*import\s+FreeCAD\b(?!\s+as\s+App)", cleaned, flags=re.MULTILINE)
    ):
        cleaned = cleaned.replace("FreeCAD.", "App.")
    if "math." in cleaned and not re.search(r"^\s*import\s+math\b", cleaned, flags=re.MULTILINE):
        cleaned = "import math\n" + cleaned
    if "Part." in cleaned and not re.search(r"^\s*import\s+Part\b", cleaned, flags=re.MULTILINE):
        cleaned = "import Part\n" + cleaned
    return cleaned.strip() + "\n"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def _failure_signature(text: str) -> str:
    exception = re.search(r"Exception while processing file:[^\[]*\[([^\]]+)\]", text)
    if exception:
        return exception.group(1).strip()
    syntax = re.search(r"invalid syntax[^\n]*", text, flags=re.IGNORECASE)
    if syntax:
        return syntax.group(0).strip()
    if "Recompute failed" in text:
        return "Recompute failed"
    scope = re.search(r"Link\(s\) to object\(s\).*?go out of the allowed scope[^\n]*", text)
    if scope:
        return scope.group(0).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else "No diagnostic output captured."


def _summarize_result(name: str, payload: dict, max_chars: int) -> str:
    stdout = payload.get("stdout") or ""
    stderr = payload.get("stderr") or ""
    combined = "\n".join(part for part in (stdout, stderr) if part)
    signature = _failure_signature(combined)
    spec_validation = payload.get("spec_validation") or {}
    failed_checks = [
        check
        for check in spec_validation.get("checks", [])
        if not check.get("passed") and check.get("severity") != "advisory"
    ][:5]
    spec_summary = ""
    if spec_validation:
        spec_summary = (
            f"local_spec_status={spec_validation.get('status')}, "
            f"local_spec_score={spec_validation.get('score')}, "
            f"failed_non_advisory_checks={failed_checks}\n"
        )
    tail = "\n".join(combined.splitlines()[-40:])
    return f"""{name}:
return_code={payload.get("return_code")}
signature={signature}
{spec_summary}\
output_tail:
{_truncate_middle(tail, max_chars)}
"""


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    for match in re.finditer(r"\{", text):
        try:
            return json.loads(text[match.start() :])
        except json.JSONDecodeError:
            continue
    return None


def _local_task_from_instruction(instruction: str) -> CadTask:
    return CadTask(
        task_id="harbor-task",
        name="CADBench FreeCAD task",
        description=instruction,
        key_parameters={},
        expected_outputs=["answer.FCStd"],
    )


def build_repair_prompt(
    instruction: str,
    attempt: int,
    execution: dict,
    artifact_check: dict,
    previous_code: str = "",
    max_feedback_chars: int = 5000,
    max_code_chars: int = 12000,
) -> str:
    previous = _truncate_middle(previous_code, max_code_chars) if previous_code else "(not captured)"
    guidance = build_repair_guidance(execution, artifact_check)
    failed_checks = guidance.get("failed_checks") or []
    failed_check_lines = "\n".join(
        f"- {item.get('name')} ({item.get('severity')}): {item.get('detail')}"
        for item in failed_checks
    ) or "- None captured."
    suggestion_lines = "\n".join(f"- {item}" for item in guidance.get("suggestions") or [])
    geometry_lines = "\n".join(
        f"- {item}" for item in guidance.get("geometry_intent_notes") or []
    ) or "- No additional geometry-intent notes captured."
    return f"""Attempt {attempt} failed local execution validation before benchmark scoring.

Task instruction:
{instruction}

Local repair diagnosis:
- failure_kind: {guidance.get("failure_kind")}
- failed_non_advisory_checks:
{failed_check_lines}

Targeted repair instructions:
{suggestion_lines}

Geometry-intent notes:
{geometry_lines}

Failure summary:
{_summarize_result("Execution", execution, max_feedback_chars)}
{_summarize_result("Artifact check", artifact_check, max_feedback_chars)}

Previous Python script:
```python
{previous}
```

Revise the FreeCAD Python script. Return only the full replacement Python code.
Focus on producing /app/answer.FCStd from /app/answer.py in headless FreeCADCmd.
Do not import GUI modules. Do not use the official benchmark score or reference file.
Patch the concrete failure above first; preserve correct dimensions and working features from the previous script.
If the previous script was truncated, returned with a Markdown fence, or ended mid-statement, return a complete script.
"""


class CadBenchFreeCadAgent(BaseAgent):
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = "model",
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_attempts: int = 3,
        context_window_tokens: int | str | None = None,
        token_safety_margin: int | str = 256,
        min_response_tokens: int | str = 128,
        compact_repair_context: bool | str = True,
        max_feedback_chars: int | str = 5000,
        max_repair_code_chars: int | str = 12000,
        reasoning_effort: str | None = None,
        reasoning_max_tokens: int | str | None = None,
        reasoning_exclude: bool | str = True,
        use_templates: bool = False,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = int(max_tokens)
        self.max_attempts = int(max_attempts)
        context_window_tokens = context_window_tokens or os.environ.get("CAD_AGENT_CONTEXT_WINDOW_TOKENS")
        self.context_window_tokens = int(context_window_tokens) if context_window_tokens else None
        self.token_safety_margin = int(token_safety_margin)
        self.min_response_tokens = int(min_response_tokens)
        self.compact_repair_context = _as_bool(compact_repair_context)
        self.max_feedback_chars = int(max_feedback_chars)
        self.max_repair_code_chars = int(max_repair_code_chars)
        self.reasoning_effort = reasoning_effort or os.environ.get("CAD_AGENT_REASONING_EFFORT")
        self.reasoning_max_tokens = (
            int(reasoning_max_tokens)
            if reasoning_max_tokens
            else (
                int(os.environ["CAD_AGENT_REASONING_MAX_TOKENS"])
                if os.environ.get("CAD_AGENT_REASONING_MAX_TOKENS")
                else None
            )
        )
        self.reasoning_exclude = _as_bool(reasoning_exclude)
        self.use_templates_requested = use_templates
        self.use_templates = False
        self.hourly_rate_usd = (
            float(os.environ["CAD_AGENT_HOURLY_RATE_USD"])
            if os.environ.get("CAD_AGENT_HOURLY_RATE_USD")
            else None
        )

    @staticmethod
    def name() -> str:
        return "cadbench-freecad-agent"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.logs_dir / "instruction.txt"
        script_path = self.logs_dir / "attempt.py"
        probe_path = self.logs_dir / "environment_probe.json"

        prompt_path.write_text(instruction, encoding="utf-8")

        probe = await environment.exec(
            "pwd; ls -la /app; command -v FreeCADCmd || true; command -v freecadcmd || true; command -v python3 || true",
            timeout_sec=30,
        )
        probe_path.write_text(
            json.dumps(
                {
                    "stdout": probe.stdout,
                    "stderr": probe.stderr,
                    "return_code": probe.return_code,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        command = (
            "set -eu\n"
            "rm -f /app/answer.FCStd\n"
            "if command -v FreeCADCmd >/dev/null 2>&1; then\n"
            "  FreeCADCmd /app/answer.py\n"
            "elif command -v freecadcmd >/dev/null 2>&1; then\n"
            "  freecadcmd /app/answer.py\n"
            "else\n"
            "  python3 /app/answer.py\n"
            "fi\n"
            "test -s /app/answer.FCStd\n"
        )
        validation_command = r"""set -eu
test -s /app/answer.FCStd
cat > /tmp/cad_agent_validate.py <<'PY'
import json
import sys
from pathlib import Path

import FreeCAD as App

path = Path("/app/answer.FCStd")
report_path = Path("/tmp/cad_agent_validate_report.json")
report = {"path": str(path), "errors": []}

def bbox_dict(shape):
    bbox = shape.BoundBox
    return {
        "x_length": float(bbox.XLength),
        "y_length": float(bbox.YLength),
        "z_length": float(bbox.ZLength),
    }

def jsonable_property(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "Value"):
        try:
            return float(value.Value)
        except Exception:
            pass
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        try:
            return {"x": float(value.x), "y": float(value.y), "z": float(value.z)}
        except Exception:
            pass
    return str(value)

def shape_stats(shape):
    surface_type_counts = {}
    cylinder_radii = []
    for face in list(getattr(shape, "Faces", [])):
        surface = getattr(face, "Surface", None)
        surface_type = getattr(surface, "TypeId", None) or type(surface).__name__
        surface_type_counts[surface_type] = surface_type_counts.get(surface_type, 0) + 1
        radius = getattr(surface, "Radius", None)
        if radius is not None:
            try:
                cylinder_radii.append(round(float(radius), 6))
            except Exception:
                pass
    return {
        "is_valid": bool(shape.isValid()),
        "area": float(getattr(shape, "Area", 0.0) or 0.0),
        "face_count": len(list(getattr(shape, "Faces", []))),
        "edge_count": len(list(getattr(shape, "Edges", []))),
        "vertex_count": len(list(getattr(shape, "Vertexes", []))),
        "surface_type_counts": surface_type_counts,
        "cylinder_radii": sorted(set(cylinder_radii)),
        "cylinder_diameters": sorted(set(round(radius * 2.0, 6) for radius in cylinder_radii)),
    }

def object_record(obj):
    property_names = list(getattr(obj, "PropertiesList", []))
    property_values = {}
    for prop in property_names:
        if prop in {
            "Radius",
            "Radius1",
            "Radius2",
            "Diameter",
            "Height",
            "Length",
            "Width",
            "Thickness",
            "Depth",
            "Angle",
        }:
            try:
                property_values[prop] = jsonable_property(getattr(obj, prop))
            except Exception:
                pass
    record = {
        "name": getattr(obj, "Name", ""),
        "label": getattr(obj, "Label", ""),
        "type_id": getattr(obj, "TypeId", ""),
        "properties": property_names,
        "property_values": property_values,
    }
    try:
        shape = getattr(obj, "Shape", None)
        if shape is not None and not shape.isNull():
            record["solid_count"] = len(list(shape.Solids))
            record["volume"] = float(shape.Volume)
            record["bbox"] = bbox_dict(shape)
            record.update(shape_stats(shape))
    except Exception:
        pass
    return record

try:
    if not path.is_file() or path.stat().st_size == 0:
        report["errors"].append("missing_or_empty_fcstd")
    else:
        doc = App.openDocument(str(path))
        doc.recompute()
        bodies = [obj for obj in doc.Objects if obj.TypeId == "PartDesign::Body"]
        report["body_count"] = len(bodies)
        report["object_count"] = len(doc.Objects)
        report["objects"] = [object_record(obj) for obj in doc.Objects]
        if len(bodies) != 1:
            report["errors"].append(f"expected_one_partdesign_body_got_{len(bodies)}")
        if bodies:
            body = bodies[0]
            tip = getattr(body, "Tip", None)
            shape_owner = tip or body
            shape = getattr(shape_owner, "Shape", None)
            if shape is None or shape.isNull():
                report["errors"].append("body_shape_is_null")
            else:
                solids = list(shape.Solids)
                report["solid_count"] = len(solids)
                report["volume"] = float(shape.Volume)
                report["bbox"] = bbox_dict(shape)
                report.update(shape_stats(shape))
                report["bodies"] = [
                    {
                        "name": getattr(body, "Name", ""),
                        "label": getattr(body, "Label", ""),
                        "type_id": getattr(body, "TypeId", ""),
                        "solid_count": len(solids),
                        "volume": float(shape.Volume),
                        "bbox": bbox_dict(shape),
                        **shape_stats(shape),
                    }
                ]
                if len(solids) != 1:
                    report["errors"].append(f"expected_one_solid_got_{len(solids)}")
                if shape.Volume <= 0:
                    report["errors"].append("non_positive_volume")
except Exception as exc:
    report["errors"].append(f"{type(exc).__name__}: {exc}")

report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
sys.exit(0 if not report["errors"] else 2)
PY
rc=0
if command -v FreeCADCmd >/dev/null 2>&1; then
  FreeCADCmd /tmp/cad_agent_validate.py || rc=$?
elif command -v freecadcmd >/dev/null 2>&1; then
  freecadcmd /tmp/cad_agent_validate.py || rc=$?
else
  python3 /tmp/cad_agent_validate.py || rc=$?
fi
cat /tmp/cad_agent_validate_report.json 2>/dev/null || true
exit "$rc"
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        local_task = _local_task_from_instruction(instruction)
        final_result = None
        final_artifact_check = None
        final_artifact_payload = None
        run_started_at = datetime.now(timezone.utc)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost_usd = 0.0
        has_cost_usage = False
        usage_observations = []
        extra_body = {}
        if self.reasoning_effort or self.reasoning_max_tokens:
            reasoning = {"exclude": self.reasoning_exclude}
            if self.reasoning_effort:
                reasoning["effort"] = self.reasoning_effort
            if self.reasoning_max_tokens:
                reasoning["max_tokens"] = self.reasoning_max_tokens
            extra_body["reasoning"] = reasoning

        def run_candidate(code: str, attempt: int, source: str):
            response_path = self.logs_dir / f"model_response-{attempt:02d}.txt"
            attempt_script_path = self.logs_dir / f"attempt-{attempt:02d}.py"
            exec_path = self.logs_dir / f"execution-{attempt:02d}.json"
            artifact_path = self.logs_dir / f"artifact-check-{attempt:02d}.json"

            response_path.write_text(code, encoding="utf-8")
            code = sanitize_code(code)
            attempt_script_path.write_text(code, encoding="utf-8")
            script_path.write_text(code, encoding="utf-8")

            return response_path, attempt_script_path, exec_path, artifact_path, code, source

        async def execute_attempt(attempt: int, source: str, exec_path: Path, artifact_path: Path):
            await environment.upload_file(script_path, "/app/answer.py")

            result = await environment.exec(command, timeout_sec=600)
            execution_payload = {
                "attempt": attempt,
                "source": source,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
            }
            exec_path.write_text(json.dumps(execution_payload, indent=2), encoding="utf-8")

            artifact_check = await environment.exec(
                validation_command,
                timeout_sec=120,
            )
            artifact_payload = {
                "attempt": attempt,
                "source": source,
                "stdout": artifact_check.stdout,
                "stderr": artifact_check.stderr,
                "return_code": artifact_check.return_code,
                "validator_return_code": artifact_check.return_code,
            }
            validation_report = _extract_json_object(
                "\n".join(part for part in (artifact_check.stdout, artifact_check.stderr) if part)
            )
            artifact_payload["local_geometry_metrics"] = validation_report
            spec_validation = validate_spec(local_task, validation_report)
            artifact_payload["spec_validation"] = spec_validation
            if artifact_payload["return_code"] == 0 and spec_validation["status"] != "ok":
                artifact_payload["return_code"] = 2
            artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
            return result, artifact_check, execution_payload, artifact_payload

        next_attempt = 1
        solved = False
        for _ in range(self.max_attempts):
            if solved:
                break
            attempt = next_attempt
            next_attempt += 1
            response_path = self.logs_dir / f"model_response-{attempt:02d}.txt"
            attempt_script_path = self.logs_dir / f"attempt-{attempt:02d}.py"
            exec_path = self.logs_dir / f"execution-{attempt:02d}.json"
            artifact_path = self.logs_dir / f"artifact-check-{attempt:02d}.json"

            request_max_tokens, estimated_prompt_tokens = cap_max_tokens_for_context(
                messages=messages,
                max_tokens=self.max_tokens,
                context_window_tokens=self.context_window_tokens,
                safety_margin=self.token_safety_margin,
            )
            completion = None
            for llm_try in range(3):
                try:
                    completion = chat_completion_with_usage(
                        base_url=self.base_url,
                        model=self.model_name or "model",
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=request_max_tokens,
                        api_key=self.api_key,
                        extra_body=extra_body,
                    )
                    break
                except ChatCompletionError as exc:
                    error_body = exc.body or str(exc)
                    if exc.prompt_tokens is not None:
                        total_prompt_tokens += int(exc.prompt_tokens)
                    if exc.completion_tokens is not None:
                        total_completion_tokens += int(exc.completion_tokens)
                    if exc.total_tokens is not None:
                        total_tokens += int(exc.total_tokens)
                    if exc.cost_usd is not None:
                        total_cost_usd += float(exc.cost_usd)
                        has_cost_usage = True
                    (self.logs_dir / f"llm-error-{attempt:02d}-{llm_try + 1:02d}.json").write_text(
                        json.dumps(
                            {
                                "attempt": attempt,
                                "llm_try": llm_try + 1,
                                "status": exc.status,
                                "message": str(exc),
                                "body": error_body[:4000],
                                "max_tokens": request_max_tokens,
                                "estimated_prompt_tokens": estimated_prompt_tokens,
                                "prompt_tokens": exc.prompt_tokens,
                                "completion_tokens": exc.completion_tokens,
                                "total_tokens": exc.total_tokens,
                                "cost_usd": exc.cost_usd,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    if is_context_window_error(exc):
                        reduced_max_tokens = reduced_max_tokens_after_context_error(
                            current_max_tokens=request_max_tokens,
                            error_text=error_body,
                            min_response_tokens=self.min_response_tokens,
                            safety_margin=self.token_safety_margin,
                        )
                        if reduced_max_tokens < request_max_tokens:
                            usage_observations.append(
                                {
                                    "attempt": attempt,
                                    "source": "llm_error",
                                    "error": "context_window",
                                    "max_tokens": request_max_tokens,
                                    "retry_max_tokens": reduced_max_tokens,
                                    "estimated_prompt_tokens": estimated_prompt_tokens,
                                }
                            )
                            request_max_tokens = reduced_max_tokens
                            continue
                    if exc.status is None or exc.status in {408, 409, 425, 429, 500, 502, 503, 504}:
                        usage_observations.append(
                                {
                                    "attempt": attempt,
                                    "source": "llm_error",
                                    "error": "transient_api_error",
                                    "status": exc.status,
                                    "llm_try": llm_try + 1,
                                    "max_tokens": request_max_tokens,
                                    "estimated_prompt_tokens": estimated_prompt_tokens,
                                    "prompt_tokens": exc.prompt_tokens,
                                    "completion_tokens": exc.completion_tokens,
                                    "total_tokens": exc.total_tokens,
                                    "cost_usd": exc.cost_usd,
                                    "usage": exc.usage,
                                }
                            )
                        if llm_try < 2:
                            await asyncio.sleep(min(2 ** llm_try, 8))
                            continue
                    raise
            if completion is None:
                raise RuntimeError("LLM completion failed without an exception.")
            response = completion.content
            usage_observations.append(
                {
                    "attempt": attempt,
                    "source": "llm",
                    "max_tokens": request_max_tokens,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "total_tokens": completion.total_tokens,
                    "cost_usd": completion.cost_usd,
                    "usage": completion.usage,
                }
            )
            if completion.prompt_tokens is not None:
                total_prompt_tokens += completion.prompt_tokens
            if completion.completion_tokens is not None:
                total_completion_tokens += completion.completion_tokens
            if completion.total_tokens is not None:
                total_tokens += completion.total_tokens
            if completion.cost_usd is not None:
                total_cost_usd += float(completion.cost_usd)
                has_cost_usage = True
            response_path.write_text(response, encoding="utf-8")
            code = sanitize_code(extract_python_code(response))
            attempt_script_path.write_text(code, encoding="utf-8")
            script_path.write_text(code, encoding="utf-8")

            result, artifact_check, execution_payload, artifact_payload = await execute_attempt(
                attempt=attempt,
                source="llm",
                exec_path=exec_path,
                artifact_path=artifact_path,
            )
            final_result = result
            final_artifact_check = artifact_check
            final_artifact_payload = artifact_payload
            if artifact_payload.get("return_code") == 0:
                solved = True
                break

            repair_guidance = build_repair_guidance(execution_payload, artifact_payload)
            (self.logs_dir / f"repair-guidance-{attempt:02d}.json").write_text(
                json.dumps(repair_guidance, indent=2),
                encoding="utf-8",
            )
            repair_message = {
                "role": "user",
                "content": build_repair_prompt(
                    instruction=instruction,
                    attempt=attempt,
                    execution=execution_payload,
                    artifact_check=artifact_payload,
                    previous_code=code,
                    max_feedback_chars=self.max_feedback_chars,
                    max_code_chars=self.max_repair_code_chars,
                ),
            }
            if self.compact_repair_context:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, repair_message]
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append(repair_message)

        (self.logs_dir / "execution.json").write_text(
            json.dumps(
                {
                    "stdout": getattr(final_result, "stdout", None),
                    "stderr": getattr(final_result, "stderr", None),
                    "return_code": getattr(final_result, "return_code", None),
                    "artifact_return_code": (
                        final_artifact_payload.get("return_code")
                        if final_artifact_payload
                        else getattr(final_artifact_check, "return_code", None)
                    ),
                    "validator_return_code": getattr(final_artifact_check, "return_code", None),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        elapsed_seconds = (datetime.now(timezone.utc) - run_started_at).total_seconds()
        has_token_usage = any(
            usage.get("prompt_tokens") is not None or usage.get("completion_tokens") is not None
            for usage in usage_observations
        )
        if has_token_usage:
            context.n_input_tokens = total_prompt_tokens
            context.n_output_tokens = total_completion_tokens
        if has_cost_usage:
            context.cost_usd = total_cost_usd
        elif self.hourly_rate_usd is not None:
            context.cost_usd = (elapsed_seconds / 3600.0) * self.hourly_rate_usd

        context.metadata = {
            "base_url": self.base_url,
            "model": self.model_name,
            "api_key_configured": bool(self.api_key),
            "agent_logs_dir": str(self.logs_dir),
            "execution_return_code": getattr(final_result, "return_code", None),
            "artifact_return_code": (
                final_artifact_payload.get("return_code")
                if final_artifact_payload
                else getattr(final_artifact_check, "return_code", None)
            ),
            "validator_return_code": getattr(final_artifact_check, "return_code", None),
            "local_spec_validation": (
                final_artifact_payload.get("spec_validation") if final_artifact_payload else None
            ),
            "max_attempts": self.max_attempts,
            "context_window_tokens": self.context_window_tokens,
            "token_safety_margin": self.token_safety_margin,
            "compact_repair_context": self.compact_repair_context,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_max_tokens": self.reasoning_max_tokens,
            "reasoning_exclude": self.reasoning_exclude,
            "use_templates": self.use_templates,
            "use_templates_requested": self.use_templates_requested,
            "template_policy": "disabled; benchmark-tuned deterministic templates removed",
            "llm_usage": usage_observations,
            "llm_total_tokens": total_tokens if usage_observations else None,
            "llm_total_cost_usd": total_cost_usd if has_cost_usage else None,
            "agent_wall_seconds": elapsed_seconds,
            "hourly_rate_usd": self.hourly_rate_usd,
            "cost_model": (
                "provider usage.cost"
                if has_cost_usage
                else (
                    "agent wall time * CAD_AGENT_HOURLY_RATE_USD"
                    if self.hourly_rate_usd is not None
                    else None
                )
            ),
        }
