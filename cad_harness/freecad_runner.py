import os
from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, Optional

from .models import ExecutionResult


MAC_FREECADCMD_CANDIDATES = [
    "/Applications/FreeCAD.app/Contents/MacOS/FreeCADCmd",
    "/Applications/FreeCAD 1.0.app/Contents/MacOS/FreeCADCmd",
]


def find_freecadcmd(explicit_path: Optional[str] = None) -> Optional[str]:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists():
            return str(candidate)

    for name in ("freecadcmd", "FreeCADCmd"):
        found = shutil.which(name)
        if found:
            return found

    for candidate in MAC_FREECADCMD_CANDIDATES:
        if Path(candidate).exists():
            return candidate

    return None


def run_freecad_script(
    script_path: Path,
    run_dir: Path,
    output_name: str = "candidate.FCStd",
    freecadcmd: Optional[str] = None,
    timeout_seconds: int = 120,
) -> ExecutionResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "freecad_stdout.log"
    stderr_path = run_dir / "freecad_stderr.log"

    binary = find_freecadcmd(freecadcmd)
    if not binary:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("FreeCADCmd/freecadcmd was not found.\n", encoding="utf-8")
        return ExecutionResult(
            status="missing_freecad",
            command=[],
            returncode=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            run_dir=run_dir,
            message="Install FreeCAD or pass --freecadcmd to execute CAD scripts.",
        )

    env: Dict[str, str] = dict(os.environ)
    env["CAD_HARNESS_RUN_DIR"] = str(run_dir)
    env["CAD_HARNESS_OUTPUT_FCSTD"] = str(run_dir / output_name)

    command: List[str] = [binary, str(script_path)]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=str(run_dir),
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            stderr.write(f"Timed out after {timeout_seconds} seconds.\n")
            return ExecutionResult(
                status="timeout",
                command=command,
                returncode=None,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                run_dir=run_dir,
                message=f"FreeCAD execution timed out after {timeout_seconds} seconds.",
            )

    status = "ok" if completed.returncode == 0 else "error"
    return ExecutionResult(
        status=status,
        command=command,
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        run_dir=run_dir,
    )


def write_geometry_inspector(run_dir: Path, artifact_name: str = "candidate.FCStd") -> Path:
    inspector_path = run_dir / "inspect_geometry.py"
    inspector_path.write_text(
        f"""import json
from pathlib import Path

import FreeCAD as App


artifact_path = Path({str(run_dir / artifact_name)!r})
metrics_path = Path({str(run_dir / "geometry_metrics.json")!r})

def bbox_payload(shape):
    bbox = shape.BoundBox
    return {{
        "x_length": float(bbox.XLength),
        "y_length": float(bbox.YLength),
        "z_length": float(bbox.ZLength),
        "x_min": float(bbox.XMin),
        "x_max": float(bbox.XMax),
        "y_min": float(bbox.YMin),
        "y_max": float(bbox.YMax),
        "z_min": float(bbox.ZMin),
        "z_max": float(bbox.ZMax),
    }}


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
            return {{"x": float(value.x), "y": float(value.y), "z": float(value.z)}}
        except Exception:
            pass
    return str(value)


def shape_stats(shape):
    surface_type_counts = {{}}
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
    return {{
        "face_count": len(list(getattr(shape, "Faces", []))),
        "edge_count": len(list(getattr(shape, "Edges", []))),
        "vertex_count": len(list(getattr(shape, "Vertexes", []))),
        "surface_type_counts": surface_type_counts,
        "cylinder_radii": sorted(set(cylinder_radii)),
        "cylinder_diameters": sorted(set(round(radius * 2.0, 6) for radius in cylinder_radii)),
    }}


def shape_payload(shape):
    if shape is None or shape.isNull():
        return {{
            "has_shape": False,
            "is_valid": False,
            "volume": 0.0,
            "area": 0.0,
            "solid_count": 0,
            "bbox": None,
            "face_count": 0,
            "edge_count": 0,
            "vertex_count": 0,
            "surface_type_counts": {{}},
            "cylinder_radii": [],
            "cylinder_diameters": [],
        }}
    payload = {{
        "has_shape": True,
        "is_valid": bool(shape.isValid()),
        "volume": float(shape.Volume),
        "area": float(shape.Area),
        "solid_count": len(list(shape.Solids)),
        "bbox": bbox_payload(shape),
    }}
    payload.update(shape_stats(shape))
    return payload


metrics = {{
    "artifact": str(artifact_path),
    "errors": [],
    "object_count": 0,
    "body_count": 0,
    "solid_count": 0,
    "volume": 0.0,
    "objects": [],
    "bodies": [],
}}

try:
    doc = App.openDocument(str(artifact_path))
    doc.recompute()
except Exception as exc:
    metrics["errors"].append(f"open_error: {{type(exc).__name__}}: {{exc}}")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    raise

objects = list(doc.Objects)
bodies = [obj for obj in objects if obj.TypeId == "PartDesign::Body"]

metrics["object_count"] = len(objects)
metrics["body_count"] = len(bodies)

for obj in objects:
    shape = getattr(obj, "Shape", None)
    property_names = list(getattr(obj, "PropertiesList", []))
    property_values = {{}}
    for prop in property_names:
        if prop in {{"Radius", "Radius1", "Radius2", "Diameter", "Height", "Length", "Width", "Thickness", "Depth", "Angle"}}:
            try:
                property_values[prop] = jsonable_property(getattr(obj, prop))
            except Exception:
                pass
    payload = {{
        "name": obj.Name,
        "label": obj.Label,
        "type_id": obj.TypeId,
        "properties": property_names,
        "property_values": property_values,
    }}
    payload.update(shape_payload(shape))
    metrics["objects"].append(payload)

for body in bodies:
    tip = getattr(body, "Tip", None)
    shape_owner = tip or body
    shape = getattr(shape_owner, "Shape", None)
    payload = {{
        "name": body.Name,
        "label": body.Label,
        "type_id": body.TypeId,
        "tip": getattr(tip, "Name", None),
    }}
    payload.update(shape_payload(shape))
    metrics["bodies"].append(payload)

if metrics["bodies"]:
    first_body = metrics["bodies"][0]
    metrics["solid_count"] = first_body["solid_count"]
    metrics["volume"] = first_body["volume"]
    metrics["bbox"] = first_body["bbox"]

metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"Wrote {{metrics_path}}")
""",
        encoding="utf-8",
    )
    return inspector_path


def run_geometry_inspection(
    run_dir: Path,
    artifact_name: str = "candidate.FCStd",
    freecadcmd: Optional[str] = None,
    timeout_seconds: int = 120,
) -> ExecutionResult:
    artifact_path = run_dir / artifact_name
    stdout_path = run_dir / "inspect_stdout.log"
    stderr_path = run_dir / "inspect_stderr.log"

    if not artifact_path.exists():
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(f"Artifact does not exist: {artifact_path}\n", encoding="utf-8")
        return ExecutionResult(
            status="missing_artifact",
            command=[],
            returncode=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            run_dir=run_dir,
            message=f"Cannot inspect missing artifact: {artifact_path}",
        )

    binary = find_freecadcmd(freecadcmd)
    if not binary:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("FreeCADCmd/freecadcmd was not found.\n", encoding="utf-8")
        return ExecutionResult(
            status="missing_freecad",
            command=[],
            returncode=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            run_dir=run_dir,
            message="Install FreeCAD or pass --freecadcmd to inspect geometry.",
        )

    inspector_path = write_geometry_inspector(run_dir, artifact_name)
    command = [binary, str(inspector_path)]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=str(run_dir),
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            stderr.write(f"Timed out after {timeout_seconds} seconds.\n")
            return ExecutionResult(
                status="timeout",
                command=command,
                returncode=None,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                run_dir=run_dir,
                message=f"Geometry inspection timed out after {timeout_seconds} seconds.",
            )

    return ExecutionResult(
        status="ok" if completed.returncode == 0 else "error",
        command=command,
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        run_dir=run_dir,
    )
