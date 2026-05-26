from dataclasses import dataclass
import re
from typing import Any, Dict, List

from .models import FeedbackReport


@dataclass
class RepairDecision:
    action: str
    reason: str
    repairable: bool
    suggestions: List[str]


def _combined_output(*payloads: Dict[str, Any]) -> str:
    parts = []
    for payload in payloads:
        parts.extend(str(payload.get(key) or "") for key in ("stdout", "stderr"))
    return "\n".join(part for part in parts if part)


def _failed_spec_checks(
    artifact_payload: Dict[str, Any],
    *,
    include_advisory: bool = False,
) -> List[Dict[str, Any]]:
    spec = artifact_payload.get("spec_validation") or {}
    return [
        check
        for check in spec.get("checks", [])
        if not check.get("passed")
        and (include_advisory or check.get("severity") != "advisory")
    ]


def _add_unique(suggestions: List[str], suggestion: str) -> None:
    if suggestion not in suggestions:
        suggestions.append(suggestion)


def _format_params(params: Dict[str, Any], limit: int = 14) -> str:
    items = []
    for name, value in sorted(params.items()):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            items.append(f"{name}={value:g}")
    return ", ".join(items[:limit])


def build_geometry_intent_notes(artifact_payload: Dict[str, Any]) -> List[str]:
    spec = artifact_payload.get("spec_validation") or {}
    params = spec.get("parsed_parameters") or {}
    bbox_lengths = spec.get("bbox_lengths") or []
    failed_checks = _failed_spec_checks(artifact_payload, include_advisory=True)
    failed_names = {str(check.get("name") or "") for check in failed_checks}
    notes: List[str] = []

    if params:
        formatted = _format_params(params)
        if formatted:
            notes.append(f"Preserve and use these parsed prompt parameters: {formatted}.")

    count_params = {
        name: value
        for name, value in params.items()
        if isinstance(value, (int, float)) and ("count" in name or "number" in name) and value >= 2
    }
    if count_params:
        notes.append(
            "Repeated-feature intent: create visible repeated features for "
            + _format_params(count_params, limit=8)
            + "; avoid replacing the pattern with one anonymous blob."
        )

    internal_params = {
        name: value
        for name, value in params.items()
        if isinstance(value, (int, float))
        and any(word in name for word in ("hole", "bore", "inner", "shaft", "slot", "keyway", "opening"))
    }
    if internal_params:
        notes.append(
            "Internal/cylindrical intent: match holes, bores, shafts, slots, and openings to "
            + _format_params(internal_params, limit=8)
            + "; diameter parameters need cylinder Radius = diameter / 2."
        )

    if bbox_lengths:
        notes.append(f"Current generated bounding-box lengths: {bbox_lengths}.")

    if {"spec:bbox_matches_prompt_dimensions", "spec:bbox_dimension_coverage", "spec:outer_dimension_anchor"} & failed_names:
        notes.append("Scale/extent hint: adjust the main stock/body dimensions to match the largest prompt dimensions instead of default placeholder sizes.")
    if "spec:cylindrical_dimension_matches_prompt" in failed_names:
        notes.append("Cylindrical-dimension hint: a cylinder radius/diameter in the model does not match a parsed hole/bore/shaft parameter.")
    if "spec:cut_feature_count" in failed_names or "spec:cylindrical_cut_geometry_hint" in failed_names:
        notes.append("Cut/opening hint: add an actual subtractive cut or pocket for requested holes, bores, through-cuts, slots, or openings.")
    if "spec:repeated_feature_hint" in failed_names or "spec:repeated_feature_names" in failed_names:
        notes.append("Pattern hint: represent requested counts with separate named features in a loop when FreeCAD pattern APIs are uncertain.")
    if "spec:parametric_metadata_hint" in failed_names:
        notes.append("Parametricity hint: keep prompt parameter names as Python variables and meaningful object labels.")

    return notes[:10]


def build_repair_guidance(
    execution_payload: Dict[str, Any],
    artifact_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify local failures into targeted, reference-free repair instructions."""
    output = _combined_output(execution_payload, artifact_payload)
    lowered = output.lower()
    failed_checks = _failed_spec_checks(artifact_payload)
    all_failed_checks = _failed_spec_checks(artifact_payload, include_advisory=True)
    failed_names = {str(check.get("name") or "") for check in failed_checks}
    all_failed_names = {str(check.get("name") or "") for check in all_failed_checks}
    suggestions: List[str] = [
        "Patch the previous script instead of sampling an unrelated design; preserve working parameters and feature intent.",
        "Return a complete replacement script only, with no Markdown fences or prose.",
    ]
    failure_kind = "local_validation_failed"

    if (
        re.search(r"\b(syntaxerror|indentationerror|eoferror)\b", lowered)
        or "invalid syntax" in lowered
        or "unmatched" in lowered
        or "was never closed" in lowered
        or "unexpected eof" in lowered
    ):
        failure_kind = "python_syntax_or_truncation"
        _add_unique(suggestions, "Fix malformed Python first: complete every assignment/expression, close brackets/strings, and return the full script.")
    if "no module named 'path'" in lowered or re.search(r"^\s*import path\b", output, flags=re.MULTILINE):
        failure_kind = "bad_path_import"
        _add_unique(suggestions, "Use `from pathlib import Path`; do not import FreeCAD's Path module for filesystem paths.")
    if "has no attribute 'sin'" in lowered or "has no attribute 'cos'" in lowered or "has no attribute 'radians'" in lowered:
        failure_kind = "math_api_error"
        _add_unique(suggestions, "Use Python `math.sin`, `math.cos`, `math.radians`, and `math.pi`; FreeCAD/App does not provide trig helpers.")
    unsupported_type = re.search(r"'([^']+)' is not a document object type", output)
    if (
        "additivepad" in lowered
        or "additivefeature" in lowered
        or "additivepocket" in lowered
        or unsupported_type
    ):
        failure_kind = "invalid_partdesign_type"
        invalid_type = unsupported_type.group(1) if unsupported_type else "the invalid PartDesign type"
        _add_unique(suggestions, f"Replace {invalid_type} with FreeCAD 0.21 types such as PartDesign::Pad, PartDesign::Pocket, PartDesign::AdditiveBox, or PartDesign::SubtractiveCylinder.")
        _add_unique(suggestions, "For repeated teeth/holes/slots, create individual primitive features in a loop instead of relying on uncertain transform or polygon feature types.")
    attribute = re.search(r"object has no attribute '([^']+)'", output)
    if attribute:
        failure_kind = "invented_freecad_property"
        attr = attribute.group(1)
        _add_unique(suggestions, f"Remove assignments to nonexistent FreeCAD property `{attr}`; keep semantic info in Python variables, comments, or object Label instead.")
        _add_unique(suggestions, "Only set documented primitive properties such as Length, Width, Height, Radius, Angle, Placement, Label, and MapMode when needed.")
    if "sketcher.line" in lowered or "sketcher.circle" in lowered or "sketcher.geometry" in lowered:
        failure_kind = "invalid_sketcher_constructor"
        _add_unique(suggestions, "Create sketch geometry with Part.LineSegment/Part.Circle objects and pass them to sketch.addGeometry; Sketcher has constraints, not geometry constructors.")
    if "go out of the allowed scope" in lowered or "links to object" in lowered:
        failure_kind = "partdesign_scope_error"
        _add_unique(suggestions, "Keep sketches, additive features, and subtractive features inside the same Body; avoid linking Body features to external Part::Feature cutters.")
    if "not part of the enumeration" in lowered or "xy_plane" in lowered or ".xy_plane" in lowered:
        failure_kind = "invalid_sketch_attachment"
        _add_unique(suggestions, "Avoid fragile origin-plane and MapMode attachment code; use unattached sketches with explicit Placement, or use reliable PartDesign primitive features.")
    if "recompute failed" in lowered or "body_shape_is_null" in lowered:
        failure_kind = "recompute_or_null_shape"
        _add_unique(suggestions, "After each feature, call doc.recompute(); add every feature to the Body and set/leave the final Body Tip as the last valid feature.")
    if "missing_or_empty_fcstd" in lowered or "geometry_metrics.json was not produced" in lowered:
        _add_unique(suggestions, "Make the script save a nonempty FCStd at CAD_HARNESS_OUTPUT_FCSTD when set, otherwise /app/answer.FCStd.")

    if "spec:single_partdesign_body" in failed_names:
        failure_kind = "wrong_body_count"
        _add_unique(suggestions, "Create exactly one PartDesign::Body and add all modeled features to that Body.")
    if "spec:single_nonempty_solid" in failed_names:
        failure_kind = "empty_or_multi_solid"
        _add_unique(suggestions, "Ensure the Body resolves to one positive-volume solid; fuse/add features through PartDesign rather than leaving separate solids.")
    if "spec:valid_shape" in failed_names:
        failure_kind = "invalid_shape"
        _add_unique(suggestions, "Avoid self-intersecting booleans and zero-thickness cuts; keep subtractive cutters smaller than the surrounding stock where appropriate.")
    if "spec:finite_positive_bbox" in failed_names:
        failure_kind = "degenerate_bbox"
        _add_unique(suggestions, "Use positive numeric dimensions in millimeters for every primitive Length/Width/Height/Radius.")
    if "spec:editable_partdesign_features" in failed_names:
        _add_unique(suggestions, "Use editable PartDesign primitives or sketch-driven Pad/Pocket features rather than a lone opaque Part::Feature.")
    if "spec:outer_dimension_anchor" in all_failed_names:
        _add_unique(suggestions, "Align the main bounding-box dimensions with the largest prompt dimensions; do not leave the model at default placeholder size.")
    if "spec:cylindrical_dimension_matches_prompt" in all_failed_names:
        if "spec:cylindrical_dimension_matches_prompt" in failed_names or failure_kind == "local_validation_failed":
            failure_kind = "cylindrical_dimension_mismatch"
        _add_unique(suggestions, "For hole/bore/shaft dimensions, set cylinder Radius to diameter/2 when the prompt names a diameter; match the parsed prompt value.")
    if "spec:cut_feature_count" in all_failed_names or "spec:cylindrical_cut_geometry_hint" in all_failed_names:
        if (
            "spec:cut_feature_count" in failed_names
            or "spec:cylindrical_cut_geometry_hint" in failed_names
            or failure_kind == "local_validation_failed"
        ):
            failure_kind = "missing_cut_feature"
        _add_unique(suggestions, "If the prompt asks for a hole/bore/cut/opening, add a real PartDesign subtractive feature, usually PartDesign::SubtractiveCylinder or a sketch-driven Pocket.")
    if "spec:not_mesh_only" in failed_names:
        failure_kind = "mesh_only"
        _add_unique(suggestions, "Do not use mesh-only geometry; build a parametric solid with PartDesign features.")

    return {
        "failure_kind": failure_kind,
        "failed_checks": [
            {
                "name": check.get("name"),
                "severity": check.get("severity"),
                "detail": check.get("detail"),
            }
            for check in failed_checks[:8]
        ],
        "suggestions": suggestions[:10],
        "geometry_intent_notes": build_geometry_intent_notes(artifact_payload),
    }


def decide_next_step(report: FeedbackReport) -> RepairDecision:
    failed = [check for check in report.checks if not check["passed"]]
    execution_status = report.execution.get("status")

    if execution_status == "missing_freecad":
        return RepairDecision(
            action="stop",
            reason="FreeCADCmd/freecadcmd is unavailable, so another generated script cannot be tested.",
            repairable=False,
            suggestions=[
                "Install FreeCAD.",
                "Pass --freecadcmd /path/to/FreeCADCmd if FreeCAD is installed outside PATH.",
            ],
        )

    if execution_status == "timeout":
        return RepairDecision(
            action="repair",
            reason="The script timed out during local FreeCAD execution.",
            repairable=True,
            suggestions=[
                "Simplify expensive booleans or pattern operations.",
                "Check for accidental infinite loops.",
                "Create cutters once and fuse them before subtracting when possible.",
            ],
        )

    if execution_status == "error":
        return RepairDecision(
            action="repair",
            reason="FreeCAD returned a non-zero exit code.",
            repairable=True,
            suggestions=[
                "Read freecad_stderr.log and fix the failing FreeCAD API call.",
                "Confirm imports, document creation, recompute, and saveAs are valid.",
                "Verify that all dimensions are numeric and in millimeters.",
            ],
        )

    missing_artifacts = [
        check for check in failed if check["name"].startswith("artifact_exists:")
    ]
    if missing_artifacts:
        return RepairDecision(
            action="repair",
            reason="The script executed but did not produce the expected artifact.",
            repairable=True,
            suggestions=[
                "Ensure the script saves to CAD_HARNESS_OUTPUT_FCSTD when set.",
                "Call doc.recompute() before saving.",
                "Write the final .FCStd file to the run directory, not the source directory.",
            ],
        )

    spec_failures = [check for check in failed if check["name"].startswith("spec:")]
    if spec_failures:
        suggestions = ["Use the local spec_validation report to patch the generated script."]
        failed_names = {check["name"] for check in spec_failures}
        if "spec:single_partdesign_body" in failed_names:
            suggestions.append("Create exactly one PartDesign::Body and add all generated features to it.")
        if "spec:single_nonempty_solid" in failed_names:
            suggestions.append("Ensure the Body tip resolves to one valid solid with positive volume.")
        if "spec:editable_partdesign_features" in failed_names:
            suggestions.append("Prefer editable PartDesign Pad/Pocket/Additive/Subtractive features over a single opaque Part::Feature.")
        if "spec:not_mesh_only" in failed_names:
            suggestions.append("Do not output mesh-only geometry; build a parametric solid model.")
        return RepairDecision(
            action="repair",
            reason="The generated FCStd failed local prompt-grounded validation.",
            repairable=True,
            suggestions=suggestions,
        )

    if failed:
        return RepairDecision(
            action="repair",
            reason="One or more local checks failed.",
            repairable=True,
            suggestions=["Use the feedback report to patch the generated script."],
        )

    return RepairDecision(
        action="done",
        reason="All local checks passed.",
        repairable=False,
        suggestions=[],
    )
