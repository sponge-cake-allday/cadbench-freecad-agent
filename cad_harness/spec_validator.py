import re
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .models import CadTask


LENGTH_UNITS_TO_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
}

PARAM_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<name>[A-Za-z][A-Za-z0-9_]{1,80})\s*=\s*"
    r"(?P<expr>.*?)(?=(?:\s+(?:and|with)\s+)?[A-Za-z][A-Za-z0-9_]{1,80}\s*=|[\n,;#]|$)",
    flags=re.IGNORECASE,
)
NUMBER_UNIT_RE = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|millimeters?|cm|centimeters?|m|meters?|in|inches?|ft|feet|foot)\b",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

DIMENSION_WORDS = (
    "length",
    "width",
    "height",
    "thickness",
    "diameter",
    "radius",
    "depth",
    "span",
    "pitch",
)
INTERNAL_WORDS = (
    "hole",
    "bore",
    "inner",
    "shaft",
    "slot",
    "keyway",
    "through",
    "opening",
)
OUTER_WORDS = (
    "outer",
    "overall",
    "body",
    "plate",
    "flange",
    "washer",
    "rail",
    "rung",
)
PRIMITIVE_HINTS = {
    "cylinder": ("cylinder", "hole", "bore", "round", "circular", "shaft", "tube", "pipe"),
    "box": ("box", "block", "plate", "rectangular", "square", "rail", "rung", "prism"),
    "cone": ("cone", "conical", "taper", "chamfer"),
    "sphere": ("sphere", "ball", "dome"),
}


@dataclass(frozen=True)
class SpecCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "important"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
        }


def parse_numeric_mm(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value)
    unit_matches = list(NUMBER_UNIT_RE.finditer(text))
    if unit_matches:
        match = unit_matches[-1]
        unit = match.group("unit").lower()
        return float(match.group("value")) * LENGTH_UNITS_TO_MM[unit]

    numbers = NUMBER_RE.findall(text)
    if numbers:
        return float(numbers[-1])
    return None


def normalize_param_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def extract_prompt_parameters(task: CadTask) -> Dict[str, float]:
    params: Dict[str, float] = {}

    for key, value in task.key_parameters.items():
        parsed = parse_numeric_mm(value)
        if parsed is not None:
            params[normalize_param_name(key)] = parsed

    prompt = task.prompt_text()
    for match in PARAM_ASSIGNMENT_RE.finditer(prompt):
        name = normalize_param_name(match.group("name"))
        if len(name) > 80:
            continue
        expr = match.group("expr")
        for delimiter in (" and ", " with "):
            head, separator, tail = expr.partition(delimiter)
            if separator and "=" in tail:
                expr = head
                break
        parsed = parse_numeric_mm(expr)
        if parsed is not None:
            params[name] = parsed

    return params


def _body_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    bodies = metrics.get("bodies") or []
    if bodies:
        return bodies[0]
    return {}


def _bbox_lengths(metrics: Dict[str, Any]) -> List[float]:
    body = _body_metrics(metrics)
    bbox = body.get("bbox") or metrics.get("bbox") or {}
    lengths = [
        bbox.get("x_length"),
        bbox.get("y_length"),
        bbox.get("z_length"),
    ]
    return [float(value) for value in lengths if isinstance(value, (int, float))]


def _finite_positive_lengths(values: Sequence[float]) -> bool:
    return bool(values) and all(math.isfinite(value) and value > 0 for value in values)


def _object_records(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(metrics.get("objects") or [])


def _feature_records(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    features = []
    for obj in _object_records(metrics):
        type_id = str(obj.get("type_id") or "")
        if type_id.startswith("PartDesign::") and type_id != "PartDesign::Body":
            features.append(obj)
    return features


def _surface_type_counts(metrics: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    sources = [_body_metrics(metrics), metrics]
    for source in sources:
        for name, count in (source.get("surface_type_counts") or {}).items():
            try:
                counts[str(name)] = max(counts.get(str(name), 0), int(count))
            except (TypeError, ValueError):
                continue
    return counts


def _numeric_dimension_properties(metrics: Dict[str, Any]) -> List[tuple[str, float]]:
    values: List[tuple[str, float]] = []
    for obj in _object_records(metrics):
        for name, value in (obj.get("property_values") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append((str(name).lower(), float(value)))
    return values


def _cylinder_dimensions(metrics: Dict[str, Any]) -> Dict[str, List[float]]:
    radii: List[float] = []
    diameters: List[float] = []
    for source in [_body_metrics(metrics), metrics, *_object_records(metrics)]:
        for radius in source.get("cylinder_radii") or []:
            if isinstance(radius, (int, float)) and radius > 0:
                radii.append(float(radius))
                diameters.append(float(radius) * 2.0)
        for diameter in source.get("cylinder_diameters") or []:
            if isinstance(diameter, (int, float)) and diameter > 0:
                diameters.append(float(diameter))
                radii.append(float(diameter) / 2.0)
    for name, value in _numeric_dimension_properties(metrics):
        if value <= 0:
            continue
        if "radius" in name:
            radii.append(value)
            diameters.append(value * 2.0)
        elif "diameter" in name:
            diameters.append(value)
            radii.append(value / 2.0)
    return {
        "radii": sorted({round(value, 6) for value in radii}),
        "diameters": sorted({round(value, 6) for value in diameters}),
    }


def _shape_text(metrics: Dict[str, Any]) -> str:
    parts = []
    for obj in _object_records(metrics):
        parts.append(str(obj.get("name") or ""))
        parts.append(str(obj.get("label") or ""))
        parts.append(str(obj.get("type_id") or ""))
    return " ".join(parts).lower()


def _metadata_text(metrics: Dict[str, Any]) -> str:
    parts = [_shape_text(metrics)]
    for obj in _object_records(metrics):
        parts.extend(str(prop) for prop in obj.get("properties") or [])
    return " ".join(parts).lower()


def _close_to_any(value: float, candidates: Iterable[float], rel_tol: float = 0.12) -> bool:
    for candidate in candidates:
        tolerance = max(2.0, abs(value) * rel_tol)
        if abs(value - candidate) <= tolerance:
            return True
    return False


def _matched_dimension_names(
    params: Dict[str, float],
    bbox_lengths: Sequence[float],
    rel_tol: float = 0.12,
) -> Dict[str, float]:
    return {
        name: value
        for name, value in params.items()
        if _is_outer_dimension_param(name)
        and value > 0
        and _close_to_any(value, bbox_lengths, rel_tol=rel_tol)
    }


def _is_dimension_param(name: str) -> bool:
    return any(word in name for word in DIMENSION_WORDS)


def _is_outer_dimension_param(name: str) -> bool:
    if not _is_dimension_param(name):
        return False
    if any(word in name for word in OUTER_WORDS):
        return True
    return not any(word in name for word in INTERNAL_WORDS)


def _is_internal_dimension_param(name: str) -> bool:
    return _is_dimension_param(name) and any(word in name for word in INTERNAL_WORDS)


def _is_radius_param(name: str) -> bool:
    return "radius" in name and "diameter" not in name


def validate_spec(task: CadTask, metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    checks: List[SpecCheck] = []
    params = extract_prompt_parameters(task)

    if not metrics:
        checks.append(
            SpecCheck(
                name="spec:geometry_metrics_available",
                passed=False,
                detail="geometry_metrics.json was not produced",
                severity="critical",
            )
        )
        return _report(checks, params, [])

    errors = metrics.get("errors") or []
    body_count = int(metrics.get("body_count") or len(metrics.get("bodies") or []))
    body = _body_metrics(metrics)
    solid_count = int(body.get("solid_count") or metrics.get("solid_count") or 0)
    volume = float(body.get("volume") or metrics.get("volume") or 0.0)
    object_records = _object_records(metrics)
    feature_records = _feature_records(metrics)
    surface_type_counts = _surface_type_counts(metrics)
    cylinder_dimensions = _cylinder_dimensions(metrics)
    shape_text = _shape_text(metrics)
    metadata_text = _metadata_text(metrics)
    bbox_lengths = _bbox_lengths(metrics)
    shape_valid = bool(body.get("is_valid", metrics.get("is_valid", True)))

    checks.extend(
        [
            SpecCheck(
                "spec:fcstd_opened",
                not errors,
                "opened generated FCStd without inspector errors"
                if not errors
                else "; ".join(str(error) for error in errors),
                "critical",
            ),
            SpecCheck(
                "spec:single_partdesign_body",
                body_count == 1,
                f"body_count={body_count}",
                "critical",
            ),
            SpecCheck(
                "spec:single_nonempty_solid",
                solid_count == 1 and volume > 0,
                f"solid_count={solid_count}, volume={volume:.6g}",
                "critical",
            ),
            SpecCheck(
                "spec:valid_shape",
                shape_valid,
                f"is_valid={shape_valid}",
                "critical",
            ),
            SpecCheck(
                "spec:finite_positive_bbox",
                _finite_positive_lengths(bbox_lengths),
                f"bbox_lengths={bbox_lengths}",
                "critical",
            ),
        ]
    )

    mesh_like = [
        obj
        for obj in object_records
        if "mesh" in str(obj.get("type_id") or "").lower()
    ]
    checks.append(
        SpecCheck(
            "spec:not_mesh_only",
            len(mesh_like) == 0,
            f"mesh_like_object_count={len(mesh_like)}",
            "important",
        )
    )

    opaque_part_features = [
        obj for obj in object_records if str(obj.get("type_id") or "") == "Part::Feature"
    ]
    checks.append(
        SpecCheck(
            "spec:editable_partdesign_features",
            len(feature_records) >= 1 and not (len(object_records) <= 2 and opaque_part_features),
            f"partdesign_feature_count={len(feature_records)}, opaque_part_feature_count={len(opaque_part_features)}",
            "important",
        )
    )

    outer_dimensions = {
        name: value
        for name, value in params.items()
        if _is_outer_dimension_param(name) and value > 0
    }
    matched_dimensions = _matched_dimension_names(params, bbox_lengths)
    if outer_dimensions:
        coverage = len(matched_dimensions) / max(1, min(len(outer_dimensions), 3))
        outer_anchor_passed = len(matched_dimensions) > 0 or len(outer_dimensions) < 2
        checks.append(
            SpecCheck(
                "spec:bbox_matches_prompt_dimensions",
                len(matched_dimensions) > 0,
                f"matched={sorted(matched_dimensions)[:6]}, checked={sorted(outer_dimensions)[:10]}, bbox_lengths={bbox_lengths}",
                "advisory",
            )
        )
        checks.append(
            SpecCheck(
                "spec:outer_dimension_anchor",
                outer_anchor_passed,
                f"matched={sorted(matched_dimensions)[:6]}, checked={sorted(outer_dimensions)[:10]}, bbox_lengths={bbox_lengths}",
                "advisory",
            )
        )
        checks.append(
            SpecCheck(
                "spec:bbox_dimension_coverage",
                coverage >= 0.5,
                f"coverage={coverage:.3g}, matched={sorted(matched_dimensions)[:8]}, bbox_lengths={bbox_lengths}",
                "advisory",
            )
        )

    internal_dimensions = {
        name: value
        for name, value in params.items()
        if _is_internal_dimension_param(name) and value > 0
    }
    if internal_dimensions:
        matched_internal = {}
        for name, value in internal_dimensions.items():
            candidates = (
                cylinder_dimensions["radii"] if _is_radius_param(name) else cylinder_dimensions["diameters"]
            )
            if _close_to_any(value, candidates, rel_tol=0.12):
                matched_internal[name] = value
        checks.append(
            SpecCheck(
                "spec:cylindrical_dimension_matches_prompt",
                bool(matched_internal),
                (
                    f"matched={sorted(matched_internal)}, checked={sorted(internal_dimensions)[:10]}, "
                    f"cylinder_dimensions={cylinder_dimensions}"
                ),
                "advisory",
            )
        )

    if params:
        param_terms = set()
        for name in params:
            param_terms.add(name)
            param_terms.update(part for part in name.split("_") if len(part) >= 4)
        matched_param_terms = sorted(term for term in param_terms if term in metadata_text)
        checks.append(
            SpecCheck(
                "spec:parametric_metadata_hint",
                len(matched_param_terms) >= min(3, max(1, len(param_terms) // 6)),
                f"matched_param_terms={matched_param_terms[:12]}",
                "advisory",
            )
        )

        largest_name, largest_value = max(outer_dimensions.items(), key=lambda item: item[1])
        checks.append(
            SpecCheck(
                "spec:largest_dimension_plausible",
                _close_to_any(largest_value, bbox_lengths, rel_tol=0.18),
                f"{largest_name}={largest_value:g}mm, bbox_lengths={bbox_lengths}",
                "advisory",
            )
        )

    prompt = task.prompt_text().lower()
    hole_like_prompt = any(
        word in prompt
        for word in ("hole", "bore", "through-hole", "through hole", "central opening", "shaft opening")
    )
    if any(word in prompt for word in ("hole", "bore", "through", "cut", "opening")):
        checks.append(
            SpecCheck(
                "spec:cut_feature_hint",
                any(word in shape_text for word in ("subtractive", "pocket", "hole", "bore", "cut", "cylinder")),
                "prompt mentions a hole/bore/cut; searched generated object names and types",
                "advisory",
            )
        )
        cut_features = [
            obj
            for obj in feature_records
            if "subtractive" in str(obj.get("type_id") or "").lower()
            or "pocket" in str(obj.get("type_id") or "").lower()
        ]
        checks.append(
            SpecCheck(
                "spec:cut_feature_count",
                len(cut_features) >= 1,
                f"cut_feature_count={len(cut_features)}",
                "advisory",
            )
        )
        cylindrical_face_count = sum(
            count
            for surface_type, count in surface_type_counts.items()
            if "cylinder" in surface_type.lower()
        )
        checks.append(
            SpecCheck(
                "spec:cylindrical_cut_geometry_hint",
                cylindrical_face_count >= 1
                or bool(cylinder_dimensions["diameters"])
                or any("cylinder" in str(obj.get("type_id") or "").lower() for obj in cut_features),
                (
                    f"cylindrical_face_count={cylindrical_face_count}, "
                    f"cylinder_dimensions={cylinder_dimensions}, surface_type_counts={surface_type_counts}"
                ),
                "advisory",
            )
        )

    count_params = {
        name: value
        for name, value in params.items()
        if ("number" in name or "count" in name) and value >= 2
    }
    if count_params:
        largest_count = int(max(count_params.values()))
        checks.append(
            SpecCheck(
                "spec:repeated_feature_hint",
                len(feature_records) >= max(1, min(largest_count, 5)),
                f"count_params={count_params}, partdesign_feature_count={len(feature_records)}",
                "advisory",
            )
        )
        object_names = [str(obj.get("name") or obj.get("label") or "").lower() for obj in object_records]
        count_terms = []
        for name in count_params:
            parts = [part for part in name.split("_") if len(part) >= 3]
            if len(parts) >= 2 and parts[-1] in {"count", "number"}:
                count_terms.append(parts[-2])
            count_terms.append(parts[-1])
        expanded_count_terms = set(count_terms)
        for term in count_terms:
            if term.endswith("s") and len(term) > 3:
                expanded_count_terms.add(term[:-1])
        matching_named_features = sum(
            1
            for object_name in object_names
            if any(term in object_name for term in expanded_count_terms)
        )
        checks.append(
            SpecCheck(
                "spec:repeated_feature_names",
                matching_named_features >= min(int(max(count_params.values())), 3),
                f"count_terms={sorted(expanded_count_terms)}, matching_named_features={matching_named_features}",
                "advisory",
            )
        )

    for primitive, hints in PRIMITIVE_HINTS.items():
        if any(hint in prompt for hint in hints):
            matching_features = [
                obj
                for obj in feature_records
                if primitive in str(obj.get("type_id") or "").lower()
                or primitive in str(obj.get("name") or "").lower()
                or primitive in str(obj.get("label") or "").lower()
            ]
            checks.append(
                SpecCheck(
                    f"spec:{primitive}_feature_hint",
                    len(matching_features) >= 1,
                    f"prompt_hints={hints}, matching_features={len(matching_features)}",
                    "advisory",
                )
            )

    if any(word in prompt for word in ("rectangular", "square", "box", "plate", "block", "prism")):
        face_count = int(body.get("face_count") or metrics.get("face_count") or 0)
        planar_face_count = sum(
            count
            for surface_type, count in surface_type_counts.items()
            if "plane" in surface_type.lower()
        )
        checks.append(
            SpecCheck(
                "spec:planar_prismatic_geometry_hint",
                face_count >= 6 or planar_face_count >= 6,
                f"face_count={face_count}, planar_face_count={planar_face_count}",
                "advisory",
            )
        )

    return _report(checks, params, bbox_lengths)


def _report(
    checks: List[SpecCheck],
    params: Dict[str, float],
    bbox_lengths: List[float],
) -> Dict[str, Any]:
    weights = {"critical": 3.0, "important": 2.0, "advisory": 1.0}
    total = sum(weights.get(check.severity, 1.0) for check in checks)
    passed = sum(weights.get(check.severity, 1.0) for check in checks if check.passed)
    score = passed / total if total else 0.0
    status = (
        "ok"
        if all(check.passed or check.severity == "advisory" for check in checks)
        else "needs_repair"
    )
    return {
        "status": status,
        "score": round(score, 6),
        "checks": [check.to_dict() for check in checks],
        "parsed_parameters": params,
        "bbox_lengths": bbox_lengths,
        "method": "prompt-grounded local validation; no reference CAD or benchmark scorer",
    }
