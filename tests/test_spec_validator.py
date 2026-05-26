import unittest

from cad_harness.models import CadTask
from cad_harness.spec_validator import (
    extract_prompt_parameters,
    parse_numeric_mm,
    validate_spec,
)


class SpecValidatorTests(unittest.TestCase):
    def test_parse_numeric_mm_prefers_final_length_unit(self):
        self.assertAlmostEqual(parse_numeric_mm("2 in * 25.4 = 50.8 mm"), 50.8)
        self.assertAlmostEqual(parse_numeric_mm("1 ft * 12 in/ft * 25.4 = 304.8 mm"), 304.8)
        self.assertAlmostEqual(parse_numeric_mm("0.5 in"), 12.7)

    def test_extract_prompt_parameters_from_task(self):
        task = CadTask(
            task_id="task",
            name="washer",
            description="A square washer.",
            key_parameters={
                "washer_width": "40 mm",
                "washer_thickness": 5,
                "diametral_pitch": "25.4 / 12 = 2.1166666667 mm",
            },
        )

        params = extract_prompt_parameters(task)

        self.assertEqual(params["washer_width"], 40)
        self.assertEqual(params["washer_thickness"], 5)
        self.assertAlmostEqual(params["diametral_pitch"], 2.1166666667)

    def test_extract_prompt_parameters_does_not_swallow_prose(self):
        task = CadTask(
            task_id="task",
            name="ladder",
            description="A ladder has ladder_number_of_rungs = 7 and side rail length = 2400 mm.",
            key_parameters={},
        )

        params = extract_prompt_parameters(task)

        self.assertEqual(params["ladder_number_of_rungs"], 7)
        self.assertNotIn("a_ladder_has_ladder_number_of_rungs", params)

    def test_valid_partdesign_metrics_score_ok(self):
        task = CadTask(
            task_id="task",
            name="square washer",
            description="A square washer with a central circular through-hole.",
            key_parameters={
                "washer_width": 40,
                "washer_thickness": 5,
                "washer_inner_hole_diameter": 7.5,
            },
        )
        metrics = {
            "errors": [],
            "body_count": 1,
            "object_count": 3,
            "solid_count": 1,
            "volume": 7800,
            "bbox": {"x_length": 40, "y_length": 40, "z_length": 5},
            "face_count": 8,
            "is_valid": True,
            "surface_type_counts": {"Part::GeomPlane": 7, "Part::GeomCylinder": 1},
            "cylinder_radii": [3.75],
            "cylinder_diameters": [7.5],
            "bodies": [
                {
                    "name": "Body",
                    "label": "Body",
                    "type_id": "PartDesign::Body",
                    "solid_count": 1,
                    "volume": 7800,
                    "is_valid": True,
                    "bbox": {"x_length": 40, "y_length": 40, "z_length": 5},
                    "face_count": 8,
                    "surface_type_counts": {"Part::GeomPlane": 7, "Part::GeomCylinder": 1},
                    "cylinder_radii": [3.75],
                    "cylinder_diameters": [7.5],
                }
            ],
            "objects": [
                {
                    "name": "Body",
                    "label": "Body",
                    "type_id": "PartDesign::Body",
                    "properties": ["washer_width", "washer_thickness", "washer_inner_hole_diameter"],
                },
                {"name": "Plate", "label": "Plate", "type_id": "PartDesign::AdditiveBox", "properties": []},
                {
                    "name": "CentralHole",
                    "label": "CentralHole",
                    "type_id": "PartDesign::SubtractiveCylinder",
                    "properties": ["Radius", "Height"],
                },
            ],
        }

        report = validate_spec(task, metrics)

        self.assertEqual(report["status"], "ok")
        self.assertGreater(report["score"], 0.9)
        self.assertIn(
            "spec:cut_feature_count",
            {check["name"] for check in report["checks"]},
        )
        checks = {check["name"]: check for check in report["checks"]}
        self.assertTrue(checks["spec:cylindrical_cut_geometry_hint"]["passed"])
        self.assertTrue(checks["spec:cylindrical_dimension_matches_prompt"]["passed"])
        self.assertTrue(checks["spec:planar_prismatic_geometry_hint"]["passed"])
        self.assertTrue(checks["spec:parametric_metadata_hint"]["passed"])

    def test_invalid_shape_needs_repair(self):
        task = CadTask(
            task_id="task",
            name="box",
            description="A rectangular block with body_length = 20 mm and body_width = 10 mm.",
            key_parameters={},
        )
        metrics = {
            "errors": [],
            "body_count": 1,
            "solid_count": 1,
            "volume": 200,
            "bbox": {"x_length": 20, "y_length": 10, "z_length": 1},
            "bodies": [
                {
                    "type_id": "PartDesign::Body",
                    "solid_count": 1,
                    "volume": 200,
                    "is_valid": False,
                    "bbox": {"x_length": 20, "y_length": 10, "z_length": 1},
                }
            ],
            "objects": [
                {"name": "Body", "label": "Body", "type_id": "PartDesign::Body"},
                {"name": "Block", "label": "Block", "type_id": "PartDesign::AdditiveBox"},
            ],
        }

        report = validate_spec(task, metrics)
        checks = {check["name"]: check for check in report["checks"]}

        self.assertEqual(report["status"], "needs_repair")
        self.assertFalse(checks["spec:valid_shape"]["passed"])

    def test_internal_cylindrical_dimension_mismatch_is_advisory(self):
        task = CadTask(
            task_id="task",
            name="washer",
            description="A washer with washer_outer_diameter = 40 mm and washer_inner_hole_diameter = 7.5 mm.",
            key_parameters={},
        )
        metrics = {
            "errors": [],
            "body_count": 1,
            "solid_count": 1,
            "volume": 100,
            "bbox": {"x_length": 40, "y_length": 40, "z_length": 5},
            "cylinder_radii": [2.0],
            "cylinder_diameters": [4.0],
            "bodies": [
                {
                    "type_id": "PartDesign::Body",
                    "solid_count": 1,
                    "volume": 100,
                    "is_valid": True,
                    "bbox": {"x_length": 40, "y_length": 40, "z_length": 5},
                    "cylinder_radii": [2.0],
                    "cylinder_diameters": [4.0],
                }
            ],
            "objects": [
                {"name": "Body", "label": "Body", "type_id": "PartDesign::Body"},
                {"name": "Plate", "label": "Plate", "type_id": "PartDesign::AdditiveCylinder"},
                {"name": "Hole", "label": "Hole", "type_id": "PartDesign::SubtractiveCylinder"},
            ],
        }

        report = validate_spec(task, metrics)
        checks = {check["name"]: check for check in report["checks"]}

        self.assertEqual(report["status"], "ok")
        self.assertFalse(checks["spec:cylindrical_dimension_matches_prompt"]["passed"])
        self.assertEqual(checks["spec:cylindrical_dimension_matches_prompt"]["severity"], "advisory")

    def test_repeated_feature_checks_are_prompt_grounded(self):
        task = CadTask(
            task_id="task",
            name="ladder",
            description=(
                "A ladder has ladder_number_of_rungs = 7 and rectangular "
                "rungs between side rails. ladder_side_rails_length = 2400 mm, "
                "side_rail_outer_to_outer_width = 480 mm, ladder_rung_thickness = 40 mm"
            ),
            key_parameters={},
        )
        metrics = {
            "errors": [],
            "body_count": 1,
            "solid_count": 1,
            "volume": 100,
            "bbox": {"x_length": 2400, "y_length": 480, "z_length": 40},
            "is_valid": True,
            "face_count": 24,
            "surface_type_counts": {"Part::GeomPlane": 24},
            "bodies": [
                {
                    "type_id": "PartDesign::Body",
                    "solid_count": 1,
                    "volume": 100,
                    "is_valid": True,
                    "bbox": {"x_length": 2400, "y_length": 480, "z_length": 40},
                    "face_count": 24,
                    "surface_type_counts": {"Part::GeomPlane": 24},
                }
            ],
            "objects": [
                {"name": "Body", "label": "Body", "type_id": "PartDesign::Body"},
                {"name": "SideRailLeft", "label": "SideRailLeft", "type_id": "PartDesign::AdditiveBox"},
                {"name": "SideRailRight", "label": "SideRailRight", "type_id": "PartDesign::AdditiveBox"},
                {"name": "Rung1", "label": "Rung1", "type_id": "PartDesign::AdditiveBox"},
                {"name": "Rung2", "label": "Rung2", "type_id": "PartDesign::AdditiveBox"},
                {"name": "Rung3", "label": "Rung3", "type_id": "PartDesign::AdditiveBox"},
            ],
        }

        report = validate_spec(task, metrics)
        checks = {check["name"]: check for check in report["checks"]}

        self.assertTrue(checks["spec:repeated_feature_hint"]["passed"])
        self.assertTrue(checks["spec:repeated_feature_names"]["passed"])
        self.assertTrue(checks["spec:box_feature_hint"]["passed"])

    def test_missing_metrics_needs_repair(self):
        task = CadTask(task_id="task", name="example", description="example", key_parameters={})

        report = validate_spec(task, None)

        self.assertEqual(report["status"], "needs_repair")
        self.assertEqual(report["checks"][0]["name"], "spec:geometry_metrics_available")


if __name__ == "__main__":
    unittest.main()
