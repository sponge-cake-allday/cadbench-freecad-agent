import unittest

from cad_harness.repair import build_repair_guidance


class RepairGuidanceTests(unittest.TestCase):
    def test_classifies_syntax_failure(self):
        guidance = build_repair_guidance(
            {"stdout": "", "stderr": "SyntaxError: invalid syntax", "return_code": 1},
            {"spec_validation": {"checks": []}},
        )

        self.assertEqual(guidance["failure_kind"], "python_syntax_or_truncation")
        self.assertTrue(any("complete every assignment" in item for item in guidance["suggestions"]))

    def test_classifies_cylindrical_dimension_failure(self):
        guidance = build_repair_guidance(
            {"stdout": "", "stderr": "", "return_code": 0},
            {
                "spec_validation": {
                    "checks": [
                        {
                            "name": "spec:cylindrical_dimension_matches_prompt",
                            "passed": False,
                            "severity": "important",
                            "detail": "checked=['washer_inner_hole_diameter']",
                        }
                    ]
                }
            },
        )

        self.assertEqual(guidance["failure_kind"], "cylindrical_dimension_mismatch")
        self.assertTrue(any("diameter/2" in item for item in guidance["suggestions"]))
        self.assertEqual(
            guidance["failed_checks"][0]["name"],
            "spec:cylindrical_dimension_matches_prompt",
        )

    def test_classifies_partdesign_scope_error(self):
        guidance = build_repair_guidance(
            {
                "stdout": "",
                "stderr": "Link(s) to object(s) 'Cut' go out of the allowed scope",
                "return_code": 1,
            },
            {"spec_validation": {"checks": []}},
        )

        self.assertEqual(guidance["failure_kind"], "partdesign_scope_error")
        self.assertTrue(any("same Body" in item for item in guidance["suggestions"]))

    def test_classifies_unsupported_document_object_type(self):
        guidance = build_repair_guidance(
            {
                "stdout": "",
                "stderr": "'PartDesign::FeatureTransform' is not a document object type",
                "return_code": 1,
            },
            {"spec_validation": {"checks": []}},
        )

        self.assertEqual(guidance["failure_kind"], "invalid_partdesign_type")
        self.assertTrue(any("FeatureTransform" in item for item in guidance["suggestions"]))

    def test_classifies_invented_freecad_property(self):
        guidance = build_repair_guidance(
            {
                "stdout": "",
                "stderr": "'PartDesign.Feature' object has no attribute 'Source'",
                "return_code": 1,
            },
            {"spec_validation": {"checks": []}},
        )

        self.assertEqual(guidance["failure_kind"], "invented_freecad_property")
        self.assertTrue(any("Source" in item for item in guidance["suggestions"]))

    def test_includes_geometry_intent_notes_from_advisory_checks(self):
        guidance = build_repair_guidance(
            {"stdout": "", "stderr": "", "return_code": 1},
            {
                "spec_validation": {
                    "parsed_parameters": {
                        "outer_diameter": 40.0,
                        "inner_hole_diameter": 7.5,
                        "number_of_teeth": 20.0,
                    },
                    "bbox_lengths": [10.0, 10.0, 2.0],
                    "checks": [
                        {
                            "name": "spec:bbox_dimension_coverage",
                            "passed": False,
                            "severity": "advisory",
                            "detail": "coverage=0",
                        },
                        {
                            "name": "spec:cylindrical_dimension_matches_prompt",
                            "passed": False,
                            "severity": "advisory",
                            "detail": "no match",
                        },
                    ],
                }
            },
        )

        notes = "\n".join(guidance["geometry_intent_notes"])
        self.assertIn("outer_diameter=40", notes)
        self.assertIn("diameter / 2", notes)
        self.assertIn("bounding-box", notes)


if __name__ == "__main__":
    unittest.main()
