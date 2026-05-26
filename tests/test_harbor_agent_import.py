import unittest


class HarborAgentImportTests(unittest.TestCase):
    def import_agent_module(self):
        try:
            import harbor  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Harbor is installed in the Conda Python used by harbor CLI, not this venv.")

        import harbor_agents.freecad_cad_agent as freecad_cad_agent

        return freecad_cad_agent

    def test_generic_import_path(self):
        freecad_cad_agent = self.import_agent_module()
        CadBenchFreeCadAgent = freecad_cad_agent.CadBenchFreeCadAgent

        self.assertEqual(
            CadBenchFreeCadAgent.import_path(),
            "harbor_agents.freecad_cad_agent:CadBenchFreeCadAgent",
        )

    def test_legacy_qwen_import_path(self):
        try:
            import harbor  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Harbor is installed in the Conda Python used by harbor CLI, not this venv.")

        from harbor_agents.qwen_cad_agent import QwenCadAgent

        self.assertEqual(
            QwenCadAgent.import_path(),
            "harbor_agents.qwen_cad_agent:QwenCadAgent",
        )

    def test_sanitize_code_removes_gui_and_normalizes_math(self):
        freecad_cad_agent = self.import_agent_module()

        code = freecad_cad_agent.sanitize_code(
            """
            import FreeCAD as App
            import PartDesignGui
            import Path
            x = App.cos(0)
            y = App.sin(0)
            p = Part.Point()
            body.Shape = feature.Solid
            pad = doc.addObject("PartDesign::AdditivePad", "Pad")
            pocket = doc.addObject('PartDesign::SubtractivePocket', 'Pocket')
            v = FreeCAD.Vector(0, 0, 0)
            """
        )
        self.assertNotIn("PartDesignGui", code)
        self.assertIn("import math", code)
        self.assertIn("import Part", code)
        self.assertIn("from pathlib import Path", code)
        self.assertIn("math.cos(0)", code)
        self.assertIn("math.sin(0)", code)
        self.assertIn("feature.Shape", code)
        self.assertIn('"PartDesign::Pad"', code)
        self.assertIn("'PartDesign::Pocket'", code)
        self.assertIn("App.Vector(0, 0, 0)", code)


if __name__ == "__main__":
    unittest.main()
