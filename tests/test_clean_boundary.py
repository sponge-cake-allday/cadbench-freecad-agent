import unittest
from pathlib import Path


class CleanBoundaryTests(unittest.TestCase):
    def test_agent_has_no_official_scorer_hooks(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "harbor_agents" / "freecad_cad_agent.py").read_text(encoding="utf-8")

        forbidden = [
            "freecad_templates",
            "template_candidate_for_instruction",
            "verifier_result",
            "reward.json",
            "/opt/grader",
            "geometry_similarity",
            "cad_spec_consistency",
            "ground_truth",
        ]
        for term in forbidden:
            with self.subTest(term=term):
                self.assertNotIn(term, source)


if __name__ == "__main__":
    unittest.main()
