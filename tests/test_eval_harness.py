import unittest
from pathlib import Path

from app.api_models import ContinuityIssue, ContinuityReport
from app.eval_harness import EvalHarness
from app.models_v2 import ActionType, ChangeProposal, DocumentAction, DocumentRef


class EvalHarnessTest(unittest.TestCase):
    def test_load_suite_and_run(self):
        fixture = Path(__file__).parent / "fixtures" / "campaign_eval_suite.json"
        harness = EvalHarness(
            answer_fn=lambda prompt: "Morn i Black Eel sa powiazani.",
            planner_fn=lambda instruction: ChangeProposal(
                summary="Test proposal",
                user_goal=instruction,
                actions=[
                    DocumentAction(
                        action_type=ActionType.replace_section,
                        target=DocumentRef(folder="03 NPC", title="Captain Mira"),
                        section="Notes",
                        content="Aktualizacja.",
                    )
                ],
            ),
            continuity_fn=lambda message, generated_text: ContinuityReport(
                ok=False,
                issues=[
                    ContinuityIssue(
                        code="fact_conflict",
                        severity="error",
                        message="Status conflict.",
                    )
                ],
            ),
        )

        suite = harness.load_suite(fixture)
        result = harness.run_suite(suite)

        self.assertTrue(result.ok)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.passed, 3)


if __name__ == "__main__":
    unittest.main()
