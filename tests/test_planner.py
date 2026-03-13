import unittest

from app.models_v2 import ProposeChangesRequest, WorldDocInfo, WorldEntityType
from app.planner import PlannerService


class PlannerServicePromptTest(unittest.TestCase):
    def test_prompt_includes_exact_folder_constraints(self):
        planner = PlannerService(generate_text_fn=lambda prompt: '{"summary":"x","user_goal":"x","needs_confirmation":true}')
        request = ProposeChangesRequest(instruction="Dodaj sekret do NPC Captain Mira")
        docs = [
            WorldDocInfo(folder="03 NPC", title="Captain Mira", doc_id="doc-1", entity_type=WorldEntityType.npc),
            WorldDocInfo(folder="06 Threads", title="Thread Tracker", doc_id="doc-2", entity_type=WorldEntityType.thread),
        ]

        prompt = planner._build_prompt(request=request, world_docs=docs, world_context="ctx")

        self.assertIn("Use only exact logical folder names", prompt)
        self.assertIn("Never invent folder names such as \"03 NPCs\"", prompt)
        self.assertIn("03 NPC", prompt)
        self.assertIn("Captain Mira", prompt)

    def test_propose_recovers_json_from_markdown_fence(self):
        planner = PlannerService(
            generate_text_fn=lambda prompt: """```json
{"summary":"ok","user_goal":"Goal","assumptions":[],"impacted_docs":[],"actions":[],"needs_confirmation":true}
```"""
        )
        request = ProposeChangesRequest(instruction="Dodaj sekret do NPC Captain Mira")

        proposal = planner.propose(request=request, world_docs=[], world_context="ctx")

        self.assertEqual(proposal.summary, "ok")
        self.assertTrue(proposal.needs_confirmation)

    def test_propose_uses_repair_pass_when_first_output_is_invalid(self):
        calls = []

        def fake_generate(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return "summary: invalid output"
            return '{"summary":"fixed","user_goal":"Goal","assumptions":[],"impacted_docs":[],"actions":[],"needs_confirmation":true}'

        planner = PlannerService(generate_text_fn=fake_generate)
        request = ProposeChangesRequest(instruction="Dodaj sekret do NPC Captain Mira")

        proposal = planner.propose(request=request, world_docs=[], world_context="ctx")

        self.assertEqual(proposal.summary, "fixed")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
