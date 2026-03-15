import unittest

from app.api_models import AssistantMode
from app.task_router import TaskRouter, TaskType


class TaskRouterTest(unittest.TestCase):
    def test_classify_editor_request_as_proposal(self):
        router = TaskRouter()

        spec = router.classify(
            message="Dopisz to do kanonu i zaktualizuj Captain Mira.",
            requested_mode=AssistantMode.auto,
            requested_intent="auto",
            requested_artifact_type=None,
            requested_save_output=False,
            requested_output_title=None,
            candidate_text=None,
        )

        self.assertEqual(spec.task_type, TaskType.propose_doc_change)
        self.assertEqual(spec.chat_intent, "proposal")
        self.assertTrue(spec.requires_confirmation)

    def test_classify_guard_request_from_candidate_text(self):
        router = TaskRouter()

        spec = router.classify(
            message="Sprawdz ten tekst z kanonem.",
            requested_mode=AssistantMode.auto,
            requested_intent="auto",
            requested_artifact_type=None,
            requested_save_output=False,
            requested_output_title=None,
            candidate_text="Captain Mira zdradzila Red Blade.",
        )

        self.assertEqual(spec.task_type, TaskType.audit_consistency)
        self.assertEqual(spec.assistant_mode, AssistantMode.guard)

    def test_classify_creative_artifact_with_save_intent(self):
        router = TaskRouter()

        spec = router.classify(
            message="Wymysl 3 hooki na sesje i zapisz wynik.",
            requested_mode=AssistantMode.auto,
            requested_intent="auto",
            requested_artifact_type=None,
            requested_save_output=False,
            requested_output_title=None,
            candidate_text=None,
        )

        self.assertEqual(spec.task_type, TaskType.create_artifact)
        self.assertTrue(spec.save_output)
        self.assertIsNotNone(spec.output_title)

    def test_classify_character_request_as_npc_brief(self):
        router = TaskRouter()

        spec = router.classify(
            message="Wymysl mi postac piracka pasujaca do Shackles.",
            requested_mode=AssistantMode.auto,
            requested_intent="auto",
            requested_artifact_type=None,
            requested_save_output=False,
            requested_output_title=None,
            candidate_text=None,
        )

        self.assertEqual(spec.task_type, TaskType.create_artifact)
        self.assertEqual(spec.chat_intent, "creative")
        self.assertEqual(spec.artifact_type, "npc_brief")

    def test_classify_multi_character_request_as_npc_pack(self):
        router = TaskRouter()

        spec = router.classify(
            message="Wymysl mi 3 postacie do Portu Peril: wojownika pirata, maga i rybaka.",
            requested_mode=AssistantMode.auto,
            requested_intent="auto",
            requested_artifact_type=None,
            requested_save_output=False,
            requested_output_title=None,
            candidate_text=None,
        )

        self.assertEqual(spec.task_type, TaskType.create_artifact)
        self.assertEqual(spec.chat_intent, "creative")
        self.assertEqual(spec.artifact_type, "npc_pack")

    def test_classify_location_creation_request_as_location_brief(self):
        router = TaskRouter()

        spec = router.classify(
            message="Wymysl miejsce: klify nieopodal Portu Peril.",
            requested_mode=AssistantMode.auto,
            requested_intent="auto",
            requested_artifact_type=None,
            requested_save_output=False,
            requested_output_title=None,
            candidate_text=None,
        )

        self.assertEqual(spec.task_type, TaskType.create_artifact)
        self.assertEqual(spec.chat_intent, "creative")
        self.assertEqual(spec.artifact_type, "location_brief")


if __name__ == "__main__":
    unittest.main()
